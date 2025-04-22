import os
import string
from functools import reduce
from collections import deque
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import random
from gymnasium import spaces
from sentence_transformers import SentenceTransformer
from typing import Optional, List

from environment.item import ItemsLoader
from environment.items_retrieval import ItemsRetrieval
from environment.items_selection import ItemsSelector
from environment.LLM import LLMRater
from environment.citation_memory import CitationMemory
from environment.reward_perturbator import RewardPerturbator
from environment.citation_reward_shaping import CitationRewardReshapingExpDecay
from environment.users import UsersLoader
from environment.users.citation_user import CitationUser
from environment.users.user_rater import RuleBasedCitationRater

import torch
import torch.nn as nn

# Debugging
import sys


class HistoryRNNEncoder(nn.Module):
    """
    Encodes a sequence of (item_embedding, click_flag) pairs
    into a single fixed‑size vector via a GRU.
    """

    def __init__(self, input_dim: int, hidden_dim: int, **gru_kwargs):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            **gru_kwargs
        )
        self.hidden_dim = hidden_dim

    def forward(self, seq: torch.Tensor):
        # seq: (batch, seq_len, input_dim)
        out, _ = self.gru(seq)              
        return out[:, -1, :]

class UserEnv(gym.Env):
    def __init__(
        self,
        render_mode: str,
        items_loader: ItemsLoader,
        user_list: List[CitationUser],
        items_selector: ItemsSelector,
        reward_shaping: CitationRewardReshapingExpDecay,
        render_path: str = "./tmp/render/",
        debug: bool = False,
        debug_path: Optional[str] = None,
        max_interactions: int = 50
    ):

        self.render_mode = render_mode
        self.render_path = render_path
        self.metadata = {"render_modes": ["human", "csv"]}
        self.max_interactions = max_interactions

        self.debug = debug
        self.debug_path = debug_path
        if self.debug:
            sys.stdout = open(self.debug_path, "w", buffering=1)

        self.rater = RuleBasedCitationRater()

        # self.users_loader = users_loader
        # self.user_list = self.users_loader.get_users()
        self.user_list = user_list
        self.num_users = len(self.user_list)

        self.items_loader = items_loader
        self.item_ids = self.items_loader.load_all_ids()
        self.num_items = len(self.item_ids)

        self.memory = CitationMemory(self.items_loader)
        self.items_selector = items_selector
        self.reward_shaping = reward_shaping

        self.num_step = 0

        self.window_size = 20
        self.item_dim = 384     # test embedding
        self.hidden_dim = 256

        self.encoder = HistoryRNNEncoder(
            input_dim=self.item_dim,
            hidden_dim=self.hidden_dim
        )
        
        self.history_buffer = deque(maxlen=self.window_size)

        # ✅ BERT model for embeddings
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')

        # ✅ Updated observation space
        self.observation_space = spaces.Dict({
            "history_embedding": spaces.Box(-np.inf, np.inf, shape=(self.hidden_dim,),  dtype=np.float32),
            "user_topics_embedding": spaces.Box(-np.inf, np.inf, shape=(3, 384), dtype=np.float32),
            "interactions": spaces.Box(0, self.max_interactions, shape=(1,), dtype=np.int32),   # num interaction for current paper
            "publication_year_quar": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
            "citation_quar": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
            "click_array": spaces.Box(0, 1, shape=(1, self.max_interactions), dtype=np.int32)
        })

        # self.user_topics_embedding = np.zeros((384 * self.num_items))
        self.click_embedding = np.zeros((self.num_items))

        self.action_space = spaces.Discrete(self.num_items)

    def reset(self,  seed: Optional[int] = None, options: Optional[dict] = None, user_id=None):
        super().reset(seed=seed)
        # self.clean_memory()
        
        # self.recommended_paper = set()

        if user_id is None:
            user_id = self.np_random.integers(low=0, high=self.num_users)

        self._user = [user for user in self.user_list if user.id == user_id][0]
        topics_embedding = self.bert_model.encode(self._user.interests)
        if topics_embedding.shape[0] < 3:
            pad = np.zeros((3 - topics_embedding.shape[0], topics_embedding.shape[1]), dtype=topics_embedding.dtype)
            topics_embedding = np.concatenate([topics_embedding, pad], axis=0)
        self.user_topics_embedding = topics_embedding
        # print(self._user)
        self._items_interact = tuple()
        self._items_click = tuple()

        zero_feat = np.zeros(self.item_dim, dtype=np.float32)
        self.history_buffer = deque([zero_feat] * self.window_size, maxlen=self.window_size)

        info = {
            "user_id": self._user.id,
            "item_interact": self._items_interact,
            "item_clicks": self._items_click
        }

        # if self.num_step > 500:
        #     self.num_step = 0
        #     self.clear_memory()

        return self._get_obs(), info

    def step(self, action: int):
        self.num_step += 1
        item = self.items_loader.load_items_from_ids(id_list=[action])[0]
        # print(item)
        user_rating = self.rater.rate(self._user, item)
        # click = 1 if random.random() < user_rating else 0
        if action in self._items_interact:
            click = 0
        else:
            click = 1 if user_rating > 0.5 else 0
        # print(f"Rating: {user_rating} >> {'clicked' if click else 'ignored'}")

        item_emb = self.bert_model.encode(item.topics).astype(np.float32).mean(axis=0)   # (384,)
        # quart_year = np.array([item.quartile_year], dtype=np.float32)
        # quart_cite = np.array([item.quartile_cite], dtype=np.float32)
        # click_arr = np.array([click], dtype=np.float32)

        # step_feat = np.concatenate([
        #     item_emb, quart_year, quart_cite, click_arr
        # ], axis=0)

        self.history_buffer.append(item_emb)

        self._items_interact = self._items_interact + (action,)
        self._items_click = self._items_click + (click,)

        reward = user_rating * 2 if click else -0.5
        self.memory.update_memory(self._user.id, [action], [reward])

        # Go until at least 5 clicks or randomize, whichever comes later
        if len(self._items_click) == self.max_interactions:
            terminated = True
        else:
            terminated = False

        observation = self._get_obs(action=action)

        item_interactions = self.memory.user_to_shown_papers[self._user.id][action]
        reward, _ = self.reward_shaping.reshape(item_interactions, reward)

        info = {
            "user_id": self._user.id,
            "item_interact": self._items_interact,
            "item_clicks": self._items_click
        }

        if terminated:
            print(f"🎯 User {self._user.id} - End episode with {self._items_click.count(1)}/{len(self._items_click)} clicks - avg = {self._items_click.count(1)/len(self._items_click)}.")
        print(f"user = {self._user.id} - action = {action} - rating = {user_rating} - click = {click} - interact_num = {observation['interactions']} - reward = {reward}")

        return observation, reward, terminated, False, info

    def _get_obs(self, action=None):
        """
        State contains:
            1. the embeddings of user interction history
            2. whether or not the paper is clicked by the user
        """


        seq = np.stack(self.history_buffer, axis=0)          # (window, step_feat_dim)
        seq_tensor = torch.from_numpy(seq)[None, ...]      # (1, window, step_feat_dim)

        with torch.no_grad():
            hist_emb = self.encoder(seq_tensor)

        hist_emb = hist_emb.squeeze(0).cpu().numpy()

        if action:
            interaction = self.memory.get_num_interaction(user_id=self._user.id, item_id=action)
            paper = self.items_loader.load_items_from_ids([action])[0]
            items_click = np.array(self._items_click)
            items_click = np.pad(items_click, (0, self.max_interactions - items_click.shape[0]), mode="constant", constant_values=0)

            obs = {
                "history_embedding": hist_emb,
                "user_topics_embedding": self.user_topics_embedding,
                "interactions": interaction,
                "publication_year_quar": paper.quartile_year,
                "citation_quar": paper.quartile_cite,
                "click_array": items_click
            }

        else:
            obs = {
                "history_embedding": hist_emb,
                "user_topics_embedding": self.user_topics_embedding,
                "interactions": 0,
                "publication_year_quar": -1,
                "citation_quar": -1,
                "click_array": np.zeros((1, self.max_interactions))
            }

        return obs

    def clean_memory(self):
        self.memory = CitationMemory(self.items_loader)


class FlatObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # ✅ Define flat observation space including embeddings
        self.observation_space = spaces.Dict({
            "history_embedding": spaces.Box(-np.inf, np.inf, shape=(self.hidden_dim,),  dtype=np.float32),
            "user_topics_embedding": spaces.Box(-np.inf, np.inf, shape=(3, 384), dtype=np.float32),
            "interactions": spaces.Box(0, 50, shape=(1,), dtype=np.int32),
            "publication_year_quar": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "citation_quar": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "click_array": spaces.Box(0, 1, shape=(1, self.max_interactions), dtype=np.int32)
        })

        self.action_space = env.action_space  # Delegate to base env

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform_obs(obs), reward, terminated, truncated, info

    def _transform_obs(self, obs):
        # No major transformation needed, just fix data types if necessary
        return {
            "history_embedding": obs["history_embedding"],
            "user_topics_embedding": obs["user_topics_embedding"],
            "interactions": obs["interactions"],
            "publication_year_quar": obs["publication_year_quar"],
            "citation_quar": obs["citation_quar"],
            "click_array": obs["click_array"]
        }
