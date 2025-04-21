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
        debug_path: str,
        items_loader: ItemsLoader,
        user_list: List[CitationUser],
        items_selector: ItemsSelector,
        reward_shaping: CitationRewardReshapingExpDecay,
        render_path: str = "./tmp/render/",
        debug: bool = False,
    ):

        self.render_mode = render_mode
        self.render_path = render_path
        self.metadata = {"render_modes": ["human", "csv"]}

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
        self.item_dim = 384 + 4     # test embedding + year + citation + user topic + interact nums
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
            # "quartile_year": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            # "quartile_cite": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            # "user_topic_embedding": spaces.Box(low=-1.0, high=1.0, shape=(384 * 3,), dtype=np.float32),
            "history_embedding":   spaces.Box(-np.inf, np.inf, shape=(self.hidden_dim,),  dtype=np.float32),
            # "topics_embedding": spaces.Box(low=-1.0, high=1.0, shape=(384 * self.num_items,), dtype=np.float32),
            # "interact_nums": spaces.Box(low=-1.0, high=1.0, shape=(self.num_items,), dtype=np.float32),
            # "click_embedding":  spaces.Box(low=-1.0, high=1.0, shape=(self.num_items,), dtype=np.float32),
            # "item_topic_embedding": spaces.Box(low=-1.0, high=1.0, shape=(384 * 3,), dtype=np.float32),
        })

        self.topics_embedding = np.zeros((384 * self.num_items))
        self.click_embedding = np.zeros((self.num_items))

        self.action_space = spaces.Discrete(self.num_items)

    def reset(self,  seed: Optional[int] = None, options: Optional[dict] = None, user_id=None):
        super().reset(seed=seed)
        self.clean_memory()
        self.num_step = 0
        # self.recommended_paper = set()

        if user_id is None:
            user_id = self.np_random.integers(low=0, high=self.num_users)

        self._user = self.user_list[user_id]
        print(self._user)
        self._items_interact = tuple()
        self._items_click = tuple()

        # ✅ Embed user's research description
        self.user_topic_embedding = self.bert_model.encode(self._user.interests)

        zero_feat = np.zeros(self.item_dim, dtype=np.float32)
        self.history_buffer = deque([zero_feat] * self.window_size, maxlen=self.window_size)

        info = {
            "user_id": self._user.id
        }

        return self._get_obs(), info

    def step(self, action: int):
        self.num_step += 1
        item = self.items_loader.load_items_from_ids(id_list=[action])[0]
        print(item)
        user_rating = self.rater.rate(self._user, item)
        click = 1 if random.random() < user_rating else 0
        print(f"{user_rating} >> {'clicked' if click else 'ignored'}")

        item_emb = self.bert_model.encode(item.topics).astype(np.float32).mean(axis=0)   # (384,)
        quart_year = np.array([item.quartile_year], dtype=np.float32)
        quart_cite = np.array([item.quartile_cite], dtype=np.float32)
        interact_num = np.array([self._items_interact.count(action)], dtype=np.float32)
        click_arr = np.array([click], dtype=np.float32)

        step_feat = np.concatenate([
            item_emb, quart_year, quart_cite,
            interact_num, click_arr
        ], axis=0)

        self.history_buffer.append(step_feat)

        self._items_interact = self._items_interact + (action,)
        self._items_click = self._items_click + (click,)

        reward = user_rating if click else 0
        self.memory.update_memory(self._user.id, [action], [reward])

        # Go until at least 5 clicks or randomize, whichever comes later
        if self._items_click.count(1) < 5:
            terminated = False
        elif len(self._items_click) == 50:
            terminated = True
        else:
            terminated = self.np_random.choice([True, False], p=[0.025, 0.975])

        if terminated:
            print(f"🎯 User {self._user.id} - End episode with {self._items_click.count(1)}/{len(self._items_click)} clicks - avg = {self._items_click.count(1)/len(self._items_click)}.")
        observation = self._get_obs()

        item_interactions = self.memory.user_to_shown_papers[self._user.id][action]
        reward, _ = self.reward_shaping.reshape(item_interactions, reward)

        info = {
            "user_id": self._user.id
        }

        return observation, reward, terminated, False, info

    def _get_obs(self):
        """
        State contains:
            1. the embeddings of user interction history
            2. whether or not the paper is clicked by the user
        """

        # interacted_paper = self.items_loader.load_items_from_ids(self._items_interact)
        # topics_embedding = [paper.topics for paper in interacted_paper]
        # topics_embedding = np.concatenate(list(self.bert_model.encode(topics_embedding)), axis=0)
        # interact_nums = [self._items_interact[:i+1].count(id) for idx, id in enumerate(self._items_interact)]
        # years = [paper.quartile_year for paper in interacted_paper]
        # cites = [paper.quartile_cite for paper in interacted_paper]
        # click_embedding = np.array(self._items_click, dtype=np.float32)

        # item_seq  = np.stack(self.item_emb_buffer, axis=0)          # (window, 384)
        # click_seq = np.array(self.click_buffer, dtype=np.float32)  # (window,)

        # # convert & add batch dim
        # item_tensor  = torch.from_numpy(item_seq)[None,...]               # (1, window, 384)
        # click_tensor = torch.from_numpy(click_seq.reshape(self.window_size,1))[None,...]  # (1, window, 1)

        # # run through the encoder
        # with torch.no_grad():
        #     hist_emb = self.encoder(item_tensor, click_tensor)  # (1, hidden_dim)
        # hist_emb = hist_emb.squeeze(0).cpu().numpy()            # (hidden_dim,)

        seq = np.stack(self.history_buffer, axis=0)          # (window, step_feat_dim)
        seq_tensor = torch.from_numpy(seq)[None, ...]      # (1, window, step_feat_dim)

        with torch.no_grad():
            hist_emb = self.encoder(seq_tensor)

        hist_emb = hist_emb.squeeze(0).cpu().numpy()

        # obs = {
        #     "history_embedding": hist_emb,
        # }

        # obs = {
        #     "quartile_year": np.array(years, dtype=np.float32),
        #     "quartile_cite": np.array(cites, dtype=np.float32),
        #     "user_topic_embedding": np.concatenate(list(self.user_topic_embedding), axis=0),
        #     "history_embedding": hist_emb.astype(np.float32),
        #     # "topics_embedding": topics_embedding,
        #     "interact_nums": interact_nums,
        #     # "click_embedding":  click_embedding,
        #     # "item_topic_embedding": np.zeros(384, dtype=np.float32),
        # }

        return {"history_embedding": hist_emb}

    def clean_memory(self):
        self.memory = CitationMemory(self.items_loader)


# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces

class FlatObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # ✅ Define flat observation space including embeddings
        self.observation_space = spaces.Dict({
            "history_embedding": spaces.Box(-np.inf, np.inf, shape=(self.hidden_dim,),  dtype=np.float32),
        })
        # self.observation_space = spaces.Dict({
        #     "item_norm_year": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        #     "item_norm_cite": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        #     "user_topic_embedding": spaces.Box(low=-1.0, high=1.0, shape=(384,), dtype=np.float32),
        #     # "item_topic_embedding": spaces.Box(low=-1.0, high=1.0, shape=(384,), dtype=np.float32),
        # })

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
            "history_embedding": obs["history_embedding"]
            # "item_norm_year": np.array(obs["item_norm_year"], dtype=np.float32),
            # "item_norm_cite": np.array(obs["item_norm_cite"], dtype=np.float32),
            # "user_topic_embedding": np.array(obs["user_topic_embedding"], dtype=np.float32),
            # "item_topic_embedding": np.array(obs["item_topic_embedding"], dtype=np.float32),
        }
