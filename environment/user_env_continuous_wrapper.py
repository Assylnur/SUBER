import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from sklearn.neighbors import NearestNeighbors

class UserEnvContinuousWrapper(gym.Wrapper):
    def __init__(self, env, embedding_dim: int = 384):
        super().__init__(env)
        self.embedding_dim = embedding_dim

        # 1) Build per‐item vectors = [topic_emb(384), year_norm(1), cite_norm(1)]
        items = env.items_loader.load_items()  # list of Citation objects
        vecs = []
        for item in items:
            # topic embedding
            t = env.bert_model.encode(item.topics).mean(axis=0).astype(np.float32)
            # u = env.bert_model.encode(env._user.interests).mean(axis=0).astype(np.float32)
            # normalized year & citation already in [0,1]
            y = np.array([item.quartile_year],  dtype=np.float32)
            c = np.array([item.quartile_cite], dtype=np.float32)
            vecs.append(np.concatenate([t, y, c], axis=0))
        self.item_vecs = np.vstack(vecs)  # shape (N_items, embed_dim+2)

        # build nearest‐neighbor index
        self.nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        self.nn.fit(self.item_vecs)

        # 2) override action_space to Box(embed_dim+2)
        low  = np.concatenate([-1.0*np.ones(self.embedding_dim), [0.0], [0.0]])
        high = np.concatenate([ 1.0*np.ones(self.embedding_dim), [1.0], [1.0]])
        self.action_space = Box(low=low, high=high, dtype=np.float32)

        # keep the discrete for debugging if you like
        self.discrete_action_space = env.action_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray):
        # action is now length embed_dim+2
        idx = self.nn.kneighbors(action.reshape(1,-1), return_distance=False)[0,0]
        # call underlying env with the discrete index
        return self.env.step(int(idx))