import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FlatObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Define a flat observation space based on transformed fields
        self.observation_space = spaces.Dict({
            "user_gender": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            "user_age": spaces.Box(low=0, high=120, shape=(1,), dtype=np.int32),
            "novelty_preference": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "reputability_bias": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        self.action_space = env.action_space  # delegate to base env

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform_obs(obs), reward, terminated, truncated, info

    def _transform_obs(self, obs):
        # Gender: "M" → 0, "F" → 1
        gender = 0 if obs["user_gender"] == "M" else 1
        age = int(obs["user_age"][0]) if isinstance(obs["user_age"], np.ndarray) else obs["user_age"]

        # Extract float preferences from description
        try:
            novelty = float(obs["user_description"].split("novelty: ")[1].split(",")[0])
            reputability = float(obs["user_description"].split("reputability bias: ")[1])
        except:
            novelty, reputability = 0.5, 0.5  # default

        return {
            "user_gender": np.array([gender], dtype=np.int32),
            "user_age": np.array([age], dtype=np.int32),
            "novelty_preference": np.array([novelty], dtype=np.float32),
            "reputability_bias": np.array([reputability], dtype=np.float32)
        }
