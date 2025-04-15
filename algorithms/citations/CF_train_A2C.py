import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

from environment.env import Simulatio4RecSys
from environment.reward_shaping import RewardShaping
from environment.reward_perturbator import RewardPerturbator
from environment.items_selection import ItemsSelector

from environment.LLM.dummy_llm import DummyLLM 
from environment.LLM.rater import LLMRater  

from environment.citations.citations_loader import CitationsLoader
from environment.citations.citations_retrieval import CitationsRetrieval
from environment.users.citation_users_loader import CitationUsersLoader

csv_path = "environment/citations/datasets/cleaned-scientometrics-and-bibliometrics-research.csv"
users_path = "environment/users/datasets/citation_users.json"

# Load data
items_loader = CitationsLoader(csv_path)
users_loader = CitationUsersLoader(users_path)

# LLM + Rater
llm = DummyLLM()
llm_rater = LLMRater(
    llm=llm,
    current_items_features_list=[],     # Can fill with embeddings later
    previous_items_features_list=[],
    llm_render=False,
    llm_query_explanation=False
)

# Environment
env = Simulatio4RecSys(
    render_mode="human",
    items_loader=items_loader,
    users_loader=users_loader,
    items_selector=ItemsSelector(),
    reward_perturbator=RewardPerturbator(),
    items_retrieval=CitationsRetrieval(),
    reward_shaping=RewardShaping(),
    llm_rater=llm_rater,
)

# Train
model = A2C("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save
model.save("models/a2c_citation_recommender")
print("✅ Training complete and model saved.")
