dataset_path = "environment/citations/datasets/cleaned-scientometrics-and-bibliometrics-research.csv"
user_dataset_path = "environment/citations/datasets/users.json"

env_config = {
    "use_normalized": True,
    "max_recommendations": 5,
    "reward_strategy": "novelty_weighted"
}
