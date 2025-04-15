import pandas as pd
import ast
import json
import random
import os
from collections import Counter


def extract_top_topics(csv_path, top_n=30):
    df = pd.read_csv(csv_path)
    all_topics = []

    for tlist in df['topics.display_name']:
        if isinstance(tlist, str):
            try:
                all_topics.extend(ast.literal_eval(tlist))
            except:
                pass  # skip bad rows

    most_common = [t for t, _ in Counter(all_topics).most_common(top_n)]
    return most_common


def generate_users(num_users, topic_pool):
    users = []
    for i in range(num_users):
        users.append({
            "id": i,
            "preferred_topics": random.sample(topic_pool, k=random.randint(1, 3)),
            "novelty_preference": round(random.uniform(0.2, 1.0), 2),
            "reputability_bias": round(random.uniform(0.2, 1.0), 2)
        })
    return users


def main():
    input_csv = "environment/citations/datasets/cleaned-scientometrics-and-bibliometrics-research.csv"
    output_json = "environment/users/datasets/citation_users.json"

    # ensure output folder exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    topics = extract_top_topics(input_csv)
    users = generate_users(num_users=100, topic_pool=topics)

    with open(output_json, "w") as f:
        json.dump(users, f, indent=2)

    print(f"Saved {len(users)} users to: {output_json}")


if __name__ == "__main__":
    main()
