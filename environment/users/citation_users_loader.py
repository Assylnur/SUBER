from environment.users import UsersLoader
from environment.users.user import User
import json

class CitationUsersLoader(UsersLoader):
    def __init__(self, json_path):
        super().__init__()
        with open(json_path) as f:
            user_data = json.load(f)
        self.users = []
        for entry in user_data:
            description = (
                "Interested in: " + ", ".join(entry["preferred_topics"]) +
                f". Prefers novelty: {entry['novelty_preference']}, reputability bias: {entry['reputability_bias']}"
            )
            user = User(
                name=f"CitationUser{entry['id']}",
                gender="M",
                age=30,
                description=description
            )
            self.users.append(user)

    def get_users(self):
        return self.users
