from environment.users.citation_user import CitationUser
import json
from abc import ABC
from typing import List

class CitationUsersLoader(ABC):
    def __init__(self, json_path):
        super().__init__()
        with open(json_path) as f:
            user_data = json.load(f)
        self.users = []
        for entry in user_data:
            user = CitationUser(
                interests=entry['preferred_topics'],
                relevance_threshold=entry['relevance_threshold'],
                reputability_bias=entry['reputability_bias'],
                novelty_bias=entry['novelty_preference']
            )
            self.users.append(user)

    def get_users(self) -> List[CitationUser]:
        return self.users
