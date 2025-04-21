class CitationUser:
    """
    Object to represent users
    """

    id_counter = 0

    def __init__(self, interests=[None], relevance_threshold=0.5, reputability_bias=0.5, novelty_bias=0.5):
        self.id = CitationUser.id_counter
        self.interests = interests
        self.relevance_threshold = relevance_threshold
        self.reputability_bias = reputability_bias
        self.novelty_bias = novelty_bias

        CitationUser.id_counter += 1

    def __str__(self) -> str:
        return f"👧🏻 id = {self.id}\ninterests={self.interests}\nrelevance_threshold={self.relevance_threshold}\nreputability_bias={self.reputability_bias}\nnovelty_bias={self.novelty_bias}"

    def __repr__(self) -> str:
        return f"👧🏻 id = {self.id}\ninterests={self.interests}\nrelevance_threshold={self.relevance_threshold}\nreputability_bias={self.reputability_bias}\nnovelty_bias={self.novelty_bias}"

    @staticmethod
    def get_num_users():
        return CitationUser.id_counter

