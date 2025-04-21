from sentence_transformers import SentenceTransformer, util
import numpy as np

class RuleBasedCitationRater:

    def __init__(self, bert_model=None):
        """
        Rule-Based Citation Rater.
        """
        super().__init__()
        
        if bert_model is None:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # default lightweight model
        else:
            self.bert_model = bert_model

    def compute_semantic_topic_score(self, user, item, alpha=0.8):
        """
        Compute cosine similarity between user interests and paper (title + topics).
        """

        matched_score = sum([
            score for topic, score in zip(item.topics, item.topic_scores)
            if topic in user.interests
        ])

        if not matched_score:
            return 0.0

        normalized_score = matched_score / len(user.interests)
        return min((1 - np.exp(-alpha * normalized_score)) * 2, 1)

    def compute_novelty_score(self, novelty_bias, item):
        """
        Higher score if normalized year matches user's novelty preference.
        """
        return novelty_bias * item.quartile_year
        return 1.0 - abs(novelty_bias - item.quartile_year)

    def compute_reputability_score(self, reputability_bias, item):
        """
        Higher score if normalized citations match user's reputability bias.
        """
        return 1 - reputability_bias * (1 - item.quartile_cite)
        # return reputability_bias * item.quartile_cite
        # return 1.0 - abs(reputability_bias - item.quartile_cite)

    def rate(self, user, item):
        """
        Rating each recommended paper based on user preference and bias
        """

        topic_score = self.compute_semantic_topic_score(user, item)
        if topic_score > user.relevance_threshold:
            novelty_score = self.compute_novelty_score(user.novelty_bias, item)
            reputability_score = self.compute_reputability_score(user.reputability_bias, item)
        else:
            novelty_score = 0
            reputability_score = 0
        # print(f"Topic score: {topic_score}, Novelty score: {novelty_score}, Reputability score: {reputability_score}")
        final_score = (
            topic_score +
            novelty_score +
            reputability_score
        ) / 3
        print(f"Final score: ({topic_score} + {novelty_score} + {reputability_score}) / 3 = {final_score}")

        # We return rating, empty explanation, empty prompt (for compatibility)
        return final_score
