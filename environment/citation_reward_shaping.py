from environment.citation_memory import UserPaperInteraction
import typing
import math

class CitationRewardReshapingExpDecay():
    def __init__(self, q=0.1):
        self.q = q
        super().__init__()

    def reshape(
        self, item_interactions: typing.List[UserPaperInteraction], rating: float
    ) -> float:

        if len(item_interactions) == 1:
            return rating, False

        print(f"item interactions = {item_interactions}")
        num_clicks = item_interactions[-1].num_clicks

        return (
            rating * math.pow(self.q, float(num_clicks)),
            False,
        )