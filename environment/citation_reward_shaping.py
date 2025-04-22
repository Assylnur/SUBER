from environment.citation_memory import UserPaperInteraction
import typing
import math

class CitationRewardReshapingExpDecay():
    def __init__(self, q=0.1):
        self.q = q
        super().__init__()

    def reshape(
        self, item_interactions: typing.List[UserPaperInteraction], raw_reward: float
    ) -> float:

        if len(item_interactions) == 1:
            return raw_reward, False

        # print(f"item interactions = {item_interactions}")
        num_clicks = item_interactions[-1].num_clicks

        if raw_reward < 0:
            min_penalty, max_penalty = 0.5, 1.0
            # compute magnitude in [0.5,1.0), growing with count
            mag = min_penalty + (max_penalty - min_penalty) * (1 - self.q**(num_clicks - 1))
            return -mag, False

        decay = self.q ** (num_clicks - 1)

        return (
            raw_reward * decay -0.5 * (1 - decay),
            # raw_reward * math.pow(self.q, float(num_clicks)),
            False,
        )