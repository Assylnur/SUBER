from environment.items_selection import ItemsSelector
from environment.citations.citation import Citation
import typing

class CitationsSelector(ItemsSelector):
    """
    Object that is responsable to select Movies, given a list of items, with predicted rating (from LLM), selects the one that the user will see.
    In this case can select more than one item. For every recommended item the user will watch it with probability p

    Attributes:
        p (float): probability to watch a item
        seed (integer): seed of the selector
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def select(
        self, items: typing.List[Citation], ratings: typing.List[float]
    ) -> typing.Tuple[typing.List[Citation], typing.List[float]]:
        """
        Given a rating by user, select paper to click on.

        Args:
            items (list of Citation): the list of items the user has to select from
            ratings (int): the list of ratings the user would give

        Return
            items: the list of items that has been recommended
            selected_ratings: the list of ratings for all recommended papers, with selected papers having 
                                rating > threshold and unselcted paper having rating 0.
        """

        selected_list = []
        selected_ratings = []

        for i in range(len(ratings)):
            if ratings[i] > self.threshold:
                selected_ratings.append(ratings[i])
            else:
                selected_ratings.append(0)

        return items, selected_ratings