from environment.item import Item

class Citation(Item):
    def __init__(self, id, title, year, topics, topic_scores, cited_by_count, norm_cite, quartile_cite, norm_year, quartile_year):
        super().__init__(id=id, display_name=title)
        self.id = id
        self.title = title
        self.year = year
        self.topics = topics
        self.topic_scores = topic_scores
        self.cited_by_count = cited_by_count
        self.norm_cite = norm_cite
        self.quartile_cite = quartile_cite
        self.norm_year = norm_year
        self.quartile_year = quartile_year

        # ✅ Add this line to avoid crashes
        self.vote_average = self.norm_cite  # or norm_year, or a custom calculation

    def __str__(self):
        return f"📄 Title: {self.title}\npublication year: {self.year}\ntopics: {self.topics}\ntopic scores = {self.topic_scores}\nquartile_cite: {self.quartile_cite}\nquartile_year: {self.quartile_year}"

    def __eq__(self, other_item: object) -> bool:
        """
        Define equality between two items if they have the same id
        """
        return self.id == other_item.id