from environment.item import Item

class Citation(Item):
    def __init__(self, id, title, year, topics, topic_scores, cited_by_count, norm_cite, norm_year):
        super().__init__(id=id, display_name=title)
        self.title = title
        self.year = year
        self.topics = topics
        self.topic_scores = topic_scores
        self.cited_by_count = cited_by_count
        self.norm_cite = norm_cite
        self.norm_year = norm_year

        # ✅ Add this line to avoid crashes
        self.vote_average = self.norm_cite  # or norm_year, or a custom calculation

    def __str__(self):
        return f"Title: {self.title} - publication year: {self.year} - topics: {self.topics} - topic scores = {self.topic_scores} - norm_cite: {self.norm_cite} - norm_year: {self.norm_year}"
