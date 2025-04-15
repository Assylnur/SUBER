import pandas as pd
import ast
from environment.citations.citation import Citation
from environment.item import ItemsLoader

class CitationsLoader(ItemsLoader):
    def __init__(self, csv_path):
        super().__init__(name_dataset="citations")
        self.data = pd.read_csv(csv_path)

    def load_all_ids(self):
        return list(self.data.index)

    def load_items_from_ids(self, id_list):
        items = []
        for idx in id_list:
            row = self.data.iloc[idx]
            topics = ast.literal_eval(row['topics.display_name']) if isinstance(row['topics.display_name'], str) else []
            scores = ast.literal_eval(row['topics.score']) if isinstance(row['topics.score'], str) else []
            item = Citation(
                id=idx,
                title=row['title'],
                year=row['publication_year'],
                topics=topics,
                topic_scores=scores,
                cited_by_count=row['cited_by_count'],
                norm_cite=row['cited_by_count_norm'],
                norm_year=row['publication_year_norm']
            )
            items.append(item)
        return items
    
    def load_items(self):
        return self.load_items_from_ids(self.load_all_ids())
