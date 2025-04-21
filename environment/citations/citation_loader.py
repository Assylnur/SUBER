import pandas as pd
import ast
from environment.citations.citation import Citation
from environment.item import ItemsLoader
from abc import ABC
from typing import List

class CitationsLoader(ABC):
    def __init__(self, csv_path):
        super().__init__()
        self.data = pd.read_csv(csv_path)

    def load_all_ids(self) -> List[int]:
        return list(self.data.index)

    def load_items_from_ids(self, id_list) -> List[Citation]:
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
                quartile_cite=row['cited_by_count_quartile'],
                norm_year=row['publication_year_norm'],
                quartile_year=row['publication_year_quartile']
            )
            items.append(item)
        return items
    
    def load_items(self):
        return self.load_items_from_ids(self.load_all_ids())
