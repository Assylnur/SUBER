from environment.items_retrieval import ItemsRetrieval

class CitationsRetrieval(ItemsRetrieval):
    def retrieve(self, curr_item, past_items, past_interactions):
        # Basic strategy: return 5 most recent or most cited
        sorted_items = sorted(
            past_items,
            key=lambda x: x.year if hasattr(x, 'year') else 0,
            reverse=True
        )
        return sorted_items[:5], past_interactions[:5]
