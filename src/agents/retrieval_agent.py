from src.rag.hybrid_search import HybridSearch


class RetrievalAgent:

    def __init__(self):

        self.search_engine = (
            HybridSearch()
        )

    def retrieve(
        self,
        query,
        top_k=5,
        filters=None
    ):

        results = self.search_engine.search(
            query=query,
            top_k=top_k,
            filters=filters
        )

        return {
            "query": query,
            "filters": filters,
            "results": results,
            "count": len(results)
        }