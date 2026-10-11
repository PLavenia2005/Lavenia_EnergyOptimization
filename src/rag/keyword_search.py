import pandas as pd

from rank_bm25 import BM25Okapi


class KeywordSearch:

    def __init__(
        self,
        data_path
    ):
        self.df = pd.read_csv(
            data_path
        )

        self.documents = (
            self.df["case_description"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.tokenized_documents = [
            self.tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    @staticmethod
    def tokenize(text):
        return (
            str(text)
            .lower()
            .split()
        )

    def search(
        self,
        query,
        top_k=5
    ):
        query_tokens = self.tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indexes = scores.argsort()[
            ::-1
        ][:top_k]

        results = []

        for index in ranked_indexes:

            results.append({
                "case_id": self.df.iloc[index][
                    "case_id"
                ],
                "document": self.documents[index],
                "score": float(
                    scores[index]
                )
            })

        return results