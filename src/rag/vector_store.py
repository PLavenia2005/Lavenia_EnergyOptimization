from pathlib import Path

import chromadb

from src.utils.config import CHROMA_DB_PATH


class VectorStore:

    def __init__(
        self,
        collection_name="energy_cases"
    ):
        Path(CHROMA_DB_PATH).mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": (
                        "Energy consumption "
                        "historical cases"
                    )
                }
            )
        )

    def add_documents(
        self,
        documents,
        embeddings,
        metadatas,
        ids
    ):
        batch_size = 5000

        total = len(documents)

        for start in range(
            0,
            total,
            batch_size
        ):
            end = min(
                start + batch_size,
                total
            )

            print(
                f"Adding documents "
                f"{start + 1}-{end} "
                f"of {total}..."
            )

            self.collection.upsert(
                documents=documents[start:end],
                embeddings=embeddings[
                    start:end
                ].tolist(),
                metadatas=metadatas[
                    start:end
                ],
                ids=ids[
                    start:end
                ]
            )

    def search(
        self,
        query_embedding,
        top_k=5,
        where=None
    ):
        kwargs = {
            "query_embeddings": [
                query_embedding.tolist()
            ],
            "n_results": top_k
        }

        if where:
            kwargs["where"] = where

        return self.collection.query(
            **kwargs
        )

    def count(self):
        return self.collection.count()