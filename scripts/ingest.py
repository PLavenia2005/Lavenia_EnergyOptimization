import pandas as pd

from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore

from src.utils.config import (
    PROCESSED_DATA_PATH
)


def create_metadata(row):

    return {
        "case_id": str(
            row["case_id"]
        ),
        "timestamp": str(
            row["timestamp"]
        ),
        "season": str(
            row["season"]
        ),
        "period": str(
            row["period"]
        ),
        "consumption_level": str(
            row["consumption_level"]
        ),
        "is_anomaly": str(
            row["is_anomaly"]
        ),
        "hour": int(
            row["hour"]
        )
    }


def main():

    print(
        "Loading processed energy cases..."
    )

    df = pd.read_csv(
        PROCESSED_DATA_PATH
    )

    print(
        f"Total cases: {len(df)}"
    )

    documents = (
        df["case_description"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    ids = (
        df["case_id"]
        .astype(str)
        .tolist()
    )

    metadatas = [
        create_metadata(row)
        for _, row in df.iterrows()
    ]

    print(
        "Creating embeddings..."
    )

    embedding_model = (
        EmbeddingModel()
    )

    embeddings = (
        embedding_model.encode(
            documents
        )
    )

    print(
        "Connecting to ChromaDB..."
    )

    vector_store = VectorStore()

    print(
        "Adding documents to ChromaDB..."
    )

    vector_store.add_documents(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(
        "\nIngestion completed successfully."
    )

    print(
        f"ChromaDB document count: "
        f"{vector_store.count()}"
    )


if __name__ == "__main__":
    main()