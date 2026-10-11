
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.keyword_search import KeywordSearch

from src.utils.config import (
    PROCESSED_DATA_PATH,
    DEFAULT_TOP_K
)


class HybridSearch:

    def __init__(self):

        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

        self.keyword_search = KeywordSearch(
            PROCESSED_DATA_PATH
        )

    def search(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        filters=None
    ):

        # -------------------------------------------------
        # Infer filters from explicit query details
        # -------------------------------------------------

        effective_filters = dict(filters or {})
        inferred_filters = self._infer_query_filters(query)

        # Explicit caller-provided filters take precedence.
        for key, value in inferred_filters.items():
            if effective_filters.get(key) is None:
                effective_filters[key] = value

        # -------------------------------------------------
        # Semantic Search
        # -------------------------------------------------

        query_embedding = self.embedding_model.encode(
            [query]
        )[0]

        semantic_results = self.vector_store.search(
            query_embedding,
            top_k=top_k * 3,
            where=self._build_filter(effective_filters)
        )

        semantic_cases = {}

        if semantic_results["ids"]:

            ids = semantic_results["ids"][0]
            documents = semantic_results["documents"][0]
            distances = semantic_results["distances"][0]
            metadatas = semantic_results["metadatas"][0]

            for i, case_id in enumerate(ids):

                distance = float(distances[i])

                semantic_score = (
                    1.0 / (1.0 + distance)
                )

                semantic_cases[case_id] = {
                    "case_id": case_id,
                    "document": documents[i],
                    "semantic_score": semantic_score,
                    "metadata": metadatas[i]
                }

        # -------------------------------------------------
        # Keyword Search
        # -------------------------------------------------

        keyword_results = self.keyword_search.search(
            query,
            top_k=top_k * 3
        )

        keyword_scores = {}

        if keyword_results:

            max_score = max(
                result["score"]
                for result in keyword_results
            )

            for result in keyword_results:

                if max_score > 0:
                    normalized_score = (
                        result["score"] / max_score
                    )
                else:
                    normalized_score = 0.0

                keyword_scores[
                    result["case_id"]
                ] = normalized_score

        # -------------------------------------------------
        # Combine Semantic + Keyword Results
        # -------------------------------------------------

        combined = {}

        for case_id, result in semantic_cases.items():

            keyword_score = keyword_scores.get(
                case_id, 0.0
            )

            # Preserve the 70% semantic + 30% keyword design.
            hybrid_score = (
                0.70 * result["semantic_score"]
                + 0.30 * keyword_score
            )

            combined[case_id] = {
                "case_id": case_id,
                "document": result["document"],
                "semantic_score": round(
                    result["semantic_score"], 4
                ),
                "keyword_score": round(
                    keyword_score, 4
                ),
                "hybrid_score": round(
                    hybrid_score, 4
                ),
                "metadata": result["metadata"]
            }

        # -------------------------------------------------
        # Add Keyword-Only Results
        # -------------------------------------------------

        for case_id, score in keyword_scores.items():

            if case_id in combined:
                continue

            row = self.keyword_search.df[
                self.keyword_search.df["case_id"] == case_id
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            metadata = {
                "season": str(row["season"]),
                "period": str(row["period"]),
                "consumption_level": str(
                    row["consumption_level"]
                ),
                "is_anomaly": str(row["is_anomaly"])
            }

            if not self._matches_filter(
                metadata,
                effective_filters
            ):
                continue

            combined[case_id] = {
                "case_id": case_id,
                "document": row["case_description"],
                "semantic_score": 0.0,
                "keyword_score": round(score, 4),
                "hybrid_score": round(
                    0.30 * score, 4
                ),
                "metadata": metadata
            }

        # -------------------------------------------------
        # Rank by Relevance with Small Contextual Bonuses
        # -------------------------------------------------

        results = list(combined.values())

        for result in results:

            result["priority_score"] = round(
                self._calculate_priority(result),
                4
            )

        results.sort(
            key=lambda item: (
                item["priority_score"],
                item["hybrid_score"]
            ),
            reverse=True
        )

        return results[:top_k]

    # =====================================================
    # Infer Scenario Filters
    # =====================================================

    def _infer_query_filters(self, query):
        """Infer metadata filters for explicitly stated scenarios."""

        query_lower = str(query or "").lower()
        inferred = {}

        off_peak_terms = [
            "off-peak",
            "off peak",
            "non-peak",
            "non peak"
        ]

        peak_terms = [
            "peak demand",
            "peak hours",
            "peak hour",
            "peak period",
            "peak consumption"
        ]

        # Check off-peak first so it is not mistaken for peak.
        if any(
            term in query_lower
            for term in off_peak_terms
        ):
            inferred["period"] = "Off-Peak"

        elif any(
            term in query_lower
            for term in peak_terms
        ):
            inferred["period"] = "Peak"

        no_anomaly_phrases = [
            "no anomalies",
            "no anomaly",
            "no unusual consumption",
            "no abnormal consumption",
            "without anomalies",
            "not anomalous",
            "no abnormality",
            "no irregularities",
            "no unusual pattern",
            "no abnormal pattern"
        ]

        explicitly_no_anomaly = any(
            phrase in query_lower
            for phrase in no_anomaly_phrases
        )

        anomaly_terms = [
            "anomaly",
            "anomalies",
            "anomalous",
            "abnormal consumption",
            "unusual consumption",
            "sudden increase",
            "suddenly increased",
            "unexpected increase",
            "energy spike",
            "consumption spike",
            "far above its normal pattern"
        ]

        # Explicit negative statements take priority.
        if explicitly_no_anomaly:
            inferred["is_anomaly"] = "False"

        elif any(
            term in query_lower
            for term in anomaly_terms
        ):
            inferred["is_anomaly"] = "True"

        return inferred

    # =====================================================
    # Metadata Filter
    # =====================================================

    def _build_filter(self, filters):

        if not filters:
            return None

        conditions = []

        mapping = {
            "season": "season",
            "period": "period",
            "consumption_level": "consumption_level",
            "is_anomaly": "is_anomaly"
        }

        for key, metadata_key in mapping.items():

            if (
                key in filters
                and filters[key] is not None
            ):
                conditions.append({
                    metadata_key: str(filters[key])
                })

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {
            "$and": conditions
        }

    # =====================================================
    # Manual Metadata Filter
    # =====================================================

    def _matches_filter(self, metadata, filters):

        if not filters:
            return True

        for key, value in filters.items():

            if value is None:
                continue

            if key not in metadata:
                continue

            if (
                str(metadata[key]).lower()
                != str(value).lower()
            ):
                return False

        return True

    # =====================================================
    # Case Prioritization
    # =====================================================

    def _calculate_priority(self, result):
        """
        Keep hybrid relevance dominant.
        Contextual bonuses must not overwhelm retrieval relevance.
        """

        hybrid_score = float(
            result.get("hybrid_score", 0.0)
        )

        metadata = result.get("metadata", {})

        score = 0.90 * hybrid_score

        consumption_level = str(
            metadata.get("consumption_level", "")
        ).lower()

        if consumption_level == "critical":
            score += 0.04

        elif consumption_level == "high":
            score += 0.03

        elif consumption_level == "moderate":
            score += 0.015

        is_anomaly = str(
            metadata.get("is_anomaly", "False")
        ).lower()

        if is_anomaly == "true":
            score += 0.035

        period = str(
            metadata.get("period", "")
        ).lower()

        if period == "peak":
            score += 0.025

        return min(score, 1.0)
