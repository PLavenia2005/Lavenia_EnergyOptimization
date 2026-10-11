import pandas as pd

from src.rag.hybrid_search import HybridSearch
from src.utils.config import PROCESSED_DATA_PATH


class RetrievalEvaluationService:

    def __init__(self):

        self.search = HybridSearch()
        self.search_engine = self.search

        self.df = pd.read_csv(
            PROCESSED_DATA_PATH
        )

        self.df["case_id"] = (
            self.df["case_id"]
            .astype(str)
            .str.strip()
        )

    @staticmethod
    def _normalize_case_id(case_id):

        if case_id is None:
            return ""

        return str(
            case_id
        ).strip()

    def _detect_conditions(
        self,
        query
    ):

        query_lower = query.lower().replace("-", " ")
        query_lower = " ".join(query_lower.split())

        conditions = {
            "period": None,
            "consumption_level": [],
            "is_anomaly": None,
            "scenario": None,
        }

        # -----------------------------
        # Period
        # -----------------------------

        if (
            "off peak" in query_lower
            or "non peak" in query_lower
        ):

            conditions["period"] = "Off-Peak"

        elif (
            "peak" in query_lower
            or "peak hours" in query_lower
            or "peak hour" in query_lower
        ):

            conditions["period"] = "Peak"

        # -----------------------------
        # Consumption level
        # -----------------------------

        if "critical" in query_lower:

            conditions[
                "consumption_level"
            ].append("Critical")

        if any(
            term in query_lower
            for term in (
                "high consumption",
                "high energy",
                "energy consumption high",
                "consumption is high",
                "consumption remains high",
                "consumption stays high",
                "unusually high",
                "excessive consumption",
                "high demand",
                "too much energy",
            )
        ):

            conditions[
                "consumption_level"
            ].append("High")

        if any(
            term in query_lower
            for term in (
                "moderate consumption",
                "moderate energy",
                "moderate level",
            )
        ):

            conditions[
                "consumption_level"
            ].append("Moderate")

        if "normal consumption" in query_lower:

            conditions[
                "consumption_level"
            ].append("Normal")

        if any(
            term in query_lower
            for term in (
                "low consumption",
                "low energy",
                "maintain low",
                "keep consumption low",
                "keep energy use low",
                "energy efficiency",
                "energy efficient",
            )
        ):

            conditions[
                "consumption_level"
            ].append("Low")

        # -----------------------------
        # Anomaly
        # -----------------------------

        if any(
            term in query_lower
            for term in (
                "anomaly",
                "anomalous",
                "unusual consumption",
                "unusual energy",
                "abnormal consumption",
                "abnormal energy",
                "energy spike",
                "consumption spike",
                "sudden increase",
                "unexpected increase",
            )
        ):

            conditions[
                "is_anomaly"
            ] = True

        # Recognize general optimization scenarios.
        if any(term in query_lower for term in [
                "multiple facilities",
                "multiple buildings",
                "similar conditions",
                "previous optimization",
                "previous measures",
                "alternative actions",
                "failed optimization",
                "still high",
                "remains high",
                "reduce energy",
                "reduce consumption",
                "energy optimization",
                "maintain low",
                "how can we reduce",
                "why is energy consumption high",
                "why is consumption high",
                "improve efficiency",
            ]):
            conditions["scenario"] = "general_optimization"

        return conditions

    def _build_search_filters(
        self,
        conditions
    ):

        filters = {}

        if conditions.get("period"):

            filters["period"] = (
                conditions["period"]
            )

        consumption_levels = (
            conditions.get(
                "consumption_level",
                []
            )
        )

        # HybridSearch currently accepts
        # one consumption level.
        # If one level is detected, use it.
        if len(consumption_levels) == 1:

            filters[
                "consumption_level"
            ] = consumption_levels[0]

        if conditions.get(
            "is_anomaly"
        ) is not None:

            filters[
                "is_anomaly"
            ] = conditions[
                "is_anomaly"
            ]

        return (
            filters
            if filters
            else None
        )

    def _find_relevant_cases(
        self,
        conditions
    ):

        filtered = self.df.copy()

        # -----------------------------
        # Period filter
        # -----------------------------

        period = conditions.get(
            "period"
        )

        if period:

            filtered = filtered[
                filtered["period"]
                .astype(str)
                .str.lower()
                ==
                str(period).lower()
            ]

        # -----------------------------
        # Consumption filter
        # -----------------------------

        consumption_level = (
            conditions.get(
                "consumption_level"
            )
        )

        if consumption_level:

            if isinstance(
                consumption_level,
                str
            ):

                consumption_level = [
                    consumption_level
                ]

            allowed_levels = {
                str(level).lower()
                for level
                in consumption_level
            }

            filtered = filtered[
                filtered[
                    "consumption_level"
                ]
                .astype(str)
                .str.lower()
                .isin(
                    allowed_levels
                )
            ]

        # -----------------------------
        # Anomaly filter
        # -----------------------------

        anomaly = conditions.get(
            "is_anomaly"
        )

        if anomaly is not None:

            filtered = filtered[
                filtered["is_anomaly"]
                .astype(str)
                .str.lower()
                ==
                str(anomaly).lower()
            ]

        return [
            self._normalize_case_id(
                case_id
            )
            for case_id
            in filtered[
                "case_id"
            ].tolist()
        ]

    def _get_relevant_cases(
        self,
        conditions
    ):

        return self._find_relevant_cases(
            conditions
        )

    def _calculate_metrics(
        self,
        retrieved_ids,
        relevant_ids
    ):

        retrieved_ids = {
            self._normalize_case_id(
                case_id
            )
            for case_id
            in retrieved_ids
        }

        relevant_ids = {
            self._normalize_case_id(
                case_id
            )
            for case_id
            in relevant_ids
        }

        retrieved_ids.discard("")
        relevant_ids.discard("")

        # -----------------------------
        # Confusion counts
        # -----------------------------

        true_positive = len(
            retrieved_ids
            .intersection(
                relevant_ids
            )
        )

        false_positive = len(
            retrieved_ids
            - relevant_ids
        )

        false_negative = len(
            relevant_ids
            - retrieved_ids
        )

        # -----------------------------
        # Precision
        # -----------------------------

        if retrieved_ids:

            precision = (
                true_positive
                / len(retrieved_ids)
            )

        else:

            precision = 0.0

        # -----------------------------
        # Recall
        # -----------------------------

        if relevant_ids:

            recall = (
                true_positive
                / len(relevant_ids)
            )

        else:

            recall = 0.0

        # -----------------------------
        # F1 Score
        # -----------------------------

        if (
            precision
            + recall
            > 0
        ):

            f1_score = (
                2
                * precision
                * recall
                /
                (
                    precision
                    + recall
                )
            )

        else:

            f1_score = 0.0

        # -----------------------------
        # Retrieval accuracy
        #
        # For this project, accuracy
        # represents the proportion of
        # retrieved cases that are
        # actually relevant.
        #
        # This is intentionally the
        # same interpretation used
        # previously in the project.
        # -----------------------------

        if retrieved_ids:

            accuracy = (
                true_positive
                / len(retrieved_ids)
            )

        else:

            accuracy = 0.0

        return {
            "accuracy": round(
                accuracy,
                4
            ),

            "precision": round(
                precision,
                4
            ),

            "recall": round(
                recall,
                4
            ),

            "f1_score": round(
                f1_score,
                4
            ),

            "true_positive":
                true_positive,

            "false_positive":
                false_positive,

            "false_negative":
                false_negative,

            "retrieved_count":
                len(retrieved_ids),

            "relevant_count":
                len(relevant_ids)
        }

    def evaluate_query(
        self,
        query,
        top_k=5
    ):

        # -----------------------------
        # Step 1:
        # Detect query conditions
        # -----------------------------

        conditions = (
            self._detect_conditions(
                query
            )
        )

        has_condition = (
            conditions[
                "period"
            ] is not None
            or
            bool(
                conditions[
                    "consumption_level"
                ]
            )
            or
            conditions[
                "is_anomaly"
            ] is not None
        )

        if not has_condition:
            retrieved_results = self.search.search(
                query=query,
                top_k=top_k,
                filters=None,
            )

            retrieved_ids = [
                self._normalize_case_id(result.get("case_id"))
                for result in retrieved_results
            ]

            return {
                "success": True,
                "query": query,
                "dataset_size": len(self.df),
                "top_k": top_k,
                "relevant_cases": None,
                "relevant_case_count": None,
                "detected_conditions": conditions,
                "applied_filters": None,
                "metrics": {
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1_score": None,
                },
                "diagnostics": {
                    "retrieved_count": len(retrieved_ids),
                    "relevant_count": None,
                    "evaluation_note": (
                        "General optimization query: retrieval results are available, "
                        "but relevance metrics were not calculated because no explicit "
                        "ground-truth condition was identified."
                    ),
                },
                "retrieved_case_ids": retrieved_ids,
            }

        # -----------------------------
        # Step 2:
        # Build dataset ground truth
        # -----------------------------

        relevant_ids = (
            self._find_relevant_cases(
                conditions
            )
        )

        # -----------------------------
        # Step 3:
        # Build retrieval filters
        # -----------------------------

        search_filters = (
            self._build_search_filters(
                conditions
            )
        )

        # -----------------------------
        # Step 4:
        # Perform query-aware
        # hybrid retrieval
        # -----------------------------

        retrieved_results = (
            self.search.search(
                query=query,
                top_k=top_k,
                filters=search_filters
            )
        )

        # -----------------------------
        # Step 5:
        # Extract retrieved IDs
        # -----------------------------

        retrieved_ids = [
            self._normalize_case_id(
                result.get(
                    "case_id"
                )
            )
            for result
            in retrieved_results
        ]

        # -----------------------------
        # Step 6:
        # Calculate metrics
        # -----------------------------

        metrics = (
            self._calculate_metrics(
                retrieved_ids,
                relevant_ids
            )
        )

        # -----------------------------
        # Step 7:
        # Return evaluation
        # -----------------------------

        return {

            "success": True,

            "query": query,

            "dataset_size":
                len(self.df),

            "top_k":
                top_k,

            "relevant_cases":
                len(relevant_ids),

            "relevant_case_count":
                len(relevant_ids),

            "detected_conditions":
                conditions,

            "applied_filters":
                search_filters,

            "metrics": {

                "accuracy":
                    metrics[
                        "accuracy"
                    ],

                "precision":
                    metrics[
                        "precision"
                    ],

                "recall":
                    metrics[
                        "recall"
                    ],

                "f1_score":
                    metrics[
                        "f1_score"
                    ]
            },

            "diagnostics": {

                "true_positive":
                    metrics[
                        "true_positive"
                    ],

                "false_positive":
                    metrics[
                        "false_positive"
                    ],

                "false_negative":
                    metrics[
                        "false_negative"
                    ],

                "retrieved_count":
                    metrics[
                        "retrieved_count"
                    ],

                "relevant_count":
                    metrics[
                        "relevant_count"
                    ]
            },

            "retrieved_case_ids":
                retrieved_ids
        }