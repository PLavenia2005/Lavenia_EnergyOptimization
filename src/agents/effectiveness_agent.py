from typing import Any, Dict, List


class EnergySavingEffectivenessAgent:

    """
    Estimates the potential effectiveness of an energy-saving recommendation.

    IMPORTANT:
    Dataset 374 does not contain historical optimization actions or measured
    savings outcomes. Therefore, this agent produces an ESTIMATED effectiveness
    score based on retrieved evidence and contextual indicators.

    It does not claim actual historical energy savings.
    """

    CATEGORY_BASE_SCORES = {
        "peak_demand": 0.85,
        "load_management": 0.85,
        "hvac": 0.78,
        "temperature": 0.78,
        "equipment": 0.72,
        "maintenance": 0.70,
        "lighting": 0.65,
        "standby": 0.60,
        "monitoring": 0.55,
        "general": 0.50,
    }

    def __init__(self):
        self.model = "Contextual Effectiveness Estimator"

    # ============================================================
    # CATEGORY DETECTION
    # ============================================================

    def _detect_category(self, text: str) -> str:

        text = str(text or "").lower()

        category_keywords = {

            "peak_demand": [
                "peak demand",
                "peak hour",
                "peak hours",
                "peak period",
                "demand management",
                "load shifting",
                "load management",
            ],

            "hvac": [
                "hvac",
                "heating",
                "cooling",
                "air conditioning",
                "air conditioner",
                "temperature control",
            ],

            "temperature": [
                "temperature",
                "thermostat",
                "thermal",
            ],

            "equipment": [
                "equipment",
                "appliance",
                "machine",
                "motor",
            ],

            "maintenance": [
                "maintenance",
                "maintain",
                "servicing",
                "service",
            ],

            "lighting": [
                "lighting",
                "lights",
                "light",
                "led",
            ],

            "standby": [
                "standby",
                "idle",
                "switch off",
                "turn off",
                "unused",
            ],

            "monitoring": [
                "monitor",
                "monitoring",
                "metering",
                "tracking",
                "measure",
            ],
        }

        for category, keywords in category_keywords.items():

            for keyword in keywords:

                if keyword in text:
                    return category

        return "general"

    # ============================================================
    # SAFE FLOAT
    # ============================================================

    def _safe_float(
        self,
        value: Any,
        default: float = 0.0
    ) -> float:

        try:

            return float(value)

        except (TypeError, ValueError):

            return default

    # ============================================================
    # AVERAGE RETRIEVAL QUALITY
    # ============================================================

    def _calculate_retrieval_quality(
        self,
        retrieved_cases: List[Dict[str, Any]]
    ) -> float:

        if not retrieved_cases:
            return 0.0

        scores = []

        for case in retrieved_cases:

            hybrid_score = self._safe_float(
                case.get("hybrid_score"),
                0.0
            )

            priority_score = self._safe_float(
                case.get("priority_score"),
                0.0
            )

            evidence_score = (
                (0.70 * hybrid_score)
                +
                (0.30 * priority_score)
            )

            scores.append(
                min(
                    max(evidence_score, 0.0),
                    1.0
                )
            )

        return sum(scores) / len(scores)

    # ============================================================
    # CONTEXT FACTOR
    # ============================================================

    def _calculate_context_factor(
        self,
        retrieved_cases: List[Dict[str, Any]]
    ) -> float:

        if not retrieved_cases:
            return 0.0

        total = 0.0

        for case in retrieved_cases:

            metadata = case.get(
                "metadata",
                {}
            )

            contribution = 0.0

            if str(
                metadata.get(
                    "consumption_level",
                    ""
                )
            ).lower() in {
                "high",
                "critical"
            }:

                contribution += 0.35

            if str(
                metadata.get(
                    "period",
                    ""
                )
            ).lower() == "peak":

                contribution += 0.25

            is_anomaly = metadata.get(
                "is_anomaly",
                False
            )

            if (
                is_anomaly is True
                or str(is_anomaly).lower() == "true"
            ):

                contribution += 0.25

            if contribution == 0:

                contribution = 0.10

            total += min(
                contribution,
                1.0
            )

        return min(
            total / len(retrieved_cases),
            1.0
        )

    # ============================================================
    # EFFECTIVENESS LEVEL
    # ============================================================

    def _get_effectiveness_level(
        self,
        score: float
    ) -> str:

        if score >= 0.75:
            return "HIGH"

        if score >= 0.50:
            return "MEDIUM"

        return "LOW"

    # ============================================================
    # MAIN PREDICTION
    # ============================================================

    def predict(
        self,
        retrieved_cases: List[Dict[str, Any]],
        recommendation: Any,
        risk_prediction: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:

        if not isinstance(
            retrieved_cases,
            list
        ):

            retrieved_cases = []

        if not recommendation:

            return {
                "success": False,
                "error": "Recommendation is required."
            }

        recommendation_text = str(
            recommendation
        )

        category = self._detect_category(
            recommendation_text
        )

        base_score = self.CATEGORY_BASE_SCORES.get(
            category,
            self.CATEGORY_BASE_SCORES["general"]
        )

        retrieval_quality = (
            self._calculate_retrieval_quality(
                retrieved_cases
            )
        )

        context_factor = (
            self._calculate_context_factor(
                retrieved_cases
            )
        )

        risk_factor = 0.0

        if isinstance(
            risk_prediction,
            dict
        ):

            risk_level = str(
                risk_prediction.get(
                    "risk_level",
                    ""
                )
            ).upper()

            if risk_level == "HIGH":
                risk_factor = 0.15

            elif risk_level == "MEDIUM":
                risk_factor = 0.08

            elif risk_level == "LOW":
                risk_factor = 0.02

        # --------------------------------------------------------
        # ESTIMATION FORMULA
        # --------------------------------------------------------
        #
        # 50% recommendation category suitability
        # 30% retrieved evidence quality
        # 15% consumption context
        #  5% demand-risk context
        #
        # This is an estimate, not measured historical savings.
        # --------------------------------------------------------

        score = (
            (0.50 * base_score)
            +
            (0.30 * retrieval_quality)
            +
            (0.15 * context_factor)
            +
            (0.05 * risk_factor)
        )

        score = min(
            max(score, 0.0),
            1.0
        )

        effectiveness_score = round(
            score * 100,
            2
        )

        effectiveness_level = (
            self._get_effectiveness_level(
                score
            )
        )

        return {
            "success": True,

            "effectiveness_score": (
                effectiveness_score
            ),

            "effectiveness_level": (
                effectiveness_level
            ),

            "recommendation_category": (
                category
            ),

            "retrieved_cases_used": (
                len(retrieved_cases)
            ),

            "retrieval_quality": round(
                retrieval_quality * 100,
                2
            ),

            "context_factor": round(
                context_factor * 100,
                2
            ),

            "risk_factor": round(
                risk_factor * 100,
                2
            ),

            "estimation_method": (
                self.model
            ),

            "historical_savings_available": (
                False
            ),

            "interpretation": (
                "Estimated potential effectiveness based "
                "on recommendation category, retrieved "
                "historical case relevance, consumption "
                "context, and demand-risk context. "
                "This is not a measured historical savings "
                "percentage."
            )
        }