from src.agents.effectiveness_agent import (
    EnergySavingEffectivenessAgent
)


def test_effectiveness_prediction():

    agent = EnergySavingEffectivenessAgent()

    retrieved_cases = [

        {
            "case_id": "CASE_001",

            "semantic_score": 0.90,

            "keyword_score": 0.80,

            "hybrid_score": 0.87,

            "priority_score": 0.90,

            "metadata": {
                "consumption_level": "High",
                "period": "Peak",
                "is_anomaly": True
            }
        },

        {
            "case_id": "CASE_002",

            "semantic_score": 0.82,

            "keyword_score": 0.75,

            "hybrid_score": 0.80,

            "priority_score": 0.85,

            "metadata": {
                "consumption_level": "High",
                "period": "Peak",
                "is_anomaly": False
            }
        }
    ]

    recommendation = """
    RECOMMENDED ACTION:
    Shift flexible energy loads away from peak hours.

    WHY:
    Historical peak-demand cases indicate that load
    management can reduce unnecessary peak consumption.

    PRIORITY:
    HIGH
    """

    risk_prediction = {
        "risk_level": "HIGH",
        "risk_probability": 0.82
    }

    result = agent.predict(
        retrieved_cases=retrieved_cases,
        recommendation=recommendation,
        risk_prediction=risk_prediction
    )

    print("\nEffectiveness Prediction:")
    print(result)

    assert result["success"] is True

    assert 0 <= result[
        "effectiveness_score"
    ] <= 100

    assert result[
        "effectiveness_level"
    ] in {
        "LOW",
        "MEDIUM",
        "HIGH"
    }

    assert result[
        "recommendation_category"
    ] == "peak_demand"

    assert result[
        "historical_savings_available"
    ] is False


def test_effectiveness_changes_with_context():

    agent = EnergySavingEffectivenessAgent()

    strong_cases = [
        {
            "hybrid_score": 0.90,
            "priority_score": 0.95,
            "metadata": {
                "consumption_level": "Critical",
                "period": "Peak",
                "is_anomaly": True
            }
        }
    ]

    weak_cases = [
        {
            "hybrid_score": 0.30,
            "priority_score": 0.25,
            "metadata": {
                "consumption_level": "Normal",
                "period": "Off-Peak",
                "is_anomaly": False
            }
        }
    ]

    recommendation = (
        "Shift flexible loads away from peak hours."
    )

    strong_result = agent.predict(
        strong_cases,
        recommendation
    )

    weak_result = agent.predict(
        weak_cases,
        recommendation
    )

    assert strong_result[
        "effectiveness_score"
    ] > weak_result[
        "effectiveness_score"
    ]


def test_effectiveness_requires_recommendation():

    agent = EnergySavingEffectivenessAgent()

    result = agent.predict(
        retrieved_cases=[],
        recommendation=""
    )

    assert result["success"] is False

    assert (
        "Recommendation is required"
        in result["error"]
    )