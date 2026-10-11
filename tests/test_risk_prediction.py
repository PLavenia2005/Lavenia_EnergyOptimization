from src.agents.risk_prediction_agent import (
    DemandRiskPredictionAgent
)


def test_risk_prediction_agent():

    agent = DemandRiskPredictionAgent(
        "data/processed/energy_cases.csv"
    )

    assert agent.is_trained is True
    assert agent.accuracy is not None

    case = {
        "appliances": 150,
        "hour": 20,
        "day_of_week_number": 2,
        "month": 7,
        "is_peak_hour": True,
        "rolling_mean_24": 100,
        "consumption_deviation": 50,
        "anomaly_score": 1.8
    }

    result = agent.predict(case)

    assert "risk_level" in result
    assert "risk_probability" in result
    assert "model_accuracy" in result
    assert "model" in result

    assert result["risk_level"] in [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]

    assert 0.0 <= result["risk_probability"] <= 1.0

    print("\nRisk Prediction:")
    print(result)