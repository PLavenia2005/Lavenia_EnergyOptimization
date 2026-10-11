from unittest.mock import Mock

from src.agents.a2a import (
    EnergyAgentCoordinator
)


def test_multi_agent_collaboration():

    coordinator = (
        EnergyAgentCoordinator()
    )

    coordinator.retriever.retrieve = Mock(
        return_value={
            "query": (
                "high energy consumption "
                "during peak hours"
            ),
            "filters": None,
            "count": 1,
            "results": [
                {
                    "case_id": "CASE_001",

                    "document": (
                        "Energy consumption: 150.0. "
                        "Hour: 20. "
                        "Day of week number: 2. "
                        "Month: 7. "
                        "Is peak hour: True. "
                        "Rolling mean 24: 100.0. "
                        "Consumption deviation: 50.0. "
                        "Anomaly score: 1.8."
                    ),

                    "metadata": {
                        "consumption_level": "High",
                        "period": "peak",
                        "season": "summer",
                        "is_anomaly": "False"
                    },

                    "semantic_score": 0.90,
                    "keyword_score": 0.80,
                    "hybrid_score": 0.87,
                    "priority_score": 0.95
                }
            ]
        }
    )

    coordinator.optimizer.recommend = Mock(
        return_value={
            "category": "PEAK_DEMAND",
            "consumption_level": "high",
            "period": "peak",
            "is_anomaly": False,
            "recommendation": (
                "Shift flexible loads "
                "away from peak hours."
            ),
            "priority": "High",
            "method": (
                "OpenRouter LLM + RAG"
            ),
            "fallback_used": False
        }
    )

    result = coordinator.collaborate(
        query=(
            "high energy consumption "
            "during peak hours"
        ),
        top_k=5
    )

    assert result["success"] is True

    assert "classification" in result
    assert "retrieval" in result
    assert "risk_prediction" in result
    assert "recommendation" in result
    assert "collaboration" in result
    assert "agent_contributions" in result

    contributions = (
        result["agent_contributions"]
    )

    assert (
        contributions[
            "guardrail_agent"
        ]["status"]
        == "COMPLETED"
    )

    assert (
        contributions[
            "classification_agent"
        ]["status"]
        == "COMPLETED"
    )

    assert (
        contributions[
            "retrieval_agent"
        ]["status"]
        == "COMPLETED"
    )

    assert (
        contributions[
            "risk_prediction_agent"
        ]["status"]
        == "COMPLETED"
    )

    assert (
        contributions[
            "optimization_agent"
        ]["status"]
        == "COMPLETED"
    )


def test_multi_agent_collaboration_rejects_invalid_query():

    coordinator = (
        EnergyAgentCoordinator()
    )

    result = coordinator.collaborate(
        query="What is the weather today?"
    )

    assert result["success"] is False

    assert "error" in result

    assert (
        "agent_contributions"
        in result
    )

    # Guardrail rejection is returned
    # under the guardrail key.
    assert (
        result[
            "agent_contributions"
        ][
            "guardrail"
        ]["status"]
        == "REJECTED"
    )


def test_multi_agent_collaboration_preserves_filters():

    coordinator = (
        EnergyAgentCoordinator()
    )

    coordinator.retriever.retrieve = Mock(
        return_value={
            "query": (
                "high energy consumption"
            ),
            "filters": {
                "season": "summer",
                "period": "peak"
            },
            "count": 0,
            "results": []
        }
    )

    coordinator.optimizer.recommend = Mock(
        return_value={
            "category": "HIGH_CONSUMPTION",
            "recommendation": (
                "Review high-consumption "
                "periods."
            ),
            "priority": "High",
            "method": (
                "OpenRouter LLM + RAG"
            ),
            "fallback_used": False
        }
    )

    result = coordinator.collaborate(
        query=(
            "high energy consumption"
        ),
        top_k=3,
        filters={
            "season": "summer",
            "period": "peak"
        }
    )

    assert result["success"] is True

    assert (
        result[
            "retrieval"
        ]["filters"]
        == {
            "season": "summer",
            "period": "peak"
        }
    )

    assert (
        result[
            "agent_contributions"
        ][
            "retrieval_agent"
        ]["filters"]
        == {
            "season": "summer",
            "period": "peak"
        }
    )