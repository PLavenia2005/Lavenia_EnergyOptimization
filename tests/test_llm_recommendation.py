from unittest.mock import patch

from src.agents.a2a import (
    EnergyAgentCoordinator
)


def test_llm_recommendation_pipeline():

    coordinator = (
        EnergyAgentCoordinator()
    )

    fake_llm_response = {
        "success": True,
        "content": (
            "RECOMMENDED ACTION:\n"
            "Shift flexible energy-intensive "
            "activities away from peak demand "
            "periods.\n\n"
            "WHY:\n"
            "The query indicates high energy "
            "consumption during peak hours, so "
            "reducing peak-period loads can "
            "improve energy efficiency.\n\n"
            "PRIORITY:\n"
            "High"
        ),
        "error": None
    }

    with patch.object(
        coordinator.optimizer.llm,
        "generate",
        return_value=fake_llm_response
    ):

        result = coordinator.process(
            query=(
                "How can we reduce high "
                "energy consumption during "
                "peak hours?"
            ),
            top_k=3,
            filters={
                "period": "Peak"
            }
        )

    assert result["success"] is True

    recommendation = result[
        "recommendation"
    ]

    assert "recommendation" in recommendation
    assert "category" in recommendation
    assert "priority" in recommendation
    assert "method" in recommendation
    assert "fallback_used" in recommendation

    assert (
        recommendation["fallback_used"]
        is False
    )

    assert (
        recommendation["method"]
        == "OpenRouter LLM + RAG"
    )

    assert (
        "peak"
        in recommendation[
            "recommendation"
        ].lower()
    )

    print(
        "\nRecommendation:"
    )

    print(
        recommendation
    )


def test_llm_pipeline_uses_fallback_when_unavailable():

    coordinator = (
        EnergyAgentCoordinator()
    )

    fake_llm_response = {
        "success": False,
        "content": None,
        "error": "OpenRouter unavailable"
    }

    with patch.object(
        coordinator.optimizer.llm,
        "generate",
        return_value=fake_llm_response
    ):

        result = coordinator.process(
            query=(
                "How can we reduce high "
                "energy consumption?"
            ),
            top_k=3
        )

    assert result["success"] is True

    recommendation = result[
        "recommendation"
    ]

    assert (
        recommendation["fallback_used"]
        is True
    )

    assert (
        recommendation["method"]
        == "Rule-Based Fallback"
    )