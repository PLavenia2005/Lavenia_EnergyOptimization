from unittest.mock import patch

from src.agents.classification_agent import (
    ClassificationAgent
)

from src.agents.optimization_agent import (
    OptimizationAgent
)

from src.agents.a2a import (
    EnergyAgentCoordinator
)


def test_high_consumption_classification():

    agent = ClassificationAgent()

    result = agent.classify(
        "How can we reduce high energy consumption?"
    )

    assert (
        result["primary_category"]
        == "HIGH_CONSUMPTION"
    )


def test_peak_demand_classification():

    agent = ClassificationAgent()

    result = agent.classify(
        "How can we reduce peak demand?"
    )

    assert (
        "PEAK_DEMAND"
        in result["categories"]
    )


def test_anomaly_classification():

    agent = ClassificationAgent()

    result = agent.classify(
        "Why is there unusual energy consumption?"
    )

    assert (
        "ANOMALY"
        in result["categories"]
    )


def test_hvac_classification():

    agent = ClassificationAgent()

    result = agent.classify(
        "How can we reduce HVAC energy consumption?"
    )

    assert (
        "HVAC"
        in result["categories"]
    )


def test_general_classification():

    agent = ClassificationAgent()

    result = agent.classify(
        "Energy optimization"
    )

    assert (
        result["primary_category"]
        == "GENERAL_ENERGY"
    )


def test_optimization_recommendation():

    agent = OptimizationAgent()

    classification = {
        "primary_category": "PEAK_DEMAND",
        "categories": ["PEAK_DEMAND"],
        "query": "How can we reduce peak demand?"
    }

    fake_llm_response = {
        "success": True,
        "content": (
            "RECOMMENDED ACTION:\n"
            "Shift flexible energy loads away "
            "from peak demand periods.\n\n"
            "WHY:\n"
            "This can reduce peak electricity "
            "demand and improve energy efficiency.\n\n"
            "PRIORITY:\n"
            "High"
        ),
        "error": None
    }

    with patch.object(
        agent.llm,
        "generate",
        return_value=fake_llm_response
    ):

        result = agent.recommend(
            classification
        )

    assert (
        result["category"]
        == "PEAK_DEMAND"
    )

    assert (
        result["priority"]
        == "High"
    )

    assert (
        result["method"]
        == "OpenRouter LLM + RAG"
    )

    assert (
        result["fallback_used"]
        is False
    )

    recommendation = (
        result["recommendation"]
        .lower()
    )

    assert "peak" in recommendation


def test_optimization_falls_back_when_llm_fails():

    agent = OptimizationAgent()

    classification = {
        "primary_category": "PEAK_DEMAND",
        "categories": ["PEAK_DEMAND"],
        "query": "How can we reduce peak demand?"
    }

    fake_llm_response = {
        "success": False,
        "content": None,
        "error": "OpenRouter unavailable"
    }

    with patch.object(
        agent.llm,
        "generate",
        return_value=fake_llm_response
    ):

        result = agent.recommend(
            classification
        )

    assert (
        result["method"]
        == "Rule-Based Fallback"
    )

    assert (
        result["fallback_used"]
        is True
    )

    assert (
        result["category"]
        == "PEAK_DEMAND"
    )


def test_optimization_falls_back_for_invalid_llm_response():

    agent = OptimizationAgent()

    classification = {
        "primary_category": "PEAK_DEMAND",
        "categories": ["PEAK_DEMAND"],
        "query": "How can we reduce peak demand?"
    }

    fake_llm_response = {
        "success": True,
        "content": "user safety: safe",
        "error": None
    }

    with patch.object(
        agent.llm,
        "generate",
        return_value=fake_llm_response
    ):

        result = agent.recommend(
            classification
        )

    assert (
        result["method"]
        == "Rule-Based Fallback"
    )

    assert (
        result["fallback_used"]
        is True
    )


def test_agent_coordinator():

    coordinator = (
        EnergyAgentCoordinator()
    )

    result = coordinator.process(
        query=(
            "How can we reduce "
            "high energy consumption?"
        ),
        top_k=3
    )

    assert result["success"] is True
    assert "classification" in result
    assert "retrieval" in result
    assert "recommendation" in result


def test_coordinator_rejects_invalid_query():

    coordinator = (
        EnergyAgentCoordinator()
    )

    result = coordinator.process(
        query="What is my food today?",
        top_k=3
    )

    assert result["success"] is False
    assert "error" in result