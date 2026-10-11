import os

import pytest

from src.services.deepeval_evaluation import (
    DeepEvalRecommendationService
)


def test_deepeval_service_initialization():

    service = (
        DeepEvalRecommendationService()
    )

    assert service is not None


def test_deepeval_requires_openrouter_key():

    service = (
        DeepEvalRecommendationService()
    )

    if not os.getenv(
        "OPENROUTER_API_KEY"
    ):
        result = (
            service.evaluate_recommendation(
                query=(
                    "How can we reduce "
                    "high energy consumption "
                    "during peak hours?"
                ),
                recommendation=(
                    "Shift flexible loads "
                    "to off-peak hours "
                    "to reduce peak demand."
                ),
                context=[
                    "Historical cases show "
                    "high consumption during "
                    "peak hours."
                ]
            )
        )

        assert result["success"] is False

        assert (
            "OPENROUTER_API_KEY"
            in result["error"]
        )


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY is not configured"
)
def test_deepeval_recommendation_evaluation():

    service = (
        DeepEvalRecommendationService()
    )

    result = (
        service.evaluate_recommendation(

            query=(
                "How can we reduce "
                "high energy consumption "
                "during peak hours?"
            ),

            recommendation=(
                "Shift flexible and "
                "non-critical loads to "
                "off-peak periods to "
                "reduce peak demand."
            ),

            context=[
                (
                    "Historical energy cases "
                    "show repeated high "
                    "consumption during peak "
                    "periods."
                ),
                (
                    "Energy best practice: "
                    "shift flexible loads "
                    "to off-peak periods."
                )
            ]
        )
    )

    assert result["success"] is True

    assert (
        0.0
        <= result["overall_score"]
        <= 1.0
    )

    assert (
        "metrics"
        in result
    )

    assert len(
        result["metrics"]
    ) >= 1