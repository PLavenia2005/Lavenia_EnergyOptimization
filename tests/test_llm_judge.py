import os

from src.services.llm_judge import (
    LLMJudgeService
)


def test_llm_judge_initialization():

    service = (
        LLMJudgeService()
    )

    assert service is not None

    assert service.model is not None


def test_prompt_generation():

    service = (
        LLMJudgeService()
    )

    prompt = service._build_prompt(

        query=(
            "How can we reduce high "
            "energy consumption during "
            "peak hours?"
        ),

        recommendation=(
            "Shift flexible loads "
            "to off-peak periods."
        ),

        context=[
            (
                "Historical cases show "
                "repeated peak consumption."
            )
        ]
    )

    assert "USER QUERY" in prompt

    assert (
        "RECOMMENDATION"
        in prompt
    )

    assert (
        "RETRIEVED CONTEXT"
        in prompt
    )


def test_valid_judge_result():

    service = (
        LLMJudgeService()
    )

    result = {

        "relevance": {
            "score": 0.9,
            "reason": "Directly addresses peak demand."
        },

        "actionability": {
            "score": 0.9,
            "reason": "The action can be implemented."
        },

        "evidence_support": {
            "score": 0.8,
            "reason": "Supported by retrieved cases."
        },

        "energy_saving_suitability": {
            "score": 0.95,
            "reason": "Targets peak energy consumption."
        },

        "overall_quality": {
            "score": 0.89,
            "reason": "Strong recommendation."
        },

        "final_verdict": "EXCELLENT"
    }

    assert (
        service._validate_result(
            result
        )
        is True
    )


def test_invalid_judge_score():

    service = (
        LLMJudgeService()
    )

    result = {

        "relevance": {
            "score": 1.5,
            "reason": "Invalid score."
        },

        "actionability": {
            "score": 0.9,
            "reason": "Good."
        },

        "evidence_support": {
            "score": 0.8,
            "reason": "Good."
        },

        "energy_saving_suitability": {
            "score": 0.9,
            "reason": "Good."
        },

        "overall_quality": {
            "score": 0.9,
            "reason": "Good."
        },

        "final_verdict": "GOOD"
    }

    assert (
        service._validate_result(
            result
        )
        is False
    )


def test_empty_query():

    service = (
        LLMJudgeService()
    )

    result = service.judge(
        query="",
        recommendation=(
            "Shift flexible loads "
            "to off-peak periods."
        ),
        context=[]
    )

    assert result["success"] is False


def test_empty_recommendation():

    service = (
        LLMJudgeService()
    )

    result = service.judge(
        query=(
            "How can we reduce "
            "peak energy consumption?"
        ),
        recommendation="",
        context=[]
    )

    assert result["success"] is False


def test_missing_api_key():

    service = (
        LLMJudgeService()
    )

    if not os.getenv(
        "OPENROUTER_API_KEY"
    ):

        result = service.judge(

            query=(
                "How can we reduce "
                "peak energy consumption?"
            ),

            recommendation=(
                "Shift flexible loads "
                "to off-peak periods."
            ),

            context=[]
        )

        assert result["success"] is False

        assert (
            "OPENROUTER_API_KEY"
            in result["error"]
        )