from src.agents.llm import LLMService
from src.services.llm_judge import LLMJudgeService


def test_live_openrouter_generation():

    service = LLMService()

    result = service.generate(
        system_prompt=(
            "You are an energy optimization "
            "assistant. Answer concisely."
        ),
        user_prompt=(
            "Give one practical way to reduce "
            "energy consumption during peak demand."
        )
    )

    print("\n\nOPENROUTER RESULT:")
    print(result)

    assert result["success"] is True
    assert result["content"]


def test_live_llm_judge():

    judge = LLMJudgeService()

    result = judge.judge(
        query=(
            "How can we reduce high energy "
            "consumption during peak hours?"
        ),
        recommendation=(
            "Shift flexible energy-intensive "
            "activities away from peak hours "
            "and stagger high-load operations."
        ),
        context=[
            (
                "Historical case shows high "
                "consumption during peak periods."
            )
        ]
    )

    print("\n\nLLM JUDGE RESULT:")
    print(result)

    assert result["success"] is True
    assert "overall_score" in result
    assert "evaluation" in result