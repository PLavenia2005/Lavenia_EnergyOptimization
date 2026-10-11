from src.agents.llm import LLMService


def test_openrouter_connection():

    llm = LLMService()

    result = llm.generate(
        system_prompt=(
            "You are an energy optimization "
            "assistant. Answer briefly."
        ),
        user_prompt=(
            "Give one simple way to reduce "
            "peak electricity consumption."
        )
    )

    print("\nLLM RESULT:")
    print(result)

    assert isinstance(
        result,
        dict
    )

    assert "success" in result

    if result["success"]:

        assert result["content"]

        assert isinstance(
            result["content"],
            str
        )

    else:

        print(
            "\nOpenRouter was not available:"
        )

        print(
            result["error"]
        )

        