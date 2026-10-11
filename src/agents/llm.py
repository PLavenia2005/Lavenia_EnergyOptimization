from openai import OpenAI

from src.utils.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL
)


class LLMService:

    def __init__(self):

        self.enabled = bool(
            OPENROUTER_API_KEY
        )

        self.client = None

        if self.enabled:

            self.client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL
            )

    def generate(
        self,
        system_prompt,
        user_prompt
    ):

        if not self.enabled:

            return {
                "success": False,
                "content": None,
                "error": (
                    "OpenRouter API key "
                    "is not configured."
                )
            }

        if not OPENROUTER_MODEL:

            return {
                "success": False,
                "content": None,
                "error": (
                    "OPENROUTER_MODEL is not configured."
                )
            }

        try:

            response = (
                self.client
                .chat
                .completions
                .create(
                    model=OPENROUTER_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
            )

            if not response:

                return {
                    "success": False,
                    "content": None,
                    "error": (
                        "OpenRouter returned "
                        "an empty response."
                    )
                }

            if not response.choices:

                return {
                    "success": False,
                    "content": None,
                    "error": (
                        "OpenRouter returned "
                        "no choices."
                    )
                }

            message = (
                response
                .choices[0]
                .message
            )

            content = (
                message.content
                if message
                else None
            )

            if not content:

                return {
                    "success": False,
                    "content": None,
                    "error": (
                        "OpenRouter returned "
                        "empty message content."
                    )
                }

            content = str(
                content
            ).strip()

            if not content:

                return {
                    "success": False,
                    "content": None,
                    "error": (
                        "OpenRouter returned "
                        "blank content."
                    )
                }

            return {
                "success": True,
                "content": content,
                "error": None
            }

        except Exception as error:

            return {
                "success": False,
                "content": None,
                "error": (
                    f"OpenRouter request failed: "
                    f"{type(error).__name__}: {error}"
                )
            }