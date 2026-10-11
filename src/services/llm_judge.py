import json
import os
import re

import requests
from dotenv import load_dotenv


load_dotenv()


class LLMJudgeService:
    """
    LLM-as-a-Judge service for evaluating energy
    optimization recommendations.

    The judge evaluates:
        - relevance
        - actionability
        - evidence support
        - energy-saving suitability
        - overall quality
    """

    def __init__(self):

        self.api_key = os.getenv(
            "OPENROUTER_API_KEY"
        )

        # Main LLM uses OPENROUTER_MODEL.
        # LLM Judge uses a separate model.
        self.model_name = os.getenv(
            "LLM_JUDGE_MODEL",
            "openai/gpt-4.1-mini"
        )

        self.base_url = os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1"
        )

    def _build_prompt(
        self,
        query,
        recommendation,
        context
    ):

        context_text = "\n".join(
            [
                f"- {item}"
                for item in context
            ]
        )

        return f"""
You are an expert evaluator for an
AI-powered energy optimization assistant.

Evaluate the recommendation against the
user's energy query and retrieved evidence.

USER QUERY:
{query}

RECOMMENDATION:
{recommendation}

RETRIEVED CONTEXT:
{context_text}

Evaluate the recommendation using these criteria:

1. Relevance
Does the recommendation directly address
the user's energy problem?

2. Actionability
Can the user reasonably implement
the recommended action?

3. Evidence Support
Is the recommendation supported by
the retrieved context?

4. Energy Saving Suitability
Is the recommendation reasonably related
to reducing energy consumption, peak demand,
or improving energy efficiency?

5. Overall Quality
How good is the recommendation overall?

Return ONLY valid JSON.

Do not use Markdown code fences.

Use exactly this structure:

{{
    "relevance": {{
        "score": 0.0,
        "reason": ""
    }},
    "actionability": {{
        "score": 0.0,
        "reason": ""
    }},
    "evidence_support": {{
        "score": 0.0,
        "reason": ""
    }},
    "energy_saving_suitability": {{
        "score": 0.0,
        "reason": ""
    }},
    "overall_quality": {{
        "score": 0.0,
        "reason": ""
    }},
    "final_verdict": ""
}}

Scores must be between 0.0 and 1.0.

The final verdict must be exactly one of:

EXCELLENT
GOOD
NEEDS_IMPROVEMENT
POOR
"""

    def _extract_json(
        self,
        content
    ):

        if not content:
            return None

        content = str(
            content
        ).strip()

        # Remove Markdown JSON fences if present.
        content = re.sub(
            r"^```json\s*",
            "",
            content,
            flags=re.IGNORECASE
        )

        content = re.sub(
            r"^```\s*",
            "",
            content
        )

        content = re.sub(
            r"\s*```$",
            "",
            content
        )

        try:

            return json.loads(
                content
            )

        except json.JSONDecodeError:
            pass

        match = re.search(
            r"\{.*\}",
            content,
            re.DOTALL
        )

        if not match:
            return None

        try:

            return json.loads(
                match.group(0)
            )

        except json.JSONDecodeError:

            return None

    def _validate_result(
        self,
        result
    ):

        if not isinstance(
            result,
            dict
        ):
            return False

        required_sections = [
            "relevance",
            "actionability",
            "evidence_support",
            "energy_saving_suitability",
            "overall_quality",
            "final_verdict"
        ]

        for section in required_sections:

            if section not in result:
                return False

        score_sections = [
            "relevance",
            "actionability",
            "evidence_support",
            "energy_saving_suitability",
            "overall_quality"
        ]

        for section in score_sections:

            value = result.get(
                section,
                {}
            )

            if not isinstance(
                value,
                dict
            ):
                return False

            if "score" not in value:
                return False

            try:

                score = float(
                    value["score"]
                )

            except (
                TypeError,
                ValueError
            ):

                return False

            if not 0.0 <= score <= 1.0:
                return False

        valid_verdicts = {
            "EXCELLENT",
            "GOOD",
            "NEEDS_IMPROVEMENT",
            "POOR"
        }

        verdict = str(
            result["final_verdict"]
        ).strip().upper()

        if verdict not in valid_verdicts:
            return False

        result["final_verdict"] = verdict

        return True

    def judge(
        self,
        query,
        recommendation,
        context=None
    ):

        if not query or not query.strip():

            return {
                "success": False,
                "error": "Query cannot be empty."
            }

        if (
            not recommendation
            or not recommendation.strip()
        ):

            return {
                "success": False,
                "error": (
                    "Recommendation cannot be empty."
                )
            }

        if context is None:
            context = []

        if not isinstance(
            context,
            list
        ):

            context = [
                str(context)
            ]

        if not self.api_key:

            return {
                "success": False,
                "error": (
                    "OPENROUTER_API_KEY "
                    "is not configured."
                )
            }

        prompt = self._build_prompt(
            query=query,
            recommendation=recommendation,
            context=context
        )

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": (
                "application/json"
            )
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict "
                        "energy recommendation "
                        "evaluator."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0
        }

        try:

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            choices = data.get(
                "choices",
                []
            )

            if not choices:

                return {
                    "success": False,
                    "error": (
                        "LLM judge returned "
                        "no choices."
                    ),
                    "raw_response": data
                }

            content = (
                choices[0]
                .get("message", {})
                .get("content", "")
            )

            result = self._extract_json(
                content
            )

            if result is None:

                return {
                    "success": False,
                    "error": (
                        "LLM judge returned "
                        "invalid JSON."
                    ),
                    "raw_response": content
                }

            if not self._validate_result(
                result
            ):

                return {
                    "success": False,
                    "error": (
                        "LLM judge returned "
                        "an invalid evaluation "
                        "structure."
                    ),
                    "raw_response": result
                }

            overall_score = float(
                result[
                    "overall_quality"
                ]["score"]
            )

            return {
                "success": True,
                "query": query,
                "overall_score": round(
                    overall_score,
                    4
                ),
                "evaluation": result
            }

        except requests.RequestException as exc:

            return {
                "success": False,
                "error": (
                    f"LLM judge request failed: "
                    f"{exc}"
                )
            }

        except Exception as exc:

            return {
                "success": False,
                "error": (
                    f"LLM judge failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            }