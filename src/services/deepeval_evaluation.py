import os

from dotenv import load_dotenv

from deepeval.metrics import (
    AnswerRelevancyMetric,
    GEval
)

from deepeval.models import OpenRouterModel

from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams
)


load_dotenv()


class DeepEvalRecommendationService:
    """
    DeepEval-based evaluation service for
    energy optimization recommendations.
    """

    def __init__(self):

        self.api_key = os.getenv(
            "OPENROUTER_API_KEY"
        )

        self.model_name = os.getenv(
            "DEEPEVAL_MODEL",
            os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1")
        )

        self.model = None

        if self.api_key:

            self.model = OpenRouterModel(
                model=self.model_name,
                api_key=self.api_key,
                base_url=os.getenv(
                    "OPENROUTER_BASE_URL",
                    "https://openrouter.ai/api/v1"
                ),
                temperature=0,
                generation_kwargs={
                    "max_tokens": 150
                    
                }
            )

    def _build_metrics(self):

        if self.model is None:

            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured."
            )

        # ---------------------------------------------------------
        # ANSWER RELEVANCY
        # ---------------------------------------------------------

        answer_relevancy = AnswerRelevancyMetric(
            model=self.model,
            threshold=0.5,
            async_mode=False
        )

        # ---------------------------------------------------------
        # ENERGY RECOMMENDATION QUALITY
        # ---------------------------------------------------------

        recommendation_quality = GEval(
            name="Energy Recommendation Quality",

            criteria=(
                "Evaluate the quality of the energy "
                "optimization recommendation. "
                "The recommendation must address the "
                "user's energy problem, provide a practical "
                "action, remain consistent with the supplied "
                "historical context, and focus on reducing "
                "energy consumption or peak demand. "
                "Penalize unsupported claims or actions "
                "not justified by the context."
            ),

            evaluation_steps=[
                (
                    "Determine whether the recommendation "
                    "directly addresses the user's query."
                ),

                (
                    "Determine whether the recommendation "
                    "contains a practical and actionable "
                    "energy-saving action."
                ),

                (
                    "Determine whether the recommendation "
                    "is supported by the supplied context."
                ),

                (
                    "Determine whether the recommendation "
                    "supports energy efficiency or demand "
                    "reduction."
                ),

                (
                    "Penalize unsupported facts, invented "
                    "measurements, invented equipment details, "
                    "or unrelated recommendations."
                )
            ],

            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.CONTEXT
            ],

            model=self.model,

            threshold=0.5,

            async_mode=False
        )

        return [
            answer_relevancy,
            recommendation_quality
        ]

    def evaluate_recommendation(
        self,
        query: str,
        recommendation: str,
        context=None
    ):

        # ---------------------------------------------------------
        # INPUT VALIDATION
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # MODEL VALIDATION
        # ---------------------------------------------------------

        if self.model is None:

            return {
                "success": False,
                "error": (
                    "OPENROUTER_API_KEY "
                    "is not configured."
                )
            }

        # ---------------------------------------------------------
        # TEST CASE
        # ---------------------------------------------------------

        test_case = LLMTestCase(
            input=query,
            actual_output=recommendation,
            context=context
        )

        try:

            metrics = self._build_metrics()

            metric_results = {}

            scores = []

            # -----------------------------------------------------
            # RUN EACH METRIC INDEPENDENTLY
            # -----------------------------------------------------

            for metric in metrics:

                metric_name = getattr(
                    metric,
                    "name",
                    metric.__class__.__name__
                )

                try:

                    metric.measure(
                        test_case
                    )

                    score = getattr(
                        metric,
                        "score",
                        None
                    )

                    reason = getattr(
                        metric,
                        "reason",
                        None
                    )

                    success = getattr(
                        metric,
                        "success",
                        score is not None
                    )

                    if score is not None:

                        score = float(score)

                        scores.append(
                            score
                        )

                    metric_results[
                        metric_name
                    ] = {
                        "score": (
                            round(
                                score,
                                4
                            )
                            if score is not None
                            else None
                        ),

                        "success": bool(
                            success
                        ),

                        "reason": reason
                    }

                except Exception as metric_error:

                    metric_results[
                        metric_name
                    ] = {
                        "score": None,
                        "success": False,
                        "reason": (
                            f"{type(metric_error).__name__}: "
                            f"{metric_error}"
                        )
                    }

            # -----------------------------------------------------
            # NO METRIC PRODUCED A SCORE
            # -----------------------------------------------------

            if not scores:

                metric_errors = {}

                for name, result in metric_results.items():

                    metric_errors[name] = {
                        "reason": result.get(
                            "reason",
                            "No score produced."
                        ),
                        "success": result.get(
                            "success",
                            False
                        )
                    }

                return {
                    "success": False,
                    "query": query,
                    "overall_score": None,
                    "metrics": metric_results,
                    "error": (
                        "DeepEval could not produce a score. "
                        "Check the individual metric errors."
                    ),
                    "diagnostics": metric_errors
                }


            # -----------------------------------------------------
            # OVERALL SCORE
            # -----------------------------------------------------

            overall_score = (
                sum(scores) /
                len(scores)
            )

            return {
                "success": True,
                "query": query,
                "overall_score": round(
                    overall_score,
                    4
                ),
                "metrics": metric_results
            }

        except Exception as exc:

            return {
                "success": False,
                "error": (
                    f"DeepEval evaluation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            }