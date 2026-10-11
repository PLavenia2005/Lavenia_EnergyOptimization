from src.agents.llm import LLMService


class OptimizationAgent:

    RECOMMENDATIONS = {

        "HIGH_CONSUMPTION": {
            "recommendation": (
                "Review high-consumption periods "
                "and identify equipment or activities "
                "that can be shifted or optimized."
            ),
            "priority": "High"
        },

        "PEAK_DEMAND": {
            "recommendation": (
                "Shift flexible energy-intensive "
                "activities away from peak hours "
                "and stagger high-load operations."
            ),
            "priority": "High"
        },

        "ANOMALY": {
            "recommendation": (
                "Investigate the unusual consumption "
                "pattern and check affected equipment "
                "or operating conditions."
            ),
            "priority": "High"
        },

        "EQUIPMENT": {
            "recommendation": (
                "Inspect equipment operation, "
                "maintenance conditions, and operating "
                "time to identify opportunities "
                "for improving efficiency."
            ),
            "priority": "Medium"
        },

        "STANDBY": {
            "recommendation": (
                "Reduce unnecessary standby and "
                "idle loads by switching off "
                "equipment when it is not required."
            ),
            "priority": "Medium"
        },

        "HVAC": {
            "recommendation": (
                "Review heating and cooling "
                "setpoints and operating schedules "
                "to improve HVAC efficiency."
            ),
            "priority": "Medium"
        },

        "LIGHTING": {
            "recommendation": (
                "Use occupancy-based lighting "
                "controls and switch off unnecessary "
                "lighting during inactive periods."
            ),
            "priority": "Medium"
        },

        "GENERAL_ENERGY": {
            "recommendation": (
                "Review current energy usage and "
                "identify practical opportunities "
                "to improve overall efficiency."
            ),
            "priority": "Medium"
        }
    }

    CONSUMPTION_RECOMMENDATIONS = {

        "low": (
            "Maintain the current efficient usage "
            "pattern and continue monitoring consumption "
            "to ensure the low-consumption level is sustained."
        ),

        "normal": (
            "Maintain current efficient operating "
            "practices and monitor consumption trends "
            "for further efficiency opportunities."
        ),

        "moderate": (
            "Review current consumption patterns and "
            "identify opportunities to optimize equipment "
            "schedules, reduce unnecessary usage, and "
            "improve operating efficiency."
        ),

        "high": (
            "Identify the periods, equipment, or "
            "activities contributing to high consumption "
            "and reduce or shift avoidable energy usage."
        ),

        "critical": (
            "Prioritize immediate investigation of the "
            "highest-consuming periods or equipment and "
            "reduce avoidable energy loads where possible."
        )
    }

    def __init__(self):
        self.llm = LLMService()

    # =========================================================
    # CASE CONTEXT
    # =========================================================

    def _get_case_context(
        self,
        retrieved_cases
    ):

        if not retrieved_cases:
            return {
                "consumption_level": None,
                "period": None,
                "is_anomaly": False
            }

        case = retrieved_cases[0]

        metadata = case.get(
            "metadata",
            {}
        )

        consumption_level = str(
            metadata.get(
                "consumption_level",
                ""
            )
        ).strip().lower()

        period = str(
            metadata.get(
                "period",
                ""
            )
        ).strip().lower()

        anomaly_value = str(
            metadata.get(
                "is_anomaly",
                "false"
            )
        ).strip().lower()

        return {
            "consumption_level": (
                consumption_level
                if consumption_level
                else None
            ),
            "period": period,
            "is_anomaly": (
                anomaly_value == "true"
            )
        }

    # =========================================================
    # RULE-BASED RECOMMENDATION
    # =========================================================

    def _rule_based_recommendation(
        self,
        classification,
        retrieved_cases=None
    ):

        retrieved_cases = (
            retrieved_cases or []
        )

        category = classification[
            "primary_category"
        ]


        # Prioritize equipment-specific advice over
        # the consumption level of the retrieved case.
        if category == "EQUIPMENT":

            result = self.RECOMMENDATIONS.get(
                "EQUIPMENT",
                {
                    "recommendation": (
                        "Inspect equipment operation, "
                        "maintenance condition, operating "
                        "schedules, and energy efficiency. "
                        "Check for malfunctioning equipment "
                        "and avoid unnecessary operation."
                    ),
                    "priority": "Medium"
                }
            )

            return {
                "category": category,
                "consumption_level": None,
                "period": None,
                "is_anomaly": False,
                "recommendation": (
                    result["recommendation"]
                ),
                "priority": result["priority"],
                "method": "Rule-Based Fallback",
                "fallback_used": True
            }

        case_context = (
            self._get_case_context(
                retrieved_cases
            )
        )

        consumption_level = (
            case_context[
                "consumption_level"
            ]
        )

        period = (
            case_context[
                "period"
            ]
        )

        is_anomaly = (
            case_context[
                "is_anomaly"
            ]
        )

        if (
            consumption_level
            in self.CONSUMPTION_RECOMMENDATIONS
        ):

            recommendation = (
                self.CONSUMPTION_RECOMMENDATIONS[
                    consumption_level
                ]
            )

            priority = "Low"

            if consumption_level == "moderate":
                priority = "Medium"

            elif consumption_level in (
                "high",
                "critical"
            ):
                priority = "High"

        else:

            result = self.RECOMMENDATIONS.get(
                category,
                self.RECOMMENDATIONS[
                    "GENERAL_ENERGY"
                ]
            )

            recommendation = (
                result["recommendation"]
            )

            priority = (
                result["priority"]
            )

        if is_anomaly:

            recommendation += (
                " Since the historical case is "
                "marked as anomalous, investigate "
                "the unusual consumption pattern "
                "and affected operating conditions."
            )

            if priority == "Low":
                priority = "Medium"

        if period == "peak":

            recommendation += (
                " Because the case occurs during "
                "a peak period, consider shifting "
                "flexible loads to lower-demand periods."
            )

            if priority == "Low":
                priority = "Medium"

        return {
            "category": category,
            "consumption_level": consumption_level,
            "period": period,
            "is_anomaly": is_anomaly,
            "recommendation": recommendation,
            "priority": priority,
            "method": "Rule-Based Fallback",
            "fallback_used": True
        }

    # =========================================================
    # BUILD RAG CONTEXT
    # =========================================================

    def _build_context(
        self,
        classification,
        retrieved_cases
    ):

        category = (
            classification[
                "primary_category"
            ]
        )

        context = (
            f"Energy issue category: "
            f"{category}\n"
        )

        if not retrieved_cases:

            context += (
                "No historical cases were retrieved.\n"
            )

            return context

        context += (
            "\nRetrieved historical energy cases:\n"
        )

        for case in retrieved_cases[:5]:

            metadata = case.get(
                "metadata",
                {}
            )

            context += (
                f"- Case ID: "
                f"{case.get('case_id')}\n"
            )

            context += (
                f"  Description: "
                f"{case.get('document')}\n"
            )

            context += (
                f"  Consumption level: "
                f"{metadata.get('consumption_level')}\n"
            )

            context += (
                f"  Period: "
                f"{metadata.get('period')}\n"
            )

            context += (
                f"  Season: "
                f"{metadata.get('season')}\n"
            )

            context += (
                f"  Anomaly: "
                f"{metadata.get('is_anomaly')}\n"
            )

        return context

    # =========================================================
    # LLM RECOMMENDATION
    # =========================================================

    def _generate_llm_recommendation(
        self,
        classification,
        retrieved_cases
    ):

        context = self._build_context(
            classification,
            retrieved_cases
        )

        system_prompt = """
You are an energy consumption optimization
assistant.

Your job is to provide a practical recommendation
for the user's energy-related query using ONLY:

1. The user's query.
2. The retrieved historical energy cases.

The retrieved cases are evidence.
Do not invent information that is not present
in the query or retrieved cases.

IMPORTANT GROUNDING RULES:

- Do not invent exact times.
- Do not invent equipment faults.
- Do not invent unauthorized activity.
- Do not invent anomaly causes.
- Do not invent measurements.
- Do not invent savings percentages.
- Do not invent energy costs.
- Do not invent carbon reductions.
- Do not invent equipment names.
- Do not invent events or operating conditions.
- Do not claim that a case contains information
  that is not explicitly shown.
- Do not change HIGH consumption into CRITICAL
  unless the retrieved case explicitly says
  the consumption level is Critical.
- Do not automatically treat a case as anomalous.
- Only mention an anomaly when the retrieved
  case explicitly has Anomaly = True.


INTENT PRIORITY RULE:

- The user's query and classification define the primary goal.
- Use retrieved cases only as supporting evidence.
- Do not replace the user's requested goal with a different
  goal merely because retrieved cases mention it.
- For EQUIPMENT queries, prioritize equipment operation,
  maintenance, operating schedules, and efficiency checks.
- Mention peak-demand scheduling only when it directly
  supports the user's equipment-efficiency goal.
- Do not claim that equipment is faulty without evidence.


The user's query may contain multiple concepts.

For example:

"How can we reduce high energy consumption
during peak hours?"

contains both:

- high consumption
- peak period

Your recommendation should address both concepts.

Use semantic understanding of the query.
Do not depend on exact keyword matching.

The recommendation should be concise and practical.

Return ONLY the recommendation in this format:

RECOMMENDED ACTION:
<practical energy optimization action>

WHY:
<short explanation based on the query and retrieved cases>

PRIORITY:
<Low, Medium, or High>

Do not return:

- User Safety
- Safety status
- System messages
- Case IDs in the recommendation
- Retrieval scores
- JSON
- unrelated information
"""

        user_prompt = f"""
USER QUERY:
{classification.get("query", "")}

CLASSIFICATION:
{classification.get("primary_category", "")}

RETRIEVED HISTORICAL CASES:
{context}

Generate a grounded energy optimization
recommendation for the user's query.

Use the retrieved cases as evidence,
but do not add facts that are not present.
"""

        # IMPORTANT:
        # Only ONE LLM call.
        # The old duplicate call after return
        # was unreachable code.
        return self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

    # =========================================================
    # LLM RESPONSE VALIDATION
    # =========================================================

    def _is_valid_llm_recommendation(
        self,
        content,
        classification
    ):

        if not isinstance(
            content,
            str
        ):
            return False

        content = content.strip()

        if len(content) < 30:
            return False

        lower_content = (
            content.lower()
        )

        invalid_phrases = [
            "user safety",
            "safety: safe",
            "safety status",
            "system message",
            "i cannot",
            "i can't",
            "unable to provide"
        ]

        for phrase in invalid_phrases:

            if phrase in lower_content:
                return False

        required_sections = [
            "recommended action",
            "why",
            "priority"
        ]

        for section in required_sections:

            if section not in lower_content:
                return False

        energy_terms = [
            "energy",
            "consumption",
            "demand",
            "load",
            "electricity",
            "equipment",
            "hvac",
            "lighting",
            "peak",
            "efficiency",
            "usage"
        ]

        if not any(
            term in lower_content
            for term in energy_terms
        ):
            return False

        category = str(
            classification.get(
                "primary_category",
                ""
            )
        ).upper()

        if category == "PEAK_DEMAND":

            if not any(
                term in lower_content
                for term in [
                    "peak",
                    "demand",
                    "load"
                ]
            ):
                return False

        return True

    # =========================================================
    # MAIN RECOMMENDATION PIPELINE
    # =========================================================

    def recommend(
        self,
        classification,
        retrieved_cases=None
    ):

        retrieved_cases = (
            retrieved_cases or []
        )

        # Always prepare deterministic fallback.
        fallback = (
            self._rule_based_recommendation(
                classification,
                retrieved_cases
            )
        )

        # Attempt LLM generation.
        llm_result = (
            self._generate_llm_recommendation(
                classification,
                retrieved_cases
            )
        )

        # OpenRouter unavailable/error.
        if not isinstance(
            llm_result,
            dict
        ):
            return fallback

        if not llm_result.get(
            "success",
            False
        ):
            return fallback

        content = llm_result.get(
            "content"
        )

        # LLM returned an unsafe,
        # malformed, irrelevant, or incomplete result.
        if not self._is_valid_llm_recommendation(
            content,
            classification
        ):
            return fallback

        category = (
            classification[
                "primary_category"
            ]
        )

        case_context = (
            self._get_case_context(
                retrieved_cases
            )
        )

        # Priority is controlled by the retrieved
        # historical case and classification rather
        # than blindly trusting an arbitrary LLM value.
        consumption_level = (
            case_context[
                "consumption_level"
            ]
        )

        if consumption_level in (
            "high",
            "critical"
        ):

            final_priority = "High"

        elif consumption_level == "moderate":

            final_priority = "Medium"

        else:

            final_priority = (
                self.RECOMMENDATIONS.get(
                    category,
                    self.RECOMMENDATIONS[
                        "GENERAL_ENERGY"
                    ]
                )["priority"]
            )

        recommendation = {

            "category": category,

            "consumption_level":
                case_context[
                    "consumption_level"
                ],

            "period":
                case_context[
                    "period"
                ],

            "is_anomaly":
                case_context[
                    "is_anomaly"
                ],

            "recommendation":
                content.strip(),

            "priority":
                final_priority,

            "method":
                "OpenRouter LLM + RAG",

            "fallback_used":
                False
        }

        if retrieved_cases:

            top_case = (
                retrieved_cases[0]
            )

            recommendation[
                "supporting_case"
            ] = {

                "case_id":
                    top_case.get(
                        "case_id"
                    ),

                "semantic_score":
                    top_case.get(
                        "semantic_score"
                    ),

                "keyword_score":
                    top_case.get(
                        "keyword_score"
                    ),

                "hybrid_score":
                    top_case.get(
                        "hybrid_score"
                    ),

                "priority_score":
                    top_case.get(
                        "priority_score"
                    )
            }

        return recommendation