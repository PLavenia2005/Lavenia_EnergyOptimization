from typing import Any, Dict, TypedDict
from langgraph.graph import END, START, StateGraph
import os
from src.agents.guardrail_agent import GuardrailAgent



from src.agents.classification_agent import (

    ClassificationAgent

)



from src.agents.retrieval_agent import (

    RetrievalAgent

)



from src.agents.optimization_agent import (

    OptimizationAgent

)



from src.agents.risk_prediction_agent import (

    DemandRiskPredictionAgent

)



from src.utils.config import (

    PROCESSED_DATA_PATH

)



from src.agents.effectiveness_agent import (

    EnergySavingEffectivenessAgent

)



from src.agents.cost_reduction_agent import (

    EnergyCostReductionAgent

)



from src.agents.carbon_reduction_agent import (

    CarbonReductionEstimationAgent

)



from src.agents.feedback_agent import (

    FeedbackLoopAgent

)



from src.services.llm_judge import (

    LLMJudgeService

)



from src.services.deepeval_evaluation import (

    DeepEvalRecommendationService

)





class EnergyAnalysisState(TypedDict, total=False):
    """State passed through the LangGraph analysis workflow."""

    query: str
    top_k: int
    filters: Any
    baseline_kwh: Any
    tariff_per_kwh: Any
    emission_factor_kg_per_kwh: Any
    reduction_rate: Any
    period: str
    recommendation_mode: str
    result: Dict[str, Any]


class EnergyAgentCoordinator:

    """

    Central coordinator for the complete

    Energy Consumption Optimization Assistant.



    The coordinator orchestrates:



    1. Guardrail validation

    2. Query classification

    3. Hybrid retrieval

    4. Case prioritization

    5. Demand risk prediction

    6. Optimization recommendation

    7. Energy-saving effectiveness prediction

    8. LLM-as-a-Judge evaluation

    9. DeepEval recommendation evaluation

    10. Cost reduction estimation

    11. Carbon reduction estimation

    12. Multi-agent collaboration summary

    13. Feedback loop



    The project uses one unified analysis workflow.

    """



    # =========================================================
    # RISK QUERY DETECTION
    # =========================================================

    RISK_RELEVANT_CATEGORIES = {
        "HIGH_CONSUMPTION", "PEAK_DEMAND", "ANOMALY"
    }

    RISK_QUERY_TERMS = {
        "high consumption", "high energy consumption", "excessive consumption",
        "overconsumption", "energy spike", "consumption spike",
        "sudden increase", "sudden rise", "unusual consumption",
        "unusual energy", "abnormal consumption", "abnormal energy",
        "energy anomaly", "consumption anomaly", "anomaly", "peak demand",
        "peak hour", "peak hours", "critical consumption", "critical energy",
        "energy risk", "consumption risk", "energy surge", "consumption surge",
        "energy increase", "consumption increase"
    }

    def _is_risk_relevant_query(self, query, classification=None):
        """Determine whether Demand Risk Prediction should run."""
        query_lower = str(query or "").strip().lower()
        if isinstance(classification, dict):
            category = (classification.get("primary_category") or classification.get("category") or classification.get("intent") or "")
            if str(category).upper().strip() in self.RISK_RELEVANT_CATEGORIES:
                return True
        return any(term in query_lower for term in self.RISK_QUERY_TERMS)

    @staticmethod
    def _has_risk_filter(filters):
        """Return True when a risk-relevant sidebar filter was explicitly selected.

        A selected filter triggers assessment; it does not force the predicted label.
        """
        if not filters:
            return False

        if hasattr(filters, "model_dump") and callable(filters.model_dump):
            filters = filters.model_dump(exclude_none=True)
        elif hasattr(filters, "dict") and callable(filters.dict):
            filters = filters.dict(exclude_none=True)

        if not isinstance(filters, dict):
            return False

        normalized = {
            str(key).strip().lower(): value
            for key, value in filters.items()
        }

        level = normalized.get("consumption_level")
        if level is not None and str(level).strip().lower() not in {
            "", "all", "any", "none", "select"
        }:
            return True

        period = normalized.get("period")
        if period is not None and str(period).strip().lower() in {
            "peak", "off-peak", "off_peak"
        }:
            return True

        anomaly = normalized.get("is_anomaly")
        if anomaly is True or str(anomaly).strip().lower() in {
            "true", "1", "yes"
        }:
            return True

        return False

    def __init__(self):



        self.guardrail = (

            GuardrailAgent()

        )



        self.classifier = (

            ClassificationAgent()

        )



        self.retriever = (

            RetrievalAgent()

        )



        self.optimizer = (

            OptimizationAgent()

        )



        self.risk_predictor = (

            DemandRiskPredictionAgent(

                PROCESSED_DATA_PATH

            )

        )



        self.effectiveness_agent = (

            EnergySavingEffectivenessAgent()

        )



        self.cost_agent = (

            EnergyCostReductionAgent()

        )



        self.carbon_agent = (

            CarbonReductionEstimationAgent()

        )



        self.feedback_agent = (

            FeedbackLoopAgent()

        )



        self.llm_judge = (

            LLMJudgeService()

        )



        self.deepeval = (

            DeepEvalRecommendationService()

        )

        # LangGraph entry point for the existing unified analysis workflow.
        self.analysis_graph = self._build_analysis_graph()
                # Temporarily disable paid evaluators during local testing.
        self.enable_llm_judge = (
            os.getenv("ENABLE_LLM_JUDGE", "false").strip().lower()
            in ("1", "true", "yes", "on")
        )

        self.enable_deepeval = (
            os.getenv("ENABLE_DEEPEVAL", "false").strip().lower()
            in ("1", "true", "yes", "on")
        )


    # =========================================================

    # QUERY VALIDATION

    # =========================================================



    def _validate_query(

        self,

        query

    ):



        validation = (

            self.guardrail.validate(

                query

            )

        )



        if not validation["valid"]:



            return {

                "valid": False,

                "error": validation[

                    "message"

                ]

            }



        return {

            "valid": True,

            "query": validation[

                "query"

            ]

        }



    # =========================================================

    # RISK INPUT BUILDER

    # =========================================================



    def _build_risk_input(

        self,

        retrieved_cases

    ):



        if not retrieved_cases:



            return None



        case = retrieved_cases[0]



        metadata = case.get(

            "metadata",

            {}

        )



        document = case.get(

            "document",

            ""

        )



        values = {}



        for part in document.split("."):



            if ":" not in part:



                continue



            key, value = part.split(

                ":",

                1

            )



            values[

                key.strip().lower()

            ] = value.strip()



        def get_float(

            key,

            default=0.0

        ):



            try:



                return float(

                    values.get(

                        key,

                        default

                    )

                )



            except (

                ValueError,

                TypeError

            ):



                return default



        def get_int(

            key,

            default=0

        ):



            try:



                return int(

                    float(

                        values.get(

                            key,

                            default

                        )

                    )

                )



            except (

                ValueError,

                TypeError

            ):



                return default



        period = str(

            metadata.get(

                "period",

                ""

            )

        ).strip().lower()



        is_peak_from_metadata = (

            period == "peak"

        )



        is_peak_from_document = (

            str(

                values.get(

                    "is_peak_hour",

                    ""

                )

            ).strip().lower()

            in [

                "true",

                "1",

                "yes"

            ]

        )



        is_peak = (

            is_peak_from_metadata

            or is_peak_from_document

        )



        appliances = get_float(

            "energy consumption"

        )



        if appliances == 0.0:



            appliances = get_float(

                "appliances"

            )



        anomaly_score = get_float(

            "anomaly score"

        )



        if anomaly_score == 0.0:



            anomaly_score = get_float(

                "anomaly_score"

            )



        return {

            "appliances": appliances,



            "hour": get_int(

                "hour"

            ),



            "day_of_week_number": get_int(

                "day_of_week_number"

            ),



            "month": get_int(

                "month"

            ),



            "is_peak_hour": is_peak,



            "rolling_mean_24": get_float(

                "rolling_mean_24"

            ),



            "consumption_deviation": (

                get_float(

                    "consumption_deviation"

                )

            ),



            "anomaly_score": anomaly_score

        }



    # =========================================================

    # MAIN UNIFIED ANALYSIS

    # =========================================================



    def _build_analysis_graph(self):
        """Build a LangGraph wrapper around the existing unified analysis."""
        graph = StateGraph(EnergyAnalysisState)
        graph.add_node("run_unified_analysis", self._run_unified_analysis_node)
        graph.add_edge(START, "run_unified_analysis")
        graph.add_edge("run_unified_analysis", END)
        return graph.compile()

    def _run_unified_analysis_node(self, state: EnergyAnalysisState):
        """Execute the existing analysis implementation as one graph node."""
        result = self._analyze_impl(
            query=state.get("query", ""),
            top_k=state.get("top_k", 5),
            filters=state.get("filters"),
            baseline_kwh=state.get("baseline_kwh"),
            tariff_per_kwh=state.get("tariff_per_kwh"),
            emission_factor_kg_per_kwh=state.get("emission_factor_kg_per_kwh"),
            reduction_rate=state.get("reduction_rate"),
            period=state.get("period", "specified period"),
            recommendation_mode=state.get("recommendation_mode", "auto"),
        )
        return {"result": result}

    def analyze(
        self,
        query,
        top_k=5,
        filters=None,
        baseline_kwh=None,
        tariff_per_kwh=None,
        emission_factor_kg_per_kwh=None,
        reduction_rate=None,
        period="specified period",
        recommendation_mode="auto",
    ):
        """Run the unified workflow through LangGraph with the same public API."""
        initial_state: EnergyAnalysisState = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
            "baseline_kwh": baseline_kwh,
            "tariff_per_kwh": tariff_per_kwh,
            "emission_factor_kg_per_kwh": emission_factor_kg_per_kwh,
            "reduction_rate": reduction_rate,
            "period": period,
            "recommendation_mode": recommendation_mode,
        }
        graph_output = self.analysis_graph.invoke(initial_state)
        return graph_output["result"]


    def _analyze_impl(

        self,

        query,

        top_k=5,

        filters=None,

        baseline_kwh=None,

        tariff_per_kwh=None,

        emission_factor_kg_per_kwh=None,

        reduction_rate=None,

        period="specified period",

        recommendation_mode="auto"

    ):

        """

        Run the complete energy optimization workflow.



        Workflow:



        Guardrail

            ↓

        Classification

            ↓

        Hybrid Retrieval

            ↓

        Demand Risk Prediction

            ↓

        Optimization Recommendation

            ↓

        Effectiveness Prediction

            ↓

        LLM-as-a-Judge

            ↓

        DeepEval

            ↓

        Cost Reduction

            ↓

        Carbon Reduction

            ↓

        Collaboration Summary

        """



        # -----------------------------------------------------

        # 1. Guardrail

        # -----------------------------------------------------



        validation = (

            self._validate_query(

                query

            )

        )



        if not validation["valid"]:



            return {

                "success": False,

                "error": validation[

                    "error"

                ]

            }



        clean_query = (

            validation["query"]

        )



        # -----------------------------------------------------

        # 2. Classification

        # -----------------------------------------------------



        classification = (

            self.classifier.classify(

                clean_query

            )

        )



        # -----------------------------------------------------

        # 3. Hybrid Retrieval

        # -----------------------------------------------------



        retrieval = (

            self.retriever.retrieve(

                query=clean_query,

                top_k=top_k,

                filters=filters

            )

        )



        retrieved_cases = (

            retrieval.get(

                "results",

                []

            )

        )



        # -----------------------------------------------------

        # 4. Demand Risk Prediction
        # -----------------------------------------------------

        risk_relevant = (
            self._is_risk_relevant_query(clean_query, classification)
            or self._has_risk_filter(filters)
        )

        if risk_relevant and retrieved_cases:
            try:
                risk_input = self._build_risk_input(retrieved_cases)
                risk_prediction = self.risk_predictor.predict(risk_input) if risk_input else None
                if risk_prediction:
                    risk_prediction["assessment_status"] = "ASSESSED"
                    risk_prediction["risk_relevance"] = True
            except Exception as error:
                risk_prediction = {
                    "risk_level": "NOT_ASSESSED", "risk_probability": None,
                    "model_accuracy": None, "model": "Random Forest",
                    "assessment_status": "ERROR", "risk_relevance": True,
                    "reason": f"Demand risk prediction failed: {type(error).__name__}: {error}",
                    "input_features": None
                }
        else:
            risk_prediction = {
                "risk_level": "NOT_ASSESSED", "risk_probability": None,
                "model_accuracy": None, "model": "Random Forest",
                "assessment_status": "NOT_ASSESSED", "risk_relevance": False,
                "reason": (
                    "Risk assessment was skipped because the query was not "
                    "identified as risk-related, no risk-relevant filter was "
                    "selected, or no retrieved cases were available."
                ),
                "input_features": None
            }

        # 5. Optimization Recommendation
        # -----------------------------------------------------

        if recommendation_mode == "rule":
            recommendation = self.optimizer._rule_based_recommendation(classification, retrieved_cases)
            recommendation["selection_mode"] = "explicit_rule"
            recommendation["requested_method"] = "Rule-Based Fallback"
        elif recommendation_mode == "llm":
            recommendation = self.optimizer.recommend(classification, retrieved_cases)
            recommendation["selection_mode"] = "explicit_llm"
            recommendation["requested_method"] = "LLM + RAG"
        else:
            recommendation = self.optimizer.recommend(classification, retrieved_cases)
            recommendation["selection_mode"] = "automatic"
            recommendation["requested_method"] = "Automatic"

        recommendation_text = ""
        if isinstance(recommendation, dict):
            recommendation_text = str(recommendation.get("recommendation", ""))
        else:
            recommendation_text = str(recommendation or "")

        # 6. Energy-Saving Effectiveness

        # -----------------------------------------------------



        effectiveness = (

            self.effectiveness_agent.predict(

                retrieved_cases=(

                    retrieved_cases

                ),

                recommendation=(

                    recommendation_text

                ),

                risk_prediction=(

                    risk_prediction

                )

            )

        )



        # -----------------------------------------------------

        # 7. Build Evaluation Context
        # -----------------------------------------------------

        context = []
        for case in retrieved_cases:
            context.append(case.get("document", ""))

        judge_context = context[:3]

        
        # 8. LLM-as-a-Judge
        if self.enable_llm_judge:
            llm_judge_result = self._run_llm_judge(
                query=clean_query,
                recommendation=recommendation_text,
                context=judge_context
            )
        else:
            llm_judge_result = {
                "success": False,
                "skipped": True,
                "message": "LLM-as-a-Judge disabled for local testing."
            }

        # 9. DeepEval
        if self.enable_deepeval:
            try:
                deepeval_result = self.deepeval.evaluate_recommendation(
                    query=clean_query,
                    recommendation=recommendation_text,
                    context=context
                )
            except Exception as error:
                deepeval_result = {
                    "success": False,
                    "error": (
                        f"DeepEval failed: "
                        f"{type(error).__name__}: {error}"
                    )
                }
        else:
            deepeval_result = {
                "success": False,
                "skipped": True,
                "message": "DeepEval disabled for local testing."
            }



        # -----------------------------------------------------

        # 10. Cost Reduction Scenario

        # -----------------------------------------------------



        cost_reduction = {

            "success": False,

            "available": False,

            "message": (

                "Cost reduction scenario requires "

                "baseline_kwh, tariff_per_kwh, and "

                "reduction_rate."

            )

        }



        if (

            baseline_kwh is not None

            and tariff_per_kwh is not None

            and reduction_rate is not None

        ):



            cost_reduction = (

                self.cost_agent.estimate(

                    baseline_kwh=(

                        baseline_kwh

                    ),

                    tariff_per_kwh=(

                        tariff_per_kwh

                    ),

                    reduction_rate=(

                        reduction_rate

                    ),

                    period=period

                )

            )



            cost_reduction[

                "available"

            ] = cost_reduction.get(

                "success",

                False

            )



        # -----------------------------------------------------

        # 11. Carbon Reduction Scenario

        # -----------------------------------------------------



        carbon_reduction = {

            "success": False,

            "available": False,

            "message": (

                "Carbon reduction scenario requires "

                "baseline_kwh, "

                "emission_factor_kg_per_kwh, "

                "and reduction_rate."

            )

        }



        if (

            baseline_kwh is not None

            and emission_factor_kg_per_kwh is not None

            and reduction_rate is not None

        ):



            carbon_reduction = (

                self.carbon_agent.estimate(

                    baseline_energy_kwh=(

                        baseline_kwh

                    ),

                    emission_factor_kg_per_kwh=(

                        emission_factor_kg_per_kwh

                    ),

                    reduction_rate=(

                        reduction_rate

                    ),

                    period=period

                )

            )



            carbon_reduction[

                "available"

            ] = carbon_reduction.get(

                "success",

                False

            )



        # -----------------------------------------------------

        # 12. Collaboration Summary

        # -----------------------------------------------------



        risk_level = None



        if risk_prediction:



            risk_level = (

                risk_prediction.get(

                    "risk_level"

                )

            )



        primary_category = (

            classification.get(

                "primary_category"

            )

        )



        collaboration_status = (

            "ANALYSIS_COMPLETED"

        )



        if risk_level == "HIGH":



            collaboration_status = (

                "HIGH_RISK_OPTIMIZATION_ANALYSIS_COMPLETED"

            )



        elif risk_level == "MEDIUM":



            collaboration_status = (

                "MODERATE_RISK_OPTIMIZATION_ANALYSIS_COMPLETED"

            )



        elif recommendation:



            collaboration_status = (

                "OPTIMIZATION_RECOMMENDATION_AVAILABLE"

            )



        # -----------------------------------------------------

        # 13. Final Unified Result

        # -----------------------------------------------------



        return {



            "success": True,



            "query": clean_query,



            "classification": (

                classification

            ),



            "retrieval": retrieval,



            "risk_prediction": (

                risk_prediction

            ),



            "recommendation": (

                recommendation

            ),



            "recommendation_mode": recommendation_mode,

            "recommendation_selection": {
                "requested_method": recommendation.get("requested_method", "Automatic"),
                "selection_mode": recommendation.get("selection_mode", "automatic"),
                "actual_method": recommendation.get("method", "Unknown"),
                "fallback_used": recommendation.get("fallback_used", False),
            },

            "effectiveness": (

                effectiveness

            ),



            "llm_judge": (

                llm_judge_result

            ),



            "deepeval": (

                deepeval_result

            ),



            "cost_reduction": (

                cost_reduction

            ),



            "carbon_reduction": (

                carbon_reduction

            ),



            "collaboration": {



                "status": (

                    collaboration_status

                ),



                "primary_category": (

                    primary_category

                ),



                "risk_level": (

                    risk_level

                ),



                "retrieved_case_count": (

                    len(retrieved_cases)

                ),



                "decision_summary": (

                    "Guardrail, classification, "

                    "hybrid retrieval, demand risk, "

                    "optimization, effectiveness, "

                    "LLM-as-a-Judge, DeepEval, and "

                    "scenario estimation agents "

                    "contributed to the unified "

                    "energy optimization analysis."

                )

            },



            "agent_contributions": {



                "guardrail_agent": {



                    "status": "COMPLETED",



                    "output": (

                        "Query validated "

                        "successfully."

                    )

                },



                "classification_agent": {



                    "status": "COMPLETED",



                    "primary_category": (

                        classification.get(

                            "primary_category"

                        )

                    ),



                    "categories": (

                        classification.get(

                            "categories",

                            []

                        )

                    )

                },



                "retrieval_agent": {



                    "status": "COMPLETED",



                    "retrieved_cases": (

                        len(retrieved_cases)

                    ),



                    "filters": filters

                },



                "risk_prediction_agent": {



                    "status": (

                        "NOT_ASSESSED"
                        if risk_prediction
                        and risk_prediction.get("assessment_status") == "NOT_ASSESSED"
                        else ("COMPLETED" if risk_prediction else "NO_INPUT")
                    ),



                    "risk_level": (

                        risk_prediction.get(

                            "risk_level"

                        )

                        if risk_prediction

                        else None

                    ),



                    "risk_probability": (

                        risk_prediction.get(

                            "risk_probability"

                        )

                        if risk_prediction

                        else None

                    )

                },



                "optimization_agent": {



                    "status": (

                        "COMPLETED"

                        if recommendation

                        else "NO_RECOMMENDATION"

                    ),



                    "method": (

                        recommendation.get(

                            "method"

                        )

                        if isinstance(

                            recommendation,

                            dict

                        )

                        else None

                    ),



                    "priority": (

                        recommendation.get(

                            "priority"

                        )

                        if isinstance(

                            recommendation,

                            dict

                        )

                        else None

                    )

                },



                "effectiveness_agent": {



                    "status": (

                        "COMPLETED"

                        if (

                            effectiveness

                            and effectiveness.get(

                                "success",

                                False

                            )

                        )

                        else "FAILED"

                    )

                },



                "llm_judge_agent": {



                    "status": (

                        "COMPLETED"

                        if (

                            llm_judge_result

                            and llm_judge_result.get(

                                "success",

                                False

                            )

                        )

                        else "UNAVAILABLE"

                    )

                },



                "deepeval_agent": {



                    "status": (

                        "COMPLETED"

                        if (

                            deepeval_result

                            and deepeval_result.get(

                                "success",

                                False

                            )

                        )

                        else "UNAVAILABLE"

                    )

                },



                "cost_reduction_agent": {



                    "status": (

                        "COMPLETED"

                        if cost_reduction.get(

                            "available",

                            False

                        )

                        else "NOT_CONFIGURED"

                    )

                },



                "carbon_reduction_agent": {



                    "status": (

                        "COMPLETED"

                        if carbon_reduction.get(

                            "available",

                            False

                        )

                        else "NOT_CONFIGURED"

                    )

                },



                "feedback_loop_agent": {



                    "status": "READY"

                }

            }

        }



    # =========================================================

    # LLM-AS-A-JUDGE

    # =========================================================



    def _run_llm_judge(

        self,

        query,

        recommendation,

        context

    ):



        try:



            return self.llm_judge.judge(

                query=query,

                recommendation=recommendation,

                context=context

            )



        except Exception as error:



            return {

                "success": False,

                "error": (

                    f"LLM Judge failed: "

                    f"{type(error).__name__}: "

                    f"{error}"

                )

            }



    # =========================================================

    # MULTI-AGENT COLLABORATION

    # =========================================================



    def collaborate(

        self,

        query,

        top_k=5,

        filters=None

    ):



        # -----------------------------------------------------

        # 1. Guardrail

        # -----------------------------------------------------



        validation = (

            self._validate_query(

                query

            )

        )



        if not validation["valid"]:



            return {



                "success": False,



                "error": validation[

                    "error"

                ],



                "agent_contributions": {



                    "guardrail_agent": {



                        "status": "REJECTED",



                        "message": validation[

                            "error"

                        ]

                    }

                }

            }



        clean_query = (

            validation["query"]

        )



        # -----------------------------------------------------

        # 2. Classification

        # -----------------------------------------------------



        classification = (

            self.classifier.classify(

                clean_query

            )

        )



        # -----------------------------------------------------

        # 3. Retrieval

        # -----------------------------------------------------



        retrieval = (

            self.retriever.retrieve(

                query=clean_query,

                top_k=top_k,

                filters=filters

            )

        )



        retrieved_cases = (

            retrieval.get(

                "results",

                []

            )

        )



        # -----------------------------------------------------

        # 4. Demand Risk Prediction
        # -----------------------------------------------------

        risk_relevant = (
            self._is_risk_relevant_query(clean_query, classification)
            or self._has_risk_filter(filters)
        )

        if risk_relevant and retrieved_cases:
            try:
                risk_input = self._build_risk_input(retrieved_cases)
                risk_prediction = self.risk_predictor.predict(risk_input) if risk_input else None
                if risk_prediction:
                    risk_prediction["assessment_status"] = "ASSESSED"
                    risk_prediction["risk_relevance"] = True
            except Exception as error:
                risk_prediction = {
                    "risk_level": "NOT_ASSESSED", "risk_probability": None,
                    "model_accuracy": None, "model": "Random Forest",
                    "assessment_status": "ERROR", "risk_relevance": True,
                    "reason": f"Demand risk prediction failed: {type(error).__name__}: {error}",
                    "input_features": None
                }
        else:
            risk_prediction = {
                "risk_level": "NOT_ASSESSED", "risk_probability": None,
                "model_accuracy": None, "model": "Random Forest",
                "assessment_status": "NOT_ASSESSED", "risk_relevance": False,
                "reason": (
                    "Risk assessment was skipped because the query was not "
                    "identified as risk-related, no risk-relevant filter was "
                    "selected, or no retrieved cases were available."
                ),
                "input_features": None
            }

        # 5. Optimization

        # -----------------------------------------------------



        recommendation = (

            self.optimizer.recommend(

                classification,

                retrieved_cases

            )

        )



        # -----------------------------------------------------

        # Collaboration reasoning

        # -----------------------------------------------------



        risk_level = None



        if risk_prediction:



            risk_level = (

                risk_prediction.get(

                    "risk_level"

                )

            )



        primary_category = (

            classification.get(

                "primary_category"

            )

        )



        retrieval_count = len(

            retrieved_cases

        )



        # -----------------------------------------------------

        # Collaboration status

        # -----------------------------------------------------



        if (

            risk_level == "HIGH"

            and recommendation

        ):



            collaboration_status = (

                "HIGH_RISK_OPTIMIZATION_REQUIRED"

            )



        elif (

            risk_level == "MEDIUM"

            and recommendation

        ):



            collaboration_status = (

                "MODERATE_RISK_OPTIMIZATION_REQUIRED"

            )



        elif recommendation:



            collaboration_status = (

                "OPTIMIZATION_RECOMMENDATION_AVAILABLE"

            )



        else:



            collaboration_status = (

                "ANALYSIS_COMPLETED"

            )



        # -----------------------------------------------------

        # Final collaboration result

        # -----------------------------------------------------



        return {



            "success": True,



            "query": clean_query,



            "classification": (

                classification

            ),



            "retrieval": retrieval,



            "risk_prediction": (

                risk_prediction

            ),



            "recommendation": (

                recommendation

            ),



            "collaboration": {



                "status": (

                    collaboration_status

                ),



                "primary_category": (

                    primary_category

                ),



                "risk_level": (

                    risk_level

                ),



                "retrieved_case_count": (

                    retrieval_count

                ),



                "decision_summary": (

                    "Classification, historical "

                    "retrieval, demand risk "

                    "prediction, and optimization "

                    "agents collaborated to produce "

                    "the energy optimization result."

                )

            },



            "agent_contributions": {



                "guardrail_agent": {



                    "status": "COMPLETED",



                    "output": (

                        "Query validated as "

                        "energy-related."

                    )

                },



                "classification_agent": {



                    "status": "COMPLETED",



                    "primary_category": (

                        classification.get(

                            "primary_category"

                        )

                    ),



                    "categories": (

                        classification.get(

                            "categories",

                            []

                        )

                    )

                },



                "retrieval_agent": {



                    "status": "COMPLETED",



                    "retrieved_cases": (

                        retrieval_count

                    ),



                    "filters": filters

                },



                "risk_prediction_agent": {



                    "status": (

                        "NOT_ASSESSED"
                        if risk_prediction
                        and risk_prediction.get("assessment_status") == "NOT_ASSESSED"
                        else ("COMPLETED" if risk_prediction else "NO_INPUT")
                    ),



                    "risk_level": (

                        risk_prediction.get(

                            "risk_level"

                        )

                        if risk_prediction

                        else None

                    ),



                    "risk_probability": (

                        risk_prediction.get(

                            "risk_probability"

                        )

                        if risk_prediction

                        else None

                    )

                },



                "optimization_agent": {



                    "status": (

                        "COMPLETED"

                        if recommendation

                        else "NO_RECOMMENDATION"

                    ),



                    "method": (

                        recommendation.get(

                            "method"

                        )

                        if isinstance(

                            recommendation,

                            dict

                        )

                        else None

                    ),



                    "priority": (

                        recommendation.get(

                            "priority"

                        )

                        if isinstance(

                            recommendation,

                            dict

                        )

                        else None

                    )

                }

            }

        }



    # =========================================================

    # FEEDBACK LOOP

    # =========================================================



    def submit_feedback(

        self,

        query,

        recommendation,

        feedback,

        rating=None,

        comment=None

    ):



        return (

            self.feedback_agent

            .submit_feedback(

                query=query,

                recommendation=recommendation,

                feedback=feedback,

                rating=rating,

                comment=comment

            )

        )



    def feedback_summary(self):



        return (

            self.feedback_agent

            .get_feedback_summary()

        )



    def feedback_records(self):



        return (

            self.feedback_agent

            .get_records()

        )



    # =========================================================

    
    # =========================================================
    # LLM RECOMMENDATION SUPPORT
    # =========================================================

    def recommend_llm(
        self,
        query,
        top_k=5,
        filters=None,
        baseline_kwh=None,
        tariff_per_kwh=None,
        emission_factor_kg_per_kwh=None,
        reduction_rate=None,
        period="specified period",
    ):
        """Run LLM + RAG with optional cost/carbon scenarios."""
        return self.analyze(
            query=query,
            top_k=top_k,
            filters=filters,
            baseline_kwh=baseline_kwh,
            tariff_per_kwh=tariff_per_kwh,
            emission_factor_kg_per_kwh=emission_factor_kg_per_kwh,
            reduction_rate=reduction_rate,
            period=period,
            recommendation_mode="llm",
        )

    # =========================================================
    # RULE-BASED RECOMMENDATION SUPPORT
    # =========================================================

    def recommend_rule(
        self,
        query,
        top_k=5,
        filters=None,
        baseline_kwh=None,
        tariff_per_kwh=None,
        emission_factor_kg_per_kwh=None,
        reduction_rate=None,
        period="specified period",
    ):
        """Run rule-based recommendations with cost/carbon scenarios."""
        return self.analyze(
            query=query,
            top_k=top_k,
            filters=filters,
            baseline_kwh=baseline_kwh,
            tariff_per_kwh=tariff_per_kwh,
            emission_factor_kg_per_kwh=emission_factor_kg_per_kwh,
            reduction_rate=reduction_rate,
            period=period,
            recommendation_mode="rule",
        )


    # =========================================================
    # MAIN PROCESS

    # =========================================================



    def process(

        self,

        query,

        top_k=5,

        filters=None

    ):



        return self.analyze(

            query=query,

            top_k=top_k,

            filters=filters

        )