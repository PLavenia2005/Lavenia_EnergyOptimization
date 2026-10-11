
from fastapi import APIRouter

from src.api.schemas import (
    AnalyzeRequest,
    SearchRequest,
    RecommendationRequest,
    EvaluationRequest,
    FeedbackRequest,
)

from src.agents.a2a import EnergyAgentCoordinator
from src.agents.guardrail_agent import GuardrailAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.services.evaluation import RetrievalEvaluationService


router = APIRouter(
    prefix="/api",
    tags=["Energy Optimization"],
)


# ============================================================
# SERVICES
# ============================================================

coordinator = EnergyAgentCoordinator()
guardrail = GuardrailAgent()
retriever = RetrievalAgent()
evaluation_service = RetrievalEvaluationService()


# ============================================================
# FILTER BUILDER
# ============================================================

def build_filters(request):
    filters = {}

    if request.season is not None:
        filters["season"] = request.season

    if request.period is not None:
        filters["period"] = request.period

    if request.consumption_level is not None:
        filters["consumption_level"] = request.consumption_level

    if request.is_anomaly is not None:
        filters["is_anomaly"] = request.is_anomaly

    return filters if filters else None


# ============================================================
# SCENARIO ARGUMENT BUILDER
# ============================================================

def build_scenario_arguments(request):
    """
    Pass optional cost and carbon scenario inputs from the
    validated API request to the coordinator.
    """
    return {
        "baseline_kwh": request.baseline_kwh,
        "tariff_per_kwh": request.tariff_per_kwh,
        "emission_factor_kg_per_kwh": (
            request.emission_factor_kg_per_kwh
        ),
        "reduction_rate": request.reduction_rate,
        "period": request.period_label,
    }


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Energy Optimization Assistant",
    }


# ============================================================
# MAIN UNIFIED ANALYSIS
# ============================================================

@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    filters = build_filters(request)

    return coordinator.analyze(
        query=request.query,
        top_k=request.top_k,
        filters=filters,
        **build_scenario_arguments(request),
    )


# ============================================================
# SEARCH SUPPORT API
# ============================================================

@router.post("/search")
def search(request: SearchRequest):
    validation = guardrail.validate(request.query)

    if not validation["valid"]:
        return {
            "success": False,
            "error": validation["message"],
        }

    filters = build_filters(request)

    result = retriever.retrieve(
        query=validation["query"],
        top_k=request.top_k,
        filters=filters,
    )

    return {
        "success": True,
        "query": validation["query"],
        "results": result["results"],
        "count": result["count"],
    }


# ============================================================
# LLM RECOMMENDATION API
# ============================================================

@router.post("/recommend/llm")
def recommend_llm(request: RecommendationRequest):
    filters = build_filters(request)

    return coordinator.recommend_llm(
        query=request.query,
        top_k=request.top_k,
        filters=filters,
        **build_scenario_arguments(request),
    )


# ============================================================
# RULE-BASED RECOMMENDATION API
# ============================================================

@router.post("/recommend/rule")
def recommend_rule(request: RecommendationRequest):
    filters = build_filters(request)

    return coordinator.recommend_rule(
        query=request.query,
        top_k=request.top_k,
        filters=filters,
        **build_scenario_arguments(request),
    )


# ============================================================
# RETRIEVAL EVALUATION API
# ============================================================

@router.post("/evaluate")
def evaluate(request: EvaluationRequest):
    return evaluation_service.evaluate_query(
        query=request.query,
        top_k=request.top_k,
    )


# ============================================================
# FEEDBACK SUBMISSION
# ============================================================

@router.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    return coordinator.submit_feedback(
        query=request.query,
        recommendation=request.recommendation,
        feedback=request.feedback,
        rating=request.rating,
        comment=request.comment,
    )


# ============================================================
# FEEDBACK SUMMARY
# ============================================================

@router.get("/feedback/summary")
def feedback_summary():
    return coordinator.feedback_summary()


# ============================================================
# FEEDBACK RECORDS
# ============================================================

@router.get("/feedback/records")
def feedback_records():
    return coordinator.feedback_records()
