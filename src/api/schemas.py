
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# MAIN ANALYSIS REQUEST
# ============================================================

class AnalyzeRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    # Retrieval filters
    season: Optional[str] = None
    period: Optional[str] = None
    consumption_level: Optional[str] = None
    is_anomaly: Optional[bool] = None

    # Cost scenario
    baseline_kwh: Optional[float] = Field(
        default=None,
        ge=0,
    )

    tariff_per_kwh: Optional[float] = Field(
        default=None,
        ge=0,
    )

    # Carbon scenario
    emission_factor_kg_per_kwh: Optional[float] = Field(
        default=None,
        gt=0,
    )

    # Assumed reduction percentage (0–100)
    reduction_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )

    period_label: str = Field(
        default="specified period",
        max_length=100,
    )


# ============================================================
# SEARCH REQUEST
# ============================================================

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    season: Optional[str] = None
    period: Optional[str] = None
    consumption_level: Optional[str] = None
    is_anomaly: Optional[bool] = None


# ============================================================
# RECOMMENDATION REQUEST
# ============================================================

class RecommendationRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    # Retrieval filters
    season: Optional[str] = None
    period: Optional[str] = None
    consumption_level: Optional[str] = None
    is_anomaly: Optional[bool] = None

    # Cost scenario
    baseline_kwh: Optional[float] = Field(
        default=None,
        ge=0,
    )

    tariff_per_kwh: Optional[float] = Field(
        default=None,
        ge=0,
    )

    # Carbon scenario
    emission_factor_kg_per_kwh: Optional[float] = Field(
        default=None,
        gt=0,
    )

    # Assumed reduction percentage (0–100)
    reduction_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )

    period_label: str = Field(
        default="specified period",
        max_length=100,
    )


# ============================================================
# RETRIEVAL EVALUATION REQUEST
# ============================================================

class EvaluationRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


# ============================================================
# FEEDBACK REQUEST
# ============================================================

class FeedbackRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
    )

    recommendation: str = Field(
        ...,
        min_length=1,
    )

    feedback: str = Field(
        ...,
        min_length=1,
    )

    rating: Optional[float] = Field(
        default=None,
        ge=1,
        le=5,
    )

    comment: Optional[str] = None
