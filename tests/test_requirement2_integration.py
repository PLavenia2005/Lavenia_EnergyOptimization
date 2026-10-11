from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


# ============================================================
# FULL REQUIREMENT 2 ANALYSIS
# ============================================================

def test_full_analysis_endpoint():

    response = client.post(
        "/api/analyze/full",
        json={
            "query": (
                "How can we reduce high "
                "energy consumption during peak hours?"
            ),
            "top_k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert (
        data["requirement"]
        == "Requirement 2 - Full Analysis"
    )

    assert "classification" in data

    assert "retrieval" in data

    assert "risk_prediction" in data

    assert "recommendation" in data

    assert "effectiveness" in data

    assert "llm_judge" in data

    assert "deepeval" in data

    assert "cost_reduction" in data

    assert "carbon_reduction" in data

    assert "collaboration" in data

    assert "agent_contributions" in data


# ============================================================
# COST + CARBON SCENARIO
# ============================================================

def test_full_analysis_with_scenarios():

    response = client.post(
        "/api/analyze/full",
        json={
            "query": (
                "How can we reduce "
                "peak energy demand?"
            ),
            "top_k": 3,
            "baseline_kwh": 1000,
            "tariff_per_kwh": 8.0,
            "emission_factor_kg_per_kwh": 0.7,
            "reduction_rate": 10,
            "period_label": "monthly scenario"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    cost = (
        data["cost_reduction"]
    )

    assert cost["success"] is True

    assert (
        cost[
            "estimated_energy_reduction_kwh"
        ]
        == 100.0
    )

    assert (
        cost[
            "estimated_cost_reduction"
        ]
        == 800.0
    )

    carbon = (
        data["carbon_reduction"]
    )

    assert carbon["success"] is True

    assert (
        carbon[
            "estimated_energy_reduction_kwh"
        ]
        == 100.0
    )

    assert (
        carbon[
            "estimated_carbon_reduction_kg_co2e"
        ]
        == 70.0
    )


# ============================================================
# NO HIDDEN COST/CARBON ASSUMPTIONS
# ============================================================

def test_scenario_estimation_is_not_guessed():

    response = client.post(
        "/api/analyze/full",
        json={
            "query": (
                "How can we reduce "
                "high energy consumption?"
            ),
            "top_k": 2
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert (
        data["cost_reduction"][
            "available"
        ] is False
    )

    assert (
        data["carbon_reduction"][
            "available"
        ] is False
    )


# ============================================================
# FEEDBACK
# ============================================================

def test_feedback_submission():

    response = client.post(
        "/api/feedback",
        json={
            "query": (
                "How can we reduce "
                "peak energy consumption?"
            ),
            "recommendation": (
                "Shift flexible loads "
                "away from peak demand periods."
            ),
            "feedback": "useful",
            "rating": 5,
            "comment": (
                "The recommendation was useful."
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert (
        data["feedback"]
        == "positive"
    )


# ============================================================
# FEEDBACK SUMMARY
# ============================================================

def test_feedback_summary():

    response = client.get(
        "/api/feedback/summary"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert "total_feedback" in data

    assert "positive_feedback" in data

    assert "negative_feedback" in data

    assert "positive_rate" in data

    assert "feedback_status" in data