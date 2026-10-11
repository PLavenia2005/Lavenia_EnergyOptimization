from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert (
        "message"
        in data
    )

    assert (
        "version"
        in data
    )


def test_health():

    response = client.get(
        "/api/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["status"]
        == "healthy"
    )


def test_analyze():

    response = client.post(
        "/api/analyze",
        json={
            "query":
                "How can we reduce high energy consumption?",
            "top_k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "classification" in data
    assert "retrieval" in data
    assert "risk_prediction" in data


def test_analyze_with_filter():

    response = client.post(
        "/api/analyze",
        json={
            "query":
                "How can we reduce peak energy consumption?",
            "top_k": 3,
            "period": "Peak"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    for result in data[
        "retrieval"
    ]["results"]:

        assert (
            result["metadata"]["period"]
            .lower()
            == "peak"
        )


def test_search():

    response = client.post(
        "/api/search",
        json={
            "query":
                "high energy consumption",
            "top_k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "results" in data
    assert "count" in data


def test_recommend():

    response = client.post(
        "/api/recommend/rule",
        json={
            "query":
                "How can we reduce peak demand?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "classification" in data
    assert "recommendation" in data


def test_invalid_energy_query():

    response = client.post(
        "/api/analyze",
        json={
            "query":
                "What is my food today?",
            "top_k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is False
    assert "error" in data


def test_evaluate():

    response = client.post(
        "/api/evaluate",
        json={
            "query":
                "How can we reduce high energy consumption during peak hours?",
            "top_k": 5
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert (
        "detected_conditions"
        in data
    )

    assert (
        "relevant_case_count"
        in data
    )

    # Overall evaluation metrics
    assert (
        "metrics"
        in data
    )

    assert (
        "accuracy"
        in data["metrics"]
    )

    assert (
        "precision"
        in data["metrics"]
    )

    assert (
        "recall"
        in data["metrics"]
    )

    assert (
        "f1_score"
        in data["metrics"]
    )

    # Evaluation diagnostics
    assert (
        "diagnostics"
        in data
    )

    assert (
        "retrieved_case_ids"
        in data
    )