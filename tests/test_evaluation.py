from src.services.evaluation import (
    RetrievalEvaluationService
)


def test_evaluation_service_initialization():

    service = (
        RetrievalEvaluationService()
    )

    assert service.search is not None
    assert service.df is not None
    assert len(service.df) > 0


def test_query_condition_detection():

    service = (
        RetrievalEvaluationService()
    )

    conditions = (
        service._detect_conditions(
            "How can we reduce high energy consumption during peak hours?"
        )
    )

    assert (
        conditions["period"]
        == "Peak"
    )

    assert (
        "High"
        in conditions[
            "consumption_level"
        ]
    )


def test_dataset_ground_truth():

    service = (
        RetrievalEvaluationService()
    )

    conditions = {

        "period": "Peak",

        "consumption_level": [
            "High",
            "Critical"
        ],

        "is_anomaly": None
    }

    relevant_ids = (
        service._find_relevant_cases(
            conditions
        )
    )

    assert isinstance(
        relevant_ids,
        list
    )

    assert len(
        relevant_ids
    ) > 0


def test_evaluate_query():

    service = (
        RetrievalEvaluationService()
    )

    result = (
        service.evaluate_query(
            "How can we reduce high energy consumption during peak hours?",
            top_k=5
        )
    )

    assert result["success"] is True

    assert (
        "detected_conditions"
        in result
    )

    assert (
        "relevant_case_count"
        in result
    )

    assert (
        "metrics"
        in result
    )

    assert (
        "accuracy"
        in result["metrics"]
    )

    assert (
        "precision"
        in result["metrics"]
    )

    assert (
        "recall"
        in result["metrics"]
    )

    assert (
        "f1_score"
        in result["metrics"]
    )


def test_evaluation_metrics_are_valid():

    service = (
        RetrievalEvaluationService()
    )

    result = (
        service.evaluate_query(
            "How can we reduce peak demand?",
            top_k=5
        )
    )

    assert result["success"] is True

    metrics = result[
        "metrics"
    ]

    assert (
        0.0
        <= metrics["accuracy"]
        <= 1.0
    )

    assert (
        0.0
        <= metrics["precision"]
        <= 1.0
    )

    assert (
        0.0
        <= metrics["recall"]
        <= 1.0
    )

    assert (
        0.0
        <= metrics["f1_score"]
        <= 1.0
    )