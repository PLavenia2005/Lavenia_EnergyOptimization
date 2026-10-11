from src.rag.hybrid_search import HybridSearch


def test_hybrid_search_initialization():

    search = HybridSearch()

    assert search.embedding_model is not None
    assert search.vector_store is not None
    assert search.keyword_search is not None


def test_hybrid_search_returns_results():

    search = HybridSearch()

    results = search.search(
        "high energy consumption",
        top_k=5
    )

    assert isinstance(
        results,
        list
    )

    assert len(results) <= 5


def test_hybrid_result_structure():

    search = HybridSearch()

    results = search.search(
        "high energy consumption",
        top_k=3
    )

    if results:

        result = results[0]

        assert "case_id" in result
        assert "document" in result
        assert "semantic_score" in result
        assert "keyword_score" in result
        assert "hybrid_score" in result
        assert "priority_score" in result
        assert "metadata" in result


def test_hybrid_scores_are_valid():

    search = HybridSearch()

    results = search.search(
        "high energy consumption",
        top_k=5
    )

    for result in results:

        assert (
            0.0
            <= result["semantic_score"]
            <= 1.0
        )

        assert (
            0.0
            <= result["keyword_score"]
            <= 1.0
        )

        assert (
            0.0
            <= result["hybrid_score"]
            <= 1.0
        )

        assert (
            0.0
            <= result["priority_score"]
            <= 1.0
        )


def test_peak_metadata_filter():

    search = HybridSearch()

    results = search.search(
        "high energy consumption",
        top_k=5,
        filters={
            "period": "Peak"
        }
    )

    for result in results:

        assert (
            result["metadata"]["period"]
            .lower()
            == "peak"
        )