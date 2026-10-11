from src.rag.keyword_search import KeywordSearch


def test_tokenize():

    result = KeywordSearch.tokenize(
        "High Energy Consumption"
    )

    assert result == [
        "high",
        "energy",
        "consumption"
    ]


def test_keyword_search():

    search = KeywordSearch(
        "data/processed/energy_cases.csv"
    )

    results = search.search(
        "high energy consumption",
        top_k=5
    )

    assert isinstance(
        results,
        list
    )

    assert len(results) <= 5


def test_keyword_result_structure():

    search = KeywordSearch(
        "data/processed/energy_cases.csv"
    )

    results = search.search(
        "energy consumption",
        top_k=3
    )

    if results:

        result = results[0]

        assert "case_id" in result
        assert "document" in result
        assert "score" in result


def test_keyword_search_returns_valid_case_ids():

    search = KeywordSearch(
        "data/processed/energy_cases.csv"
    )

    results = search.search(
        "energy",
        top_k=5
    )

    valid_ids = set(
        search.df["case_id"]
    )

    for result in results:

        assert result["case_id"] in valid_ids
