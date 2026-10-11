from src.agents.feedback_agent import FeedbackLoopAgent


def test_feedback_submission():

    agent = FeedbackLoopAgent()

    result = agent.submit_feedback(
        query="How can I reduce peak energy consumption?",
        recommendation=(
            "Shift flexible loads away from peak demand periods "
            "where operationally possible."
        ),
        feedback="useful",
        rating=5,
        comment="The recommendation was relevant."
    )

    print("\nFeedback Submission:")
    print(result)

    assert result["success"] is True
    assert result["feedback_id"] == 1
    assert result["feedback"] == "positive"
    assert result["rating"] == 5
    assert result["message"] == "Feedback recorded successfully."


def test_negative_feedback():

    agent = FeedbackLoopAgent()

    result = agent.submit_feedback(
        query="How can I reduce energy consumption?",
        recommendation="Review energy usage patterns.",
        feedback="not useful",
        rating=2
    )

    assert result["success"] is True
    assert result["feedback"] == "negative"
    assert result["rating"] == 2


def test_feedback_summary():

    agent = FeedbackLoopAgent()

    agent.submit_feedback(
        query="Query 1",
        recommendation="Recommendation 1",
        feedback="useful",
        rating=5
    )

    agent.submit_feedback(
        query="Query 2",
        recommendation="Recommendation 2",
        feedback="useful",
        rating=4
    )

    agent.submit_feedback(
        query="Query 3",
        recommendation="Recommendation 3",
        feedback="not useful",
        rating=2
    )

    result = agent.get_feedback_summary()

    print("\nFeedback Summary:")
    print(result)

    assert result["success"] is True
    assert result["total_feedback"] == 3
    assert result["positive_feedback"] == 2
    assert result["negative_feedback"] == 1

    assert result["positive_rate"] == 66.67
    assert result["average_rating"] == 3.67
    assert result["feedback_status"] == "MODERATE"


def test_feedback_without_rating():

    agent = FeedbackLoopAgent()

    result = agent.submit_feedback(
        query="How can I reduce standby consumption?",
        recommendation="Turn off unused equipment.",
        feedback="helpful"
    )

    assert result["success"] is True
    assert result["feedback"] == "positive"
    assert result["rating"] is None


def test_invalid_feedback():

    agent = FeedbackLoopAgent()

    result = agent.submit_feedback(
        query="Energy query",
        recommendation="Energy recommendation",
        feedback="unknown"
    )

    assert result["success"] is False


def test_invalid_rating():

    agent = FeedbackLoopAgent()

    result = agent.submit_feedback(
        query="Energy query",
        recommendation="Energy recommendation",
        feedback="useful",
        rating=6
    )

    assert result["success"] is False


def test_missing_query():

    agent = FeedbackLoopAgent()

    result = agent.submit_feedback(
        query="",
        recommendation="Energy recommendation",
        feedback="useful"
    )

    assert result["success"] is False