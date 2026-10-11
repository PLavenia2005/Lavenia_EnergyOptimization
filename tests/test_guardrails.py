from src.agents.guardrail_agent import GuardrailAgent


def test_valid_energy_query():

    agent = GuardrailAgent()

    result = agent.validate(
        "How can we reduce energy consumption?"
    )

    assert result["valid"] is True


def test_empty_query():

    agent = GuardrailAgent()

    result = agent.validate("")

    assert result["valid"] is False

    assert (
        result["message"]
        == "Query cannot be empty."
    )


def test_short_query():

    agent = GuardrailAgent()

    result = agent.validate("abc")

    assert result["valid"] is False


def test_long_query():

    agent = GuardrailAgent()

    result = agent.validate(
        "energy " * 100
    )

    assert result["valid"] is False


def test_non_energy_query():

    agent = GuardrailAgent()

    result = agent.validate(
        "What is my food today?"
    )

    assert result["valid"] is False

    assert (
        "energy consumption"
        in result["message"]
    )


def test_energy_consumption_query():

    agent = GuardrailAgent()

    result = agent.validate(
        "Why is electricity consumption high?"
    )

    assert result["valid"] is True


def test_peak_demand_query():

    agent = GuardrailAgent()

    result = agent.validate(
        "How can we reduce peak demand?"
    )

    assert result["valid"] is True


def test_equipment_query():

    agent = GuardrailAgent()

    result = agent.validate(
        "Why is equipment consuming too much energy?"
    )

    assert result["valid"] is True


def test_hvac_query():

    agent = GuardrailAgent()

    result = agent.validate(
        "How can we reduce HVAC energy consumption?"
    )

    assert result["valid"] is True

