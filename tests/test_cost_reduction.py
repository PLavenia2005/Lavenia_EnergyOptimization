from src.agents.cost_reduction_agent import (
    EnergyCostReductionAgent
)


def test_cost_reduction_estimation():

    agent = EnergyCostReductionAgent()

    result = agent.estimate(
        baseline_kwh=1000,
        tariff_per_kwh=8,
        reduction_rate=0.10,
        period="monthly"
    )

    print("\nCost Reduction Prediction:")
    print(result)

    assert result["success"] is True

    assert (
        result["baseline_energy_kwh"]
        == 1000
    )

    assert (
        result["tariff_per_kwh"]
        == 8
    )

    assert (
        result["baseline_cost"]
        == 8000
    )

    assert (
        result["assumed_reduction_rate"]
        == 10
    )

    assert (
        result[
            "estimated_energy_reduction_kwh"
        ]
        == 100
    )

    assert (
        result[
            "estimated_cost_reduction"
        ]
        == 800
    )

    assert (
        result[
            "estimated_remaining_energy_kwh"
        ]
        == 900
    )

    assert (
        result[
            "estimated_remaining_cost"
        ]
        == 7200
    )

    assert (
        result["historical_savings_available"]
        is False
    )

    assert (
        result["is_scenario_estimate"]
        is True
    )


def test_percentage_reduction_input():

    agent = EnergyCostReductionAgent()

    result = agent.estimate(
        baseline_kwh=500,
        tariff_per_kwh=10,
        reduction_rate=20
    )

    assert result["success"] is True

    assert (
        result["assumed_reduction_rate"]
        == 20
    )

    assert (
        result[
            "estimated_energy_reduction_kwh"
        ]
        == 100
    )

    assert (
        result[
            "estimated_cost_reduction"
        ]
        == 1000
    )


def test_invalid_negative_baseline():

    agent = EnergyCostReductionAgent()

    result = agent.estimate(
        baseline_kwh=-100,
        tariff_per_kwh=8,
        reduction_rate=0.10
    )

    assert result["success"] is False

    assert (
        "baseline_kwh cannot be negative"
        in result["error"]
    )


def test_invalid_reduction_rate():

    agent = EnergyCostReductionAgent()

    result = agent.estimate(
        baseline_kwh=1000,
        tariff_per_kwh=8,
        reduction_rate=150
    )

    assert result["success"] is False

    assert (
        "reduction_rate must be between"
        in result["error"]
    )