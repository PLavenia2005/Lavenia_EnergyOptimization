from src.agents.carbon_reduction_agent import (
    CarbonReductionEstimationAgent
)


def test_carbon_reduction_estimation():

    agent = CarbonReductionEstimationAgent()

    result = agent.estimate(
        baseline_energy_kwh=1000,
        emission_factor_kg_per_kwh=0.7,
        reduction_rate=10,
        period="monthly"
    )

    print("\nCarbon Reduction Prediction:")
    print(result)

    assert result["success"] is True

    assert result["baseline_energy_kwh"] == 1000.0

    assert result["emission_factor_kg_per_kwh"] == 0.7

    assert result["baseline_emissions_kg_co2e"] == 700.0

    assert result["assumed_reduction_rate"] == 10.0

    assert result["estimated_energy_reduction_kwh"] == 100.0

    assert result["estimated_carbon_reduction_kg_co2e"] == 70.0

    assert result["estimated_remaining_energy_kwh"] == 900.0

    assert result["estimated_remaining_emissions_kg_co2e"] == 630.0

    assert result["is_scenario_estimate"] is True

    assert result["historical_emission_factor_available"] is False

    assert result["historical_carbon_savings_available"] is False


def test_percentage_reduction_input():

    agent = CarbonReductionEstimationAgent()

    result = agent.estimate(
        baseline_energy_kwh=2000,
        emission_factor_kg_per_kwh=0.5,
        reduction_rate=20
    )

    assert result["success"] is True

    assert result["assumed_reduction_rate"] == 20.0

    assert result["estimated_energy_reduction_kwh"] == 400.0

    assert result["estimated_carbon_reduction_kg_co2e"] == 200.0


def test_invalid_negative_baseline():

    agent = CarbonReductionEstimationAgent()

    result = agent.estimate(
        baseline_energy_kwh=-1000,
        emission_factor_kg_per_kwh=0.7,
        reduction_rate=10
    )

    assert result["success"] is False


def test_invalid_emission_factor():

    agent = CarbonReductionEstimationAgent()

    result = agent.estimate(
        baseline_energy_kwh=1000,
        emission_factor_kg_per_kwh=-0.7,
        reduction_rate=10
    )

    assert result["success"] is False


def test_invalid_reduction_rate():

    agent = CarbonReductionEstimationAgent()

    result = agent.estimate(
        baseline_energy_kwh=1000,
        emission_factor_kg_per_kwh=0.7,
        reduction_rate=120
    )

    assert result["success"] is False