class CarbonReductionEstimationAgent:

    def estimate(
        self,
        baseline_energy_kwh,
        emission_factor_kg_per_kwh,
        reduction_rate,
        period="specified period"
    ):
        # Validate baseline energy
        try:
            baseline_energy_kwh = float(baseline_energy_kwh)
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": "Baseline energy must be a valid number."
            }

        if baseline_energy_kwh <= 0:
            return {
                "success": False,
                "error": "Baseline energy must be greater than zero."
            }

        # Validate emission factor
        try:
            emission_factor_kg_per_kwh = float(
                emission_factor_kg_per_kwh
            )
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": "Emission factor must be a valid number."
            }

        if emission_factor_kg_per_kwh <= 0:
            return {
                "success": False,
                "error": "Emission factor must be greater than zero."
            }

        # Validate reduction rate
        try:
            reduction_rate = float(reduction_rate)
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": "Reduction rate must be a valid number."
            }

        # Accept both decimal and percentage input
        if reduction_rate > 1:
            reduction_rate = reduction_rate / 100

        if reduction_rate < 0 or reduction_rate > 1:
            return {
                "success": False,
                "error": "Reduction rate must be between 0% and 100%."
            }

        # Baseline carbon emissions
        baseline_emissions = (
            baseline_energy_kwh
            * emission_factor_kg_per_kwh
        )

        # Estimated energy reduction
        estimated_energy_reduction = (
            baseline_energy_kwh
            * reduction_rate
        )

        # Estimated carbon reduction
        estimated_carbon_reduction = (
            estimated_energy_reduction
            * emission_factor_kg_per_kwh
        )

        # Remaining energy
        remaining_energy = (
            baseline_energy_kwh
            - estimated_energy_reduction
        )

        # Remaining emissions
        remaining_emissions = (
            baseline_emissions
            - estimated_carbon_reduction
        )

        return {
            "success": True,
            "period": period,
            "baseline_energy_kwh": round(
                baseline_energy_kwh, 2
            ),
            "emission_factor_kg_per_kwh": round(
                emission_factor_kg_per_kwh, 4
            ),
            "baseline_emissions_kg_co2e": round(
                baseline_emissions, 2
            ),
            "assumed_reduction_rate": round(
                reduction_rate * 100, 2
            ),
            "estimated_energy_reduction_kwh": round(
                estimated_energy_reduction, 2
            ),
            "estimated_carbon_reduction_kg_co2e": round(
                estimated_carbon_reduction, 2
            ),
            "estimated_remaining_energy_kwh": round(
                remaining_energy, 2
            ),
            "estimated_remaining_emissions_kg_co2e": round(
                remaining_emissions, 2
            ),
            "estimation_method": (
                "Scenario-Based Carbon Reduction Estimator"
            ),
            "historical_emission_factor_available": False,
            "historical_carbon_savings_available": False,
            "is_scenario_estimate": True,
            "interpretation": (
                "Estimated carbon reduction calculated from "
                "the supplied baseline energy, emission factor, "
                "and assumed reduction rate. This is a scenario "
                "estimate and not a measured historical carbon saving."
            )
        }


carbon_reduction_agent = CarbonReductionEstimationAgent()