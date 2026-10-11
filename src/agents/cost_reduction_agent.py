from typing import Any, Dict


class EnergyCostReductionAgent:

    """
    Estimates potential electricity-cost reduction from an explicitly
    supplied energy baseline, tariff, and reduction assumption.

    IMPORTANT:
    The project dataset does not contain historical electricity tariffs
    or measured optimization savings. Therefore, this agent does not
    invent either value.

    The reduction rate is an explicit scenario assumption.
    """

    def __init__(self):

        self.model = (
            "Scenario-Based Cost Reduction Estimator"
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_positive(
        self,
        value: Any,
        field_name: str
    ):

        try:

            value = float(value)

        except (TypeError, ValueError):

            raise ValueError(
                f"{field_name} must be a numeric value."
            )

        if value < 0:

            raise ValueError(
                f"{field_name} cannot be negative."
            )

        return value

    # ============================================================
    # MAIN ESTIMATION
    # ============================================================

    def estimate(
        self,
        baseline_kwh: Any,
        tariff_per_kwh: Any,
        reduction_rate: Any,
        period: str = "specified period"
    ) -> Dict[str, Any]:

        try:

            baseline_kwh = self._validate_positive(
                baseline_kwh,
                "baseline_kwh"
            )

            tariff_per_kwh = self._validate_positive(
                tariff_per_kwh,
                "tariff_per_kwh"
            )

            reduction_rate = self._validate_positive(
                reduction_rate,
                "reduction_rate"
            )

        except ValueError as error:

            return {
                "success": False,
                "error": str(error)
            }

        # Accept either:
        # 0.10  -> 10%
        # 10    -> 10%
        if reduction_rate > 1:

            reduction_rate = (
                reduction_rate / 100
            )

        if reduction_rate > 1:

            return {
                "success": False,
                "error": (
                    "reduction_rate must be between "
                    "0 and 1, or between 0 and 100 "
                    "when supplied as a percentage."
                )
            }

        baseline_cost = (
            baseline_kwh
            * tariff_per_kwh
        )

        estimated_energy_reduction = (
            baseline_kwh
            * reduction_rate
        )

        estimated_cost_reduction = (
            estimated_energy_reduction
            * tariff_per_kwh
        )

        estimated_remaining_energy = (
            baseline_kwh
            - estimated_energy_reduction
        )

        estimated_remaining_cost = (
            estimated_remaining_energy
            * tariff_per_kwh
        )

        return {

            "success": True,

            "period": period,

            "baseline_energy_kwh": round(
                baseline_kwh,
                4
            ),

            "tariff_per_kwh": round(
                tariff_per_kwh,
                4
            ),

            "baseline_cost": round(
                baseline_cost,
                4
            ),

            "assumed_reduction_rate": round(
                reduction_rate * 100,
                2
            ),

            "estimated_energy_reduction_kwh": round(
                estimated_energy_reduction,
                4
            ),

            "estimated_cost_reduction": round(
                estimated_cost_reduction,
                4
            ),

            "estimated_remaining_energy_kwh": round(
                estimated_remaining_energy,
                4
            ),

            "estimated_remaining_cost": round(
                estimated_remaining_cost,
                4
            ),

            "currency": "user-specified tariff currency",

            "estimation_method": self.model,

            "historical_tariff_available": False,

            "historical_savings_available": False,

            "is_scenario_estimate": True,

            "interpretation": (
                "Estimated cost reduction calculated from "
                "the supplied baseline energy, electricity "
                "tariff, and assumed reduction rate. "
                "This is a scenario estimate and not a "
                "measured historical saving."
            )
        }