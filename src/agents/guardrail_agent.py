import re


class GuardrailAgent:

    ENERGY_KEYWORDS = [
        "energy",
        "electricity",
        "power consumption",
        "energy consumption",
        "energy usage",
        "electricity usage",
        "electricity consumption",
        "power usage",
        "power demand",
        "peak demand",
        "peak hour",
        "peak hours",
        "consumption",
        "usage",
        "load",
        "appliance",
        "equipment",
        "machine",
        "device",
        "standby",
        "idle",
        "hvac",
        "heating",
        "cooling",
        "temperature",
        "lighting",
        "lights",
        "maintenance",
        "anomaly",
        "unusual consumption",
        "energy efficiency",
        "energy saving",
        "energy savings",
        "reduce energy",
        "reduce consumption",
        "reduce electricity",
        "high consumption",
        "high energy",
        "electricity cost",
        "energy cost",
        "peak consumption"
    ]

    def validate(self, query: str):

        # ---------------------------------------------
        # Basic validation
        # ---------------------------------------------

        if query is None:

            return {
                "valid": False,
                "message": "Query is required."
            }

        query = query.strip()

        if not query:

            return {
                "valid": False,
                "message": "Query cannot be empty."
            }

        if len(query) < 5:

            return {
                "valid": False,
                "message": (
                    "Query is too short. "
                    "Please provide more details."
                )
            }

        if len(query) > 500:

            return {
                "valid": False,
                "message": (
                    "Query is too long. "
                    "Please keep it under 500 characters."
                )
            }

        if not re.search(
            r"[a-zA-Z0-9]",
            query
        ):

            return {
                "valid": False,
                "message": (
                    "Query must contain "
                    "meaningful text."
                )
            }

        # ---------------------------------------------
        # Energy domain validation
        # ---------------------------------------------

        query_lower = query.lower()

        is_energy_related = any(
            keyword in query_lower
            for keyword in self.ENERGY_KEYWORDS
        )

        if not is_energy_related:

            return {
                "valid": False,
                "message": (
                    "This assistant only handles "
                    "energy consumption and optimization "
                    "questions. Please enter an "
                    "energy-related query."
                )
            }

        # ---------------------------------------------
        # Validation successful
        # ---------------------------------------------

        return {
            "valid": True,
            "message": "Query passed validation.",
            "query": query
        }

