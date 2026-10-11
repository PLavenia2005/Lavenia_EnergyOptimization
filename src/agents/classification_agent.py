
import re


class ClassificationAgent:

    CLASSIFICATIONS = {
        "HIGH_CONSUMPTION": [
            "high consumption",
            "high energy",
            "excessive consumption",
            "too much energy",
            "energy usage is high",
            "high electricity",
            "consumption remains high",
            "unusually high energy consumption",
        ],

        "PEAK_DEMAND": [
            "peak demand",
            "peak hour",
            "peak hours",
            "peak consumption",
            "demand peak",
            "evening peak",
        ],

        "ANOMALY": [
            "anomaly",
            "unusual consumption",
            "abnormal consumption",
            "unexpected consumption",
            "sudden increase",
            "spike",
            "far above its normal pattern",
            "suddenly increased",
        ],

        "EQUIPMENT": [
            "equipment",
            "machine",
            "appliance",
            "device",
            "maintenance",
            "inefficiency",
            "inefficient",
        ],

        "STANDBY": [
            "standby",
            "idle",
            "inactive",
            "unused",
            "overnight",
        ],

        "HVAC": [
            "hvac",
            "heating",
            "cooling",
            "temperature",
            "air conditioning",
            "air conditioner",
        ],

        "LIGHTING": [
            "lighting",
            "lights",
            "light consumption",
        ],
    }

    def classify(self, query: str):
        query_lower = re.sub(
            r"\s+", " ", str(query or "").lower()
        ).strip()

        matched_categories = []

        # Detect the scenario described by the query.
        normal_scenario = any(
            phrase in query_lower
            for phrase in [
                "consumption is stable",
                "operating normally",
                "no anomalies",
                "no anomaly",
                "no unusual consumption",
                "maintain efficiency",
                "normal operating conditions",
            ]
        )

        off_peak_scenario = any(
            phrase in query_lower
            for phrase in [
                "off-peak",
                "off peak",
                "non-peak",
                "non peak",
            ]
        )

        # A normal, stable off-peak scenario should not be
        # classified as peak demand or anomalous merely because
        # those words appear in a negated or comparative phrase.
        if normal_scenario and off_peak_scenario:
            return {
                "primary_category": "GENERAL_ENERGY",
                "categories": ["GENERAL_ENERGY"],
                "query": query,
                "scenario": "NORMAL_OFF_PEAK",
            }

        for category, keywords in self.CLASSIFICATIONS.items():
            for keyword in keywords:
                if keyword not in query_lower:
                    continue

                # "No anomalies" and "no unusual consumption"
                # must not activate the ANOMALY category.
                if category == "ANOMALY":
                    negated = any(
                        phrase in query_lower
                        for phrase in [
                            "no anomalies",
                            "no anomaly",
                            "no unusual consumption",
                            "no abnormal consumption",
                        ]
                    )
                    if negated:
                        continue

                # "Off-peak hours" must not activate PEAK_DEMAND.
                if category == "PEAK_DEMAND" and off_peak_scenario:
                    continue

                matched_categories.append(category)
                break

        if not matched_categories:
            primary_category = "GENERAL_ENERGY"
            matched_categories = ["GENERAL_ENERGY"]
        else:
            primary_category = matched_categories[0]

        return {
            "primary_category": primary_category,
            "categories": matched_categories,
            "query": query,
            "scenario": (
                "OFF_PEAK" if off_peak_scenario else "UNSPECIFIED"
            ),
        }
