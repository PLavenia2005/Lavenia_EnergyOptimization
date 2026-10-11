import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DemandRiskPredictionAgent:

    def __init__(self, data_path):

        self.data_path = data_path

        self.model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            max_depth=10,
            min_samples_leaf=5
        )

        self.feature_columns = [
            "appliances",
            "hour",
            "day_of_week_number",
            "month",
            "is_peak_hour",
            "rolling_mean_24",
            "consumption_deviation",
            "anomaly_score"
        ]

        self.is_trained = False
        self.accuracy = None

        self._train_model()

    # =========================================================
    # LOAD DATA
    # =========================================================

    def _load_data(self):

        df = pd.read_csv(
            self.data_path
        )

        required_columns = (
            self.feature_columns
            + ["consumption_level"]
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                f"{missing_columns}"
            )

        return df

    def _parse_bool(self, value):
        """Convert supported Boolean representations safely."""
        if isinstance(value, bool):
            return value

        if pd.isna(value):
            raise ValueError("is_peak_hour cannot be missing.")

        if isinstance(value, (int, float)):
            if value in (0, 1):
                return bool(value)
            raise ValueError(f"Invalid Boolean value for is_peak_hour: {value}")

        normalized = str(value).strip().lower()

        if normalized in ("true", "1", "yes"):
            return True

        if normalized in ("false", "0", "no"):
            return False

        raise ValueError(f"Invalid Boolean value for is_peak_hour: {value}")

    # =========================================================
    # CREATE RISK LABELS
    # =========================================================

    def _create_risk_labels(self, df):

        df = df.copy()

        def assign_risk(row):

            consumption_level = str(
                row["consumption_level"]
            ).strip().lower()

            anomaly_score = float(
                row["anomaly_score"]
            )

            is_peak = self._parse_bool(
                row["is_peak_hour"]
            )

            if (
                consumption_level == "critical"
                or anomaly_score >= 2.0
            ):
                return "HIGH"

            if (
                consumption_level == "high"
                or is_peak
                or anomaly_score >= 1.0
            ):
                return "MEDIUM"

            return "LOW"

        df["demand_risk"] = df.apply(
            assign_risk,
            axis=1
        )

        return df

    # =========================================================
    # TRAIN MODEL
    # =========================================================

    def _train_model(self):

        df = self._load_data()

        # Remove incomplete training rows before Boolean/numeric conversion.
        df = df.dropna(
            subset=self.feature_columns + ["consumption_level"]
        )

        # Normalize Boolean values before training the model.
        df["is_peak_hour"] = df["is_peak_hour"].apply(
            self._parse_bool
        )

        df = self._create_risk_labels(
            df
        )

        df = df.dropna(
            subset=(
                self.feature_columns
                + ["demand_risk"]
            )
        )

        X = df[
            self.feature_columns
        ]

        y = df[
            "demand_risk"
        ]

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

        self.model.fit(
            X_train,
            y_train
        )

        predictions = self.model.predict(
            X_test
        )

        self.accuracy = accuracy_score(
            y_test,
            predictions
        )

        self.is_trained = True

        print(
            "Demand risk model trained."
        )

        print(
            f"Demand risk accuracy: "
            f"{self.accuracy:.4f}"
        )

        print(
            "Risk classes: "
            f"{list(self.model.classes_)}"
        )

    # =========================================================
    # VALIDATE INPUT
    # =========================================================

    def _validate_case_data(
        self,
        case_data
    ):

        if not isinstance(
            case_data,
            dict
        ):

            raise ValueError(
                "case_data must be a dictionary."
            )

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in case_data
        ]

        if missing_columns:

            raise ValueError(
                "Missing prediction features: "
                f"{missing_columns}"
            )

        # Validate numeric features and reject missing/non-finite values.
        for column in self.feature_columns:
            if column == "is_peak_hour":
                self._parse_bool(case_data[column])
                continue

            try:
                numeric_value = float(case_data[column])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Prediction feature '{column}' must be numeric."
                ) from None

            if pd.isna(numeric_value) or numeric_value in (float("inf"), float("-inf")):
                raise ValueError(
                    f"Prediction feature '{column}' must be a finite number."
                )

    # =========================================================
    # PREDICT
    # =========================================================

    def predict(
        self,
        case_data
    ):

        if not self.is_trained:

            raise RuntimeError(
                "Demand risk model is not trained."
            )

        self._validate_case_data(
            case_data
        )

        # Work with a copy so the caller's dictionary is not modified.
        case_data = dict(case_data)
        case_data["is_peak_hour"] = self._parse_bool(
            case_data["is_peak_hour"]
        )

        row = pd.DataFrame(
            [case_data]
        )

        features = row[
            self.feature_columns
        ]

        prediction = self.model.predict(
            features
        )[0]

        probabilities = (
            self.model.predict_proba(
                features
            )[0]
        )

        probability_map = dict(
            zip(
                self.model.classes_,
                probabilities
            )
        )

        risk_probability = float(
            probability_map.get(
                prediction,
                0.0
            )
        )

        input_features = {

            "appliances": float(
                case_data["appliances"]
            ),

            "hour": int(
                case_data["hour"]
            ),

            "day_of_week_number": int(
                case_data[
                    "day_of_week_number"
                ]
            ),

            "month": int(
                case_data["month"]
            ),

            "is_peak_hour": self._parse_bool(
                case_data["is_peak_hour"]
            ),

            "rolling_mean_24": float(
                case_data[
                    "rolling_mean_24"
                ]
            ),

            "consumption_deviation": float(
                case_data[
                    "consumption_deviation"
                ]
            ),

            "anomaly_score": float(
                case_data["anomaly_score"]
            )
        }

        return {

            "risk_level": prediction,

            "risk_probability": round(
                risk_probability,
                4
            ),

            "model_accuracy": round(
                float(
                    self.accuracy
                ),
                4
            ),

            "model": "Random Forest",

            "input_features": input_features
        }