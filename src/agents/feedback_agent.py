from datetime import datetime


class FeedbackLoopAgent:

    def __init__(self):
        self.feedback_records = []

    def submit_feedback(
        self,
        query,
        recommendation,
        feedback,
        rating=None,
        comment=None
    ):
        # Validate query
        if not query or not str(query).strip():
            return {
                "success": False,
                "error": "Query is required."
            }

        # Validate recommendation
        if not recommendation or not str(recommendation).strip():
            return {
                "success": False,
                "error": "Recommendation is required."
            }

        # Validate feedback
        if not feedback:
            return {
                "success": False,
                "error": "Feedback is required."
            }

        feedback_value = str(feedback).strip().lower()

        allowed_feedback = {
            "useful",
            "not useful",
            "helpful",
            "not helpful"
        }

        if feedback_value not in allowed_feedback:
            return {
                "success": False,
                "error": (
                    "Feedback must be one of: "
                    "useful, not useful, helpful, not helpful."
                )
            }

        # Normalize feedback
        if feedback_value in {"useful", "helpful"}:
            normalized_feedback = "positive"
        else:
            normalized_feedback = "negative"

        # Validate rating when provided
        if rating is not None:

            try:
                rating = float(rating)
            except (TypeError, ValueError):
                return {
                    "success": False,
                    "error": "Rating must be a valid number."
                }

            if rating < 1 or rating > 5:
                return {
                    "success": False,
                    "error": "Rating must be between 1 and 5."
                }

        # Create feedback record
        record = {
            "feedback_id": len(self.feedback_records) + 1,
            "timestamp": datetime.now().isoformat(),
            "query": str(query).strip(),
            "recommendation": str(recommendation).strip(),
            "feedback": normalized_feedback,
            "rating": rating,
            "comment": comment
        }

        self.feedback_records.append(record)

        return {
            "success": True,
            "feedback_id": record["feedback_id"],
            "feedback": normalized_feedback,
            "rating": rating,
            "comment": comment,
            "message": "Feedback recorded successfully."
        }

    def get_feedback_summary(self):

        total_feedback = len(self.feedback_records)

        if total_feedback == 0:
            return {
                "success": True,
                "total_feedback": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "positive_rate": 0.0,
                "average_rating": None,
                "feedback_status": "NO_FEEDBACK"
            }

        positive_feedback = sum(
            1
            for record in self.feedback_records
            if record["feedback"] == "positive"
        )

        negative_feedback = sum(
            1
            for record in self.feedback_records
            if record["feedback"] == "negative"
        )

        ratings = [
            record["rating"]
            for record in self.feedback_records
            if record["rating"] is not None
        ]

        average_rating = None

        if ratings:
            average_rating = round(
                sum(ratings) / len(ratings),
                2
            )

        positive_rate = round(
            (positive_feedback / total_feedback) * 100,
            2
        )

        if positive_rate >= 80:
            feedback_status = "STRONG"
        elif positive_rate >= 60:
            feedback_status = "MODERATE"
        else:
            feedback_status = "NEEDS_IMPROVEMENT"

        return {
            "success": True,
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "positive_rate": positive_rate,
            "average_rating": average_rating,
            "feedback_status": feedback_status
        }

    def get_records(self):

        return {
            "success": True,
            "total_feedback": len(self.feedback_records),
            "records": self.feedback_records
        }


feedback_loop_agent = FeedbackLoopAgent()