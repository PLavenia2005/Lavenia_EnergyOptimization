from src.services.deepeval_evaluation import (
    DeepEvalRecommendationService
)
import json

service = DeepEvalRecommendationService()

query = (
    "How can we reduce unusually high energy "
    "consumption during peak hours?"
)

recommendation = """
RECOMMENDED ACTION:
Shift non-essential energy use away from peak
hours and review equipment settings for efficiency.

WHY:
Retrieved historical cases show high energy
consumption during peak periods.

PRIORITY:
High
"""

context = [
    "Historical case: high energy consumption during peak hours.",
    "Historical case: critical consumption observed during a peak period."
]

print("DeepEval model:", service.model_name)
print("API key configured:", bool(service.api_key))
print("Running one DeepEval evaluation with two metrics...")

result = service.evaluate_recommendation(
    query=query,
    recommendation=recommendation,
    context=context
)

print("\n===== DEEPEVAL RESULT =====")
print(json.dumps(result, indent=2))
