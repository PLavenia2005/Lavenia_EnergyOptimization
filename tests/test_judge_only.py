from src.services.llm_judge import LLMJudgeService
import json

judge = LLMJudgeService()

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

print("Judge model:", judge.model_name)
print("API key configured:", bool(judge.api_key))
print("Sending one Judge evaluation request...")

result = judge.judge(
    query=query,
    recommendation=recommendation,
    context=context
)

print("\n===== JUDGE RESULT =====")
print(json.dumps(result, indent=2))
