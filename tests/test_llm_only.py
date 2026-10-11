from src.agents.classification_agent import ClassificationAgent
from src.rag.hybrid_search import HybridSearch
from src.agents.optimization_agent import OptimizationAgent

queries = [
    ("1. Peak energy consumption",
     "How can we reduce unusually high energy consumption during peak hours?"),
    ("2. Energy optimization across facilities",
     "How can we identify and optimize energy consumption across facilities that are consuming more energy than expected?"),
    ("3. Persistent high consumption",
     "Energy consumption remains high despite previous optimization efforts. Find similar historical cases and recommend further improvements."),
    ("4. Normal consumption",
     "Energy consumption is within its normal range during off-peak hours. What general practices can help maintain energy efficiency?"),
    ("5. Moderate consumption",
     "Energy consumption is moderately elevated during off-peak hours. Find similar historical cases and recommend practical ways to improve efficiency."),
    ("6. High consumption",
     "Energy consumption is unusually high during peak hours. Find similar historical cases and recommend ways to reduce consumption."),
    ("7. Critical consumption",
     "Energy consumption has reached a critical level and is significantly above its normal pattern. Find similar historical cases and recommend urgent steps to investigate and reduce consumption."),
]

classifier = ClassificationAgent()
search = HybridSearch()
agent = OptimizationAgent()

for name, query in queries:
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    try:
        classification = classifier.classify(query)
        cases = search.search(query, top_k=5)

        # Direct LLM generation: does not call agent.recommend(),
        # which is the method that invokes the rule-based fallback.
        result = agent._generate_llm_recommendation(
            classification, cases
        )

        print("Category:", classification.get("primary_category"))
        print("Retrieved cases:", len(cases))
        print("LLM success:", result.get("success"))

        if result.get("success"):
            content = result.get("content", "")
            print("Validation:", agent._is_valid_llm_recommendation(
                content, classification
            ))
            print("\nRecommendation:\n", content)
        else:
            print("LLM error:", result.get("error"))
            print("No automatic fallback was requested by this test.")

    except Exception as exc:
        print("Test error:", str(exc))

print("\nAll seven direct LLM tests finished.")
