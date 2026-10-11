import requests

import streamlit as st

# =========================================================

# CONFIGURATION

# =========================================================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(

    page_title="Energy Consumption Optimization Assistant",

    page_icon="⚡",

    layout="wide"

)

# =========================================================

# HEADER

# =========================================================

st.title(

    "⚡ AI-Powered Energy Consumption Optimization Assistant"

)

st.caption(

    "Unified AI-powered energy analysis using Hybrid RAG, "

    "Random Forest demand-risk prediction, OpenRouter LLM, "

    "LLM-as-a-Judge and DeepEval."

)

# =========================================================

# SIDEBAR

# =========================================================

st.sidebar.title("⚙️ Analysis Configuration")

top_k = st.sidebar.slider(

    "Retrieved Cases",

    min_value=1,

    max_value=10,

    value=5

)

season = st.sidebar.selectbox(

    "Season",

    [

        "All",

        "Winter",

        "Spring",

        "Summer",

        "Autumn"

    ]

)

period = st.sidebar.selectbox(

    "Period",

    [

        "All",

        "Peak",

        "Off-Peak"

    ]

)

consumption_level = st.sidebar.selectbox(

    "Consumption Level",

    [

        "All",

        "Normal",

        "Moderate",

        "High",

        "Critical"

    ]

)

anomaly = st.sidebar.selectbox(

    "Anomaly",

    [

        "All",

        "True",

        "False"

    ]

)

# =========================================================

# SCENARIO ESTIMATION

# =========================================================

st.sidebar.divider()

st.sidebar.subheader(

    "💰 Impact Scenario Inputs"

)

st.sidebar.caption(

    "Optional inputs are used only for scenario-based "

    "cost and carbon estimation. Values are never guessed."

)

baseline_kwh = st.sidebar.number_input(

    "Baseline Energy (kWh)",

    min_value=0.0,

    value=0.0,

    step=100.0

)

tariff_per_kwh = st.sidebar.number_input(

    "Tariff (per kWh)",

    min_value=0.0,

    value=0.0,

    step=0.5

)

emission_factor = st.sidebar.number_input(

    "Emission Factor (kg CO₂e/kWh)",

    min_value=0.0,

    value=0.0,

    step=0.1

)

reduction_rate = st.sidebar.number_input(

    "Scenario Reduction Rate (%)",

    min_value=0.0,

    max_value=100.0,

    value=0.0,

    step=1.0

)

period_label = st.sidebar.text_input(

    "Scenario Period",

    value="specified period"

)

# =========================================================

# API HEALTH

# =========================================================

try:

    health = requests.get(

        f"{API_URL}/api/health",

        timeout=5

    )

    if health.status_code == 200:

        st.sidebar.success(

            "🟢 API Connected"

        )

    else:

        st.sidebar.error(

            "🔴 API Error"

        )

except requests.exceptions.RequestException:

    st.sidebar.error(

        "🔴 API Not Connected"

    )

# =========================================================

# HELPER FUNCTIONS

# =========================================================

def build_payload():

    payload = {

        "query": query,

        "top_k": top_k

    }

    if season != "All":

        payload["season"] = season

    if period != "All":

        payload["period"] = period

    if consumption_level != "All":

        payload["consumption_level"] = (

            consumption_level

        )

    if anomaly != "All":

        payload["is_anomaly"] = (

            anomaly == "True"

        )

    # Optional scenario inputs

    if baseline_kwh > 0:

        payload["baseline_kwh"] = (

            baseline_kwh

        )

    if tariff_per_kwh > 0:

        payload["tariff_per_kwh"] = (

            tariff_per_kwh

        )

    if emission_factor > 0:

        payload["emission_factor_kg_per_kwh"] = (

            emission_factor

        )

    if reduction_rate > 0:

        payload["reduction_rate"] = (

            reduction_rate

        )

    payload["period_label"] = (

        period_label

    )

    return payload

def post_request(

    endpoint,

    payload,

    timeout=600

):

    try:

        response = requests.post(

            f"{API_URL}{endpoint}",

            json=payload,

            timeout=timeout

        )

        try:

            data = response.json()

        except ValueError:

            return {

                "success": False,

                "error": (

                    f"API returned invalid JSON "

                    f"(HTTP {response.status_code})."

                )

            }

        if response.status_code != 200:

            return {

                "success": False,

                "error": data.get(

                    "detail",

                    data.get(

                        "error",

                        f"HTTP {response.status_code}"

                    )

                )

            }

        return data

    except requests.exceptions.RequestException as error:

        return {

            "success": False,

            "error": str(error)

        }

def run_analysis(endpoint, spinner_text, mode):
    if not query.strip():
        st.warning("Please enter an energy-related question.")
        return False

    payload = build_payload()

    with st.spinner(spinner_text):
        result = post_request(
            endpoint,
            payload,
            timeout=600
        )

    if not result.get("success", False):
        st.error(
            result.get(
                "error",
                "Energy analysis failed."
            )
        )
        return False

    st.session_state["analysis"] = result
    st.session_state["query"] = query
    st.session_state["analysis_mode"] = mode
    st.session_state.pop("evaluation", None)

    recommendation = result.get("recommendation", {})
    if (
        mode == "llm"
        and isinstance(recommendation, dict)
        and recommendation.get("fallback_used", False)
    ):
        st.warning(
            "⚠️ OpenRouter LLM was unavailable or returned an invalid "
            "response, so the backend used the Rule-Based Fallback."
        )
    else:
        st.success(
            "✅ Unified energy analysis completed using "
            + ("LLM + RAG." if mode == "llm" else "Rule-Based Fallback.")
        )

    return True



def format_score(value):

    if value is None:
        return "N/A"

    # Handle Judge metrics returned as dictionaries
    if isinstance(value, dict):
        value = value.get("score")

        if value is None:
            return "N/A"

    try:
        score = float(value)

        # Scores from 0 to 1 are displayed as percentages
        if 0 <= score <= 1:
            return f"{score:.2%}"

        return f"{score:.2f}"

    except (ValueError, TypeError):
        return str(value)


def display_recommendation(

    recommendation,

    title="Optimization Recommendation"

):

    if not recommendation:

        st.info(

            "Recommendation is unavailable."

        )

        return

    if not isinstance(

        recommendation,

        dict

    ):

        st.info(

            str(recommendation)

        )

        return

    st.subheader(title)

    recommendation_text = (

        recommendation.get(

            "recommendation",

            ""

        )

    )

    if recommendation_text:

        st.info(

            recommendation_text

        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Priority",

            recommendation.get(

                "priority",

                "N/A"

            )

        )

    with col2:

        st.metric(

            "Category",

            recommendation.get(

                "category",

                "N/A"

            )

        )

    with col3:

        st.metric(

            "Method",

            recommendation.get(

                "method",

                "N/A"

            )

        )

    if recommendation.get(

        "fallback_used",

        False

    ):

        st.warning(

            "Rule-Based recommendation selected. The recommendation was generated using the rule-based engine."

        )

    supporting_case = (

        recommendation.get(

            "supporting_case"

        )

    )

    if supporting_case:

        st.write(

            "**Supporting Historical Case:** "

            f"{supporting_case.get('case_id', 'N/A')}"

        )

        with st.expander(

            "View Supporting Case Details"

        ):

            st.json(

                supporting_case

            )

# =========================================================

# QUERY

# =========================================================

st.subheader(

    "🔎 Energy Optimization Query"

)

query = st.text_area(

    "Ask an energy-related question",

    placeholder=(

        "Example: How can we reduce high "

        "energy consumption during peak hours?"

    ),

    height=100

)

# =========================================================

# MAIN ANALYSIS BUTTONS

# =========================================================

st.subheader("Choose Analysis Method")

col1, col2 = st.columns(2)

with col1:
    llm_analysis_button = st.button(
        "🤖 Analyze with LLM + RAG",
        type="primary",
        use_container_width=True
    )

with col2:
    rule_analysis_button = st.button(
        "🛡️ Analyze with Rule-Based Fallback",
        use_container_width=True
    )

# =========================================================

# UNIFIED ANALYSIS

# =========================================================

if llm_analysis_button:
    run_analysis(
        "/api/recommend/llm",
        "Running LLM + RAG analysis. This includes hybrid retrieval, "
        "risk prediction, recommendation, LLM Judge and DeepEval...",
        "llm"
    )

elif rule_analysis_button:
    run_analysis(
        "/api/recommend/rule",
        "Running Rule-Based Fallback analysis. This includes hybrid "
        "retrieval, risk prediction, recommendation, LLM Judge and DeepEval...",
        "rule"
    )

# =========================================================

# LOAD ANALYSIS

# =========================================================

analysis = st.session_state.get(

    "analysis"

)

# =========================================================

# RESULTS

# =========================================================

if analysis:

    tabs = st.tabs(

        [

            "📊 Analysis",

            "🔎 Retrieval",

            "⚠️ Demand Risk",

            "💡 Recommendation",

            "🎯 Effectiveness",

            "⚖️ LLM Judge",

            "🧪 DeepEval",

            "💰 Cost",

            "🌱 Carbon",

            "🤝 Collaboration",

            "🔄 Feedback",

            "📈 Evaluation"

        ]

    )

    # =====================================================

    # TAB 1 - ANALYSIS

    # =====================================================

    with tabs[0]:

        st.header(

            "📊 Unified Energy Analysis"

        )

        analysis_mode = analysis.get(
            "recommendation_selection",
            {}
        ).get(
            "selection_mode",
            st.session_state.get(
                "analysis_mode",
                "automatic"
            )
        )

        if analysis_mode == "explicit_llm":
            st.info("🤖 Analysis Mode: LLM + RAG")
        elif analysis_mode == "explicit_rule":
            st.info("🛡️ Analysis Mode: Rule-Based Fallback")
        else:
            st.info("⚙️ Analysis Mode: Automatic")


        classification = (

            analysis.get(

                "classification",

                {}

            )

        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(

                "Primary Category",

                classification.get(

                    "primary_category",

                    "N/A"

                )

            )

        with col2:

            categories = (

                classification.get(

                    "categories",

                    []

                )

            )

            if categories:

                st.metric(

                    "Detected Categories",

                    len(categories)

                )

            else:

                st.metric(

                    "Detected Categories",

                    1

                )

        with col3:

            st.metric(

                "Retrieved Cases",

                len(

                    analysis.get(

                        "retrieval",

                        {}

                    ).get(

                        "results",

                        []

                    )

                )

            )

        st.divider()

        st.write(

            "**User Query:**"

        )

        st.info(

            analysis.get(

                "query",

                query

            )

        )

        if categories:

            st.write(

                "**Detected Categories:** "

                + ", ".join(categories)

            )

        risk = analysis.get(

            "risk_prediction",

            {}

        )

        effectiveness = analysis.get(

            "effectiveness",

            {}

        )

        judge = analysis.get(

            "llm_judge",

            {}

        )

        deepeval = analysis.get(

            "deepeval",

            {}

        )

        st.divider()

        st.subheader(

            "📌 Analysis Summary"

        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(

                "Demand Risk",

                risk.get(

                    "risk_level",

                    "N/A"

                )

            )

        with col2:

            score = effectiveness.get(

                "effectiveness_score"

            )

            st.metric(

                "Effectiveness",

                f"{float(score):.2f}/100"

                if score is not None

                else "N/A"

            )

        with col3:

            st.metric(

                "LLM Judge",

                format_score(

                    judge.get(

                        "overall_score"

                    )

                )

                if judge.get("success")

                else "Unavailable"

            )

        with col4:

            st.metric(

                "DeepEval",

                format_score(

                    deepeval.get(

                        "overall_score"

                    )

                )

                if deepeval.get("success")

                else "Unavailable"

            )

        st.divider()

        st.subheader(

            "🧩 Unified Workflow"

        )

        st.write(

            "User Query → Guardrails → Classification → "

            "Hybrid Retrieval → Demand Risk → "

            "Optimization Recommendation → Effectiveness → "

            "LLM Judge → DeepEval → Cost/Carbon Scenario → "

            "Feedback"

        )

    # =====================================================

    # TAB 2 - RETRIEVAL

    # =====================================================

    with tabs[1]:

        st.header(

            "🔎 Hybrid Retrieval"

        )

        retrieval = (

            analysis.get(

                "retrieval",

                {}

            )

        )

        cases = (

            retrieval.get(

                "results",

                []

            )

        )

        st.metric(

            "Retrieved Cases",

            len(cases)

        )

        if not cases:

            st.info(

                "No historical cases were retrieved."

            )

        for index, case in enumerate(

            cases

        ):

            with st.expander(

                f"{index + 1}. "

                f"{case.get('case_id', 'Unknown Case')}"

            ):

                st.write(

                    case.get(

                        "document",

                        "No document available."

                    )

                )

                col1, col2, col3, col4 = (

                    st.columns(4)

                )

                with col1:

                    st.metric(

                        "Semantic",

                        f"{case.get('semantic_score', 0):.4f}"

                    )

                with col2:

                    st.metric(

                        "Keyword",

                        f"{case.get('keyword_score', 0):.4f}"

                    )

                with col3:

                    st.metric(

                        "Hybrid",

                        f"{case.get('hybrid_score', 0):.4f}"

                    )

                with col4:

                    st.metric(

                        "Priority",

                        f"{case.get('priority_score', 0):.4f}"

                    )

                st.write(

                    "**Metadata**"

                )

                st.json(

                    case.get(

                        "metadata",

                        {}

                    )

                )

    # =====================================================

    # TAB 3 - DEMAND RISK

    # =====================================================

    with tabs[2]:

        st.header(

            "⚠️ Demand Risk Prediction"

        )

        risk = (

            analysis.get(

                "risk_prediction"

            )

        )

        if not risk:

            st.info(

                "Risk prediction is unavailable."

            )

        else:

            col1, col2, col3, col4 = (

                st.columns(4)

            )

            with col1:

                st.metric(

                    "Risk Level",

                    risk.get(

                        "risk_level",

                        "N/A"

                    )

                )

            with col2:

                st.metric(

                    "Probability",

                    format_score(

                        risk.get(

                            "risk_probability"

                        )

                    )

                )

            with col3:

                st.metric(

                    "Model",

                    risk.get(

                        "model",

                        "N/A"

                    )

                )

            with col4:

                st.metric(

                    "Model Accuracy",

                    format_score(

                        risk.get(

                            "model_accuracy"

                        )

                    )

                )

            st.divider()

            risk_level = risk.get(

                "risk_level"

            )

            if risk_level == "NOT_ASSESSED":

                st.info(
                    "ℹ️ Demand risk was not assessed for this query. "
                    "The Random Forest model runs only when the query "
                    "describes high consumption, peak demand, anomalies, "
                    "or abnormal energy usage."
                )

                reason = risk.get("reason")
                if reason:
                    st.caption(reason)

            elif risk_level == "HIGH":

                st.error(

                    "🔴 High demand risk detected. "

                    "Immediate review of consumption "

                    "patterns is recommended."

                )

            elif risk_level == "MEDIUM":

                st.warning(

                    "🟡 Medium demand risk detected. "

                    "Review peak usage and unusual "

                    "consumption patterns."

                )

            elif risk_level == "LOW":

                st.success(

                    "🟢 Low demand risk detected."

                )

            else:

                st.info(
                    f"ℹ️ Risk status: {risk_level or 'N/A'}"
                )

            features = risk.get(

                "input_features"

            )

            if features:

                with st.expander(

                    "View Risk Model Input Features"

                ):

                    st.json(

                        features

                    )

    # =====================================================

    # TAB 4 - RECOMMENDATION

    # =====================================================

    with tabs[3]:

        st.header(

            "💡 Optimization Recommendation"

        )

        recommendation = (

            analysis.get(

                "recommendation"

            )

        )

        display_recommendation(

            recommendation

        )

        if isinstance(

            recommendation,

            dict

        ):

            st.divider()

            st.subheader(

                "Recommendation Details"

            )

            col1, col2 = st.columns(2)

            with col1:

                st.write(

                    "**Recommendation Method:** "

                    f"{recommendation.get('method', 'N/A')}"

                )

                st.write(

                    "**Requested Method:** "

                    f"{recommendation.get('requested_method', 'N/A')}"

                )

                st.write(

                    "**Priority:** "

                    f"{recommendation.get('priority', 'N/A')}"

                )

            with col2:

                st.write(

                    "**Category:** "

                    f"{recommendation.get('category', 'N/A')}"

                )

                st.write(

                    "**Selection Mode:** "

                    f"{recommendation.get('selection_mode', 'N/A')}"

                )

                st.write(

                    "**Fallback Used:** "

                    f"{recommendation.get('fallback_used', False)}"

                )

    # =====================================================

    # TAB 5 - EFFECTIVENESS

    # =====================================================

    with tabs[4]:

        st.header(

            "🎯 Energy-Saving Effectiveness"

        )

        effectiveness = (

            analysis.get(

                "effectiveness",

                {}

            )

        )

        if not effectiveness:

            st.info(

                "Effectiveness estimation is unavailable."

            )

        else:

            score = effectiveness.get(

                "effectiveness_score"

            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(

                    "Effectiveness Score",

                    f"{float(score):.2f}/100"

                    if score is not None

                    else "N/A"

                )

            with col2:

                st.metric(

                    "Effectiveness Level",

                    effectiveness.get(

                        "effectiveness_level",

                        "N/A"

                    )

                )

            with col3:

                st.metric(

                    "Recommendation Category",

                    effectiveness.get(

                        "recommendation_category",

                        "N/A"

                    )

                )

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:

                value = effectiveness.get(

                    "retrieval_quality"

                )

                st.metric(

                    "Retrieval Quality",

                    f"{float(value):.2f}"

                    if value is not None

                    else "N/A"

                )

            with col2:

                value = effectiveness.get(

                    "context_factor"

                )

                st.metric(

                    "Context Factor",

                    f"{float(value):.2f}"

                    if value is not None

                    else "N/A"

                )

            with col3:

                value = effectiveness.get(

                    "risk_factor"

                )

                st.metric(

                    "Risk Factor",

                    f"{float(value):.2f}"

                    if value is not None

                    else "N/A"

                )

            st.info(

                effectiveness.get(

                    "interpretation",

                    "Effectiveness interpretation unavailable."

                )

            )

    # =====================================================

    # TAB 6 - LLM JUDGE

    # =====================================================

    with tabs[5]:

        st.header(

            "⚖️ LLM-as-a-Judge"

        )

        judge = (

            analysis.get(

                "llm_judge",

                {}

            )

        )

        if not judge:

            st.info(

                "LLM Judge result is unavailable."

            )

        elif not judge.get(

            "success",

            False

        ):

            st.warning(

                judge.get(

                    "error",

                    "LLM Judge evaluation failed."

                )

            )

        else:

            evaluation = judge.get(

                "evaluation",

                {}

            )

            if isinstance(

                evaluation,

                dict

            ):

                col1, col2, col3 = (

                    st.columns(3)

                )

                with col1:

                    st.metric(

                        "Overall Quality",

                        format_score(

                            evaluation.get(

                                "overall_quality"

                            )

                        )

                    )

                with col2:

                    st.metric(

                        "Verdict",

                        evaluation.get(

                            "final_verdict",

                            judge.get(

                                "verdict",

                                "N/A"

                            )

                        )

                    )

                with col3:

                    st.metric(

                        "Overall Score",

                        format_score(

                            judge.get(

                                "overall_score"

                            )

                        )

                    )

                st.divider()

                st.subheader(

                    "Judge Evaluation"

                )

                score_items = [

                    (

                        "Relevance",

                        evaluation.get(

                            "relevance"

                        )

                    ),

                    (

                        "Actionability",

                        evaluation.get(

                            "actionability"

                        )

                    ),

                    (

                        "Evidence Support",

                        evaluation.get(

                            "evidence_support"

                        )

                    ),

                    (

                        "Energy-Saving Suitability",

                        evaluation.get(

                            "energy_saving_suitability"

                        )

                    ),

                    (

                        "Overall Quality",

                        evaluation.get(

                            "overall_quality"

                        )

                    )

                ]

                for label, value in score_items:

                    if isinstance(

                        value,

                        dict

                    ):

                        st.write(

                            f"**{label}:** "

                            f"{format_score(value.get('score'))}"

                        )

                        reason = value.get(

                            "reason"

                        )

                        if reason:

                            st.caption(

                                reason

                            )

                with st.expander(

                    "View Complete LLM Judge Response"

                ):

                    st.json(

                        judge

                    )

    # =====================================================

    # TAB 7 - DEEPEVAL

    # =====================================================

    with tabs[6]:

        st.header(

            "🧪 DeepEval Recommendation Evaluation"

        )

        deepeval = (

            analysis.get(

                "deepeval",

                {}

            )

        )

        if not deepeval:

            st.info(

                "DeepEval result is unavailable."

            )

        elif not deepeval.get(

            "success",

            False

        ):

            st.warning(

                deepeval.get(

                    "error",

                    "DeepEval evaluation failed."

                )

            )

        else:

            col1, col2 = st.columns(2)

            with col1:

                st.metric(

                    "Overall Score",

                    format_score(

                        deepeval.get(

                            "overall_score"

                        )

                    )

                )

            with col2:

                st.metric(

                    "Status",

                    "PASSED"

                    if deepeval.get(

                        "success",

                        False

                    )

                    else "FAILED"

                )

            st.divider()

            metrics = deepeval.get(

                "metrics",

                {}

            )

            if metrics:

                st.subheader(

                    "DeepEval Metrics"

                )

                for metric_name, metric_data in metrics.items():

                    if isinstance(

                        metric_data,

                        dict

                    ):

                        score = metric_data.get(

                            "score"

                        )

                        st.write(

                            f"**{metric_name}**"

                        )

                        if score is not None:

                            st.metric(

                                "Score",

                                format_score(score)

                            )

                        reason = metric_data.get(

                            "reason"

                        )

                        if reason:

                            st.caption(

                                reason

                            )

                        st.divider()

                with st.expander(

                    "View Complete DeepEval Response"

                ):

                    st.json(

                        deepeval

                    )

    # =====================================================

    # TAB 8 - COST

    # =====================================================

    with tabs[7]:

        st.header(

            "💰 Cost Reduction Scenario"

        )

        cost = (

            analysis.get(

                "cost_reduction",

                {}

            )

        )

        if not cost:

            st.info(

                "Cost reduction data is unavailable."

            )

        elif not cost.get(

            "available",

            cost.get(

                "success",

                False

            )

        ):

            st.info(

                cost.get(

                    "message",

                    "Cost reduction scenario is unavailable."

                )

            )

            st.caption(

                "Provide baseline energy, tariff and "

                "reduction-rate inputs in the sidebar "

                "to calculate a scenario estimate."

            )

        elif not cost.get(

            "success",

            False

        ):

            st.warning(

                cost.get(

                    "error",

                    "Cost reduction estimation failed."

                )

            )

        else:

            col1, col2, col3 = (

                st.columns(3)

            )

            with col1:

                st.metric(

                    "Baseline Energy",

                    f"{cost.get('baseline_energy_kwh', cost.get('baseline_kwh', 0)):,.2f} kWh"

                )

            with col2:

                st.metric(

                    "Energy Reduction",

                    f"{cost.get('estimated_energy_reduction_kwh', 0):,.2f} kWh"

                )

            with col3:

                st.metric(

                    "Estimated Cost Reduction",

                    f"{cost.get('estimated_cost_reduction', 0):,.2f}"

                )

            st.divider()

            st.write(

                "**Estimation Type:** "

                f"{cost.get('estimation_method', 'Scenario estimate')}"

            )

            st.info(

                cost.get(

                    "interpretation",

                    "This is a scenario estimate, not a measured historical saving."

                )

            )

            with st.expander(

                "View Cost Calculation"

            ):

                st.json(

                    cost

                )

    # =====================================================

    # TAB 9 - CARBON

    # =====================================================

    with tabs[8]:

        st.header(

            "🌱 Carbon Reduction Scenario"

        )

        carbon = (

            analysis.get(

                "carbon_reduction",

                {}

            )

        )

        if not carbon:

            st.info(

                "Carbon reduction data is unavailable."

            )

        elif not carbon.get(

            "available",

            carbon.get(

                "success",

                False

            )

        ):

            st.info(

                carbon.get(

                    "message",

                    "Carbon reduction scenario is unavailable."

                )

            )

            st.caption(

                "Provide baseline energy, emission factor "

                "and reduction-rate inputs in the sidebar "

                "to calculate a scenario estimate."

            )

        elif not carbon.get(

            "success",

            False

        ):

            st.warning(

                carbon.get(

                    "error",

                    "Carbon reduction estimation failed."

                )

            )

        else:

            col1, col2, col3 = (

                st.columns(3)

            )

            with col1:

                st.metric(

                    "Baseline Energy",

                    f"{carbon.get('baseline_energy_kwh', carbon.get('baseline_kwh', 0)):,.2f} kWh"

                )

            with col2:

                st.metric(

                    "Energy Reduction",

                    f"{carbon.get('estimated_energy_reduction_kwh', 0):,.2f} kWh"

                )

            with col3:

                st.metric(

                    "Carbon Reduction",

                    f"{carbon.get('estimated_carbon_reduction_kg_co2e', 0):,.2f} kg CO₂e"

                )

            st.divider()

            st.write(

                "**Estimation Type:** "

                f"{carbon.get('estimation_method', 'Scenario estimate')}"

            )

            st.info(

                carbon.get(

                    "interpretation",

                    "This is a scenario estimate, not a measured historical reduction."

                )

            )

            with st.expander(

                "View Carbon Calculation"

            ):

                st.json(

                    carbon

                )

    # =====================================================

    # TAB 10 - COLLABORATION

    # =====================================================

    with tabs[9]:

        st.header(

            "🤝 Multi-Agent Collaboration"

        )

        collaboration = (

            analysis.get(

                "collaboration",

                {}

            )

        )

        if collaboration:

            summary = (

                collaboration.get(

                    "summary"

                )

                if isinstance(

                    collaboration,

                    dict

                )

                else None

            )

            if summary:

                st.info(

                    summary

                )

            with st.expander(

                "View Collaboration Details"

            ):

                st.json(

                    collaboration

                )

        else:

            st.info(

                "Collaboration summary is unavailable."

            )

        contributions = (

            analysis.get(

                "agent_contributions",

                {}

            )

        )

        st.divider()

        st.subheader(

            "🧩 Agent Contributions"

        )

        if contributions:

            for agent_name, contribution in (

                contributions.items()

            ):

                with st.expander(

                    agent_name.replace(

                        "_",

                        " "

                    ).title()

                ):

                    st.json(

                        contribution

                    )

        else:

            st.info(

                "No agent contribution details available."

            )

    # =====================================================

    # TAB 11 - FEEDBACK

    # =====================================================

    with tabs[10]:

        st.header(

            "🔄 Recommendation Feedback Loop"

        )

        recommendation_data = (

            analysis.get(

                "recommendation",

                {}

            )

        )

        default_recommendation = ""

        if isinstance(

            recommendation_data,

            dict

        ):

            default_recommendation = (

                recommendation_data.get(

                    "recommendation",

                    ""

                )

            )

        feedback_recommendation = st.text_area(

            "Recommendation",

            value=default_recommendation,

            height=120

        )

        feedback_type = st.selectbox(

            "Feedback",

            [

                "useful",

                "not useful",

                "positive",

                "negative"

            ]

        )

        rating = st.slider(

            "Rating",

            min_value=1,

            max_value=5,

            value=5

        )

        comment = st.text_area(

            "Comment",

            placeholder=(

                "Optional feedback about the recommendation."

            )

        )

        submit_feedback_button = st.button(

            "📤 Submit Feedback",

            type="primary",

            use_container_width=True

        )

        if submit_feedback_button:

            if not feedback_recommendation.strip():

                st.warning(

                    "Please provide a recommendation."

                )

            else:

                feedback_payload = {

                    "query": analysis.get(

                        "query",

                        query

                    ),

                    "recommendation": (

                        feedback_recommendation

                    ),

                    "feedback": feedback_type,

                    "rating": rating,

                    "comment": comment

                }

                with st.spinner(

                    "Submitting feedback..."

                ):

                    feedback_result = post_request(

                        "/api/feedback",

                        feedback_payload,

                        timeout=60

                    )

                if feedback_result.get(

                    "success",

                    False

                ):

                    st.success(

                        "Feedback submitted successfully."

                    )

                    st.json(

                        feedback_result

                    )

                else:

                    st.error(

                        feedback_result.get(

                            "error",

                            "Feedback submission failed."

                        )

                    )

        st.divider()

        summary_button = st.button(

            "📊 Refresh Feedback Summary",

            use_container_width=True

        )

        if summary_button:

            try:

                response = requests.get(

                    f"{API_URL}/api/feedback/summary",

                    timeout=30

                )

                summary_result = (

                    response.json()

                )

            except requests.exceptions.RequestException as error:

                summary_result = {

                    "success": False,

                    "error": str(error)

                }

            if summary_result.get(

                "success",

                False

            ):

                col1, col2, col3, col4 = (

                    st.columns(4)

                )

                with col1:

                    st.metric(

                        "Total Feedback",

                        summary_result.get(

                            "total_feedback",

                            0

                        )

                    )

                with col2:

                    st.metric(

                        "Positive",

                        summary_result.get(

                            "positive_feedback",

                            0

                        )

                    )

                with col3:

                    st.metric(

                        "Negative",

                        summary_result.get(

                            "negative_feedback",

                            0

                        )

                    )

                with col4:

                    rate = summary_result.get(

                        "positive_rate",

                        0

                    )

                    st.metric(

                        "Positive Rate",

                        f"{float(rate):.2f}%"

                        if float(rate) > 1

                        else f"{float(rate):.2%}"

                    )

                st.write(

                    "**Feedback Status:** "

                    f"{summary_result.get('feedback_status', 'N/A')}"

                )

            else:

                st.warning(

                    summary_result.get(

                        "error",

                        "Feedback summary unavailable."

                    )

                )

        records_button = st.button(

            "📋 View Feedback Records",

            use_container_width=True

        )

        if records_button:

            try:

                response = requests.get(

                    f"{API_URL}/api/feedback/records",

                    timeout=30

                )

                records_result = (

                    response.json()

                )

            except requests.exceptions.RequestException as error:

                records_result = {

                    "success": False,

                    "error": str(error)

                }

            if records_result.get(

                "success",

                False

            ):

                records = records_result.get(

                    "records",

                    []

                )

                if records:

                    st.dataframe(

                        records,

                        use_container_width=True

                    )

                else:

                    st.info(

                        "No feedback records available."

                    )

            else:

                st.warning(

                    records_result.get(

                        "error",

                        "Feedback records unavailable."

                    )

                )

    # =====================================================

    # TAB 12 - EVALUATION

    # =====================================================

    with tabs[11]:

        st.header(

            "📈 Query-Specific Retrieval Evaluation"

        )

        st.write(

            "Evaluate retrieval performance for the "

            "current energy query using the historical dataset."

        )

        current_query = st.session_state.get(

            "query",

            analysis.get(

                "query",

                ""

            )

        )

        if current_query:

            st.info(

                f"**Current Query:** {current_query}"

            )

        else:

            st.warning(

                "Please analyze an energy query first."

            )

        evaluate_button = st.button(

            "📊 Evaluate Current Query",

            type="primary",

            use_container_width=True

        )

        if evaluate_button:

            if not current_query.strip():

                st.warning(

                    "Please enter and analyze "

                    "an energy-related query first."

                )

            else:

                evaluation_payload = {

                    "query": current_query,

                    "top_k": top_k

                }

                with st.spinner(

                    "Evaluating retrieval quality..."

                ):

                    evaluation_result = post_request(

                        "/api/evaluate",

                        evaluation_payload,

                        timeout=240

                    )

                st.session_state[

                    "evaluation"

                ] = evaluation_result

        evaluation = (

            st.session_state.get(

                "evaluation"

            )

        )

        if evaluation:

            if not evaluation.get(

                "success",

                False

            ):

                st.warning(

                    evaluation.get(

                        "error",

                        "Evaluation unavailable."

                    )

                )

            else:

                st.success(

                    "Query evaluation completed."

                )

                st.subheader(

                    "📋 Evaluation Summary"

                )

                col1, col2, col3 = (

                    st.columns(3)

                )

                with col1:

                    st.metric(

                        "Dataset Size",

                        f"{evaluation.get('dataset_size', 0):,}"

                    )

                with col2:

                    relevant_cases = evaluation.get("relevant_cases")

                    st.metric(

                        "Relevant Cases",

                        f"{relevant_cases:,}"
                        if relevant_cases is not None
                        else "N/A"

                    )

                with col3:

                    st.metric(

                        "Top-K",

                        evaluation.get(

                            "top_k",

                            top_k

                        )

                    )

                conditions = evaluation.get(

                    "detected_conditions",

                    {}

                )

                if conditions:

                    st.subheader(

                        "🎯 Detected Query Conditions"

                    )

                    condition_items = []

                    for key, value in conditions.items():

                        label = (

                            key

                            .replace(

                                "_",

                                " "

                            )

                            .title()

                        )

                        condition_items.append(

                            f"**{label}:** {value}"

                        )

                    st.write(

                        " | ".join(

                            condition_items

                        )

                    )

                metrics = evaluation.get(

                    "metrics",

                    {}

                )

                st.subheader(

                    "📊 Retrieval Performance"

                )

                col1, col2, col3, col4 = (

                    st.columns(4)

                )

                with col1:

                    accuracy = metrics.get("accuracy")

                    st.metric(

                        "Accuracy",

                        f"{accuracy * 100:.2f}%"
                        if accuracy is not None
                        else "N/A"

                    )

                with col2:

                    precision = metrics.get("precision")

                    st.metric(

                        "Precision",

                        f"{precision * 100:.2f}%"
                        if precision is not None
                        else "N/A"

                    )

                with col3:

                    recall = metrics.get("recall")

                    st.metric(

                        "Recall",

                        f"{recall * 100:.2f}%"
                        if recall is not None
                        else "N/A"

                    )

                with col4:

                    f1_score = metrics.get("f1_score")

                    st.metric(

                        "F1 Score",

                        f"{f1_score * 100:.2f}%"
                        if f1_score is not None
                        else "N/A"

                    )

                diagnostics = evaluation.get(

                    "diagnostics",

                    {}

                )

                st.subheader(

                    "🔍 Evaluation Details"

                )

                col1, col2, col3 = (

                    st.columns(3)

                )

                with col1:

                    true_positive = diagnostics.get("true_positive")

                    st.metric(

                        "Relevant Retrieved",

                        true_positive
                        if true_positive is not None
                        else "N/A"

                    )

                with col2:

                    false_positive = diagnostics.get("false_positive")

                    st.metric(

                        "Non-Relevant Retrieved",

                        false_positive
                        if false_positive is not None
                        else "N/A"

                    )

                with col3:

                    false_negative = diagnostics.get("false_negative")

                    st.metric(

                        "Relevant Cases Missed",

                        false_negative
                        if false_negative is not None
                        else "N/A"

                    )

                retrieved_ids = evaluation.get(

                    "retrieved_case_ids",

                    []

                )

                st.subheader(

                    "📚 Retrieved Case IDs"

                )

                if retrieved_ids:

                    st.write(

                        ", ".join(

                            retrieved_ids

                        )

                    )

                else:

                    st.info(

                        "No cases were retrieved."

                    )

        else:

            st.info(

                "Click **Evaluate Current Query** "

                "to calculate Accuracy, Precision, "

                "Recall and F1 Score."

            )

# =========================================================

# FOOTER

# =========================================================

st.divider()

st.caption(

    "AI-Powered Energy Consumption Optimization Assistant "

    "| Hybrid RAG | Random Forest | OpenRouter | "

    "LLM Judge | DeepEval | Feedback Loop"

)