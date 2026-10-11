# AI-Powered Energy Consumption Optimization Assistant

## Project Overview

The **AI-Powered Energy Consumption Optimization Assistant** is a local
application that helps users explore historical energy-consumption
records, retrieve similar cases, and review energy optimization
recommendations. It combines data preprocessing, hybrid retrieval
(semantic and keyword search), rule-based logic, and optional
LLM-assisted analysis.

The application is designed as a decision-support prototype. Cost,
carbon, and potential-reduction values are estimates based on supplied
scenario assumptions; they are not verified post-intervention savings.

## Problem Statement

Energy-consumption records can be difficult to investigate manually.
Analysts need a way to search for comparable historical cases,
understand time-related consumption patterns, and review possible
actions using the available evidence. This project provides a
natural-language interface over processed energy cases and a
recommendation workflow.

## Business Objective

-   Make relevant historical energy cases easier to find.
-   Support investigation of high-consumption and peak-period scenarios.
-   Provide rule-based and optional LLM-assisted recommendations.
-   Allow scenario-based cost and emissions estimates when the required
    inputs are supplied.
-   Support feedback and evaluation to help review the system's output
    quality.

## Solution Summary

1.  Load the source energy dataset and preprocess its timestamp and
    consumption/context fields.
2.  Create processed energy cases with time, consumption, and
    anomaly-related features.
3.  Generate text embeddings and store searchable representations in the
    vector database.
4.  Retrieve relevant cases using semantic similarity and keyword
    matching.
5.  Send the query, retrieved context, and optional scenario inputs
    through the analysis/recommendation workflow.
6.  Return results through FastAPI and the Streamlit interface.
7.  Support feedback and evaluation through dedicated API endpoints and
    tests.

The application uses a coordinator/unified workflow. Do not interpret
each agent class as a separate LangGraph node unless the current source
code explicitly defines it that way.

## Technology Stack

  ------------------------------------------------------------------------------
  Category                            Technology / component
  ----------------------------------- ------------------------------------------
  Programming language                Python

  Backend API                         FastAPI

  Request validation                  Pydantic

  Frontend                            Streamlit

  Embedding model                     `sentence-transformers/all-MiniLM-L6-v2`

  Vector database                     ChromaDB

  Retrieval                           Semantic vector search plus keyword search

  LLM integration                     OpenRouter-compatible API configuration

  Configured model                    `openai/gpt-4.1` (as configured in the
                                      project environment)

  Testing                             Pytest

  Data format                         CSV
  ------------------------------------------------------------------------------

**LLM fallback:** The project includes rule-based
recommendation/analysis paths so users can test supported functionality
without relying exclusively on an LLM response. LLM-based evaluation is
optional and may be disabled to avoid unintended API calls or costs.

## Dataset

-   **Dataset name:** Appliances Energy Prediction
-   **Source:** UCI Machine Learning Repository
-   **Dataset link:**
    https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
-   **Raw dataset path:** `data/raw/energy_consumption.csv`
-   **Processed dataset path:** `data/processed/energy_cases.csv`
-   **Knowledge-base path:**
    `data/knowledge_base/energy_best_practices.csv`

The source dataset contains timestamped appliance energy-consumption
observations and indoor/outdoor environmental measurements. The
project's ingestion output reported **19,735 processed cases**.

Preprocessing derives time features such as hour, day of week, month,
year, season, period, and peak-hour indicators. It also derives
rolling-average, consumption-deviation, consumption-level, and
anomaly-related fields, plus case identifiers and descriptions.
Consumption categories and anomaly flags are derived by project logic;
they are not independently verified ground-truth labels. The source data
represents a single residential setting.

## Project Structure

``` text
Lavenia_EnergyOptimization/
├── src/
│   ├── api/              # FastAPI routes and request schemas
│   ├── agents/           # Coordinator and analysis/recommendation components
│   ├── rag/              # Embeddings, vector store, hybrid and keyword retrieval
│   ├── services/         # Preprocessing and evaluation services
│   ├── utils/            # Configuration and helper utilities
│   └── main.py           # FastAPI application
├── frontend/
│   └── frontend.py       # Streamlit frontend
├── data/
│   ├── raw/
│   ├── processed/
│   └── knowledge_base/
├── scripts/              # Ingestion scripts, if present in the checked-out source
├── tests/                # Automated tests
├── architecture/
│   └── Architecture_Diagram.pdf
├── design/
│   └── Design_Document.pdf
├── docs/
│   ├── API_Documentation.pdf
│   ├── Dataset_Details.pdf
│   └── Evaluation_Report.pdf
├── presentation/
│   └── Project_Presentation.pptx
├── requirements.txt
└── README.md
```

Keep this tree aligned with the actual files included in your final
submission. Do not include generated local databases, virtual
environments, or credentials in the repository.

## Installation Instructions

Run the following commands from the project root in PowerShell.

### 1. Create and activate a virtual environment

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell prevents script activation, use an approved activation
method for your environment or call the virtual environment's Python
executable directly.

### 2. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a dependency installation fails, review the error and the package
versions in `requirements.txt` rather than installing arbitrary
versions.

## Environment Setup

The application may use environment variables for LLM access and
optional evaluation features. Check `src/utils/config.py` and the
current LLM integration code for the exact variable names expected by
your working copy.

Create a local `.env` file in the project root if required. Example
settings used by this project include:

``` dotenv
OPENROUTER_API_KEY=replace_with_your_own_key
OPENROUTER_MODEL=openai/gpt-4.1
ENABLE_LLM_JUDGE=false
ENABLE_DEEPEVAL=false
```

-   Replace the placeholder with your own API key.
-   Never commit `.env`, API keys, or other credentials to GitHub.
-   Keep optional model-based evaluation disabled during routine use
    unless you deliberately intend to run it and accept possible API
    costs.
-   Do not paste API keys into screenshots, README files, logs, or
    source code.

## Dependency Installation and Search Index Preparation

If the ingestion script exists in your checked-out source, run it from
the project root after installing dependencies:

``` powershell
python -m scripts.ingest
```

This step loads processed cases, creates embeddings, and populates the
vector store. The first run can take time while the embedding model is
loaded and records are embedded. If the source dataset or preprocessing
code changes, regenerate the processed data using the preprocessing
workflow implemented in the current source before re-running ingestion.

## API Key Configuration

The application can use an OpenRouter-compatible API key for optional
LLM-assisted functionality. Configure the key in the local `.env` file
using the variable name expected by the current configuration module.

Do not publish the key. If no valid key is configured, use the project's
rule-based recommendation endpoint where applicable. Availability of
each fallback depends on the selected endpoint and current application
configuration.

## Local Deployment Instructions

Start the backend and frontend in separate terminals.

### Terminal 1 --- Start the FastAPI backend

``` powershell
uvicorn src.main:app --reload
```

Local URLs:

-   Backend base URL: http://127.0.0.1:8000
-   Swagger UI: http://127.0.0.1:8000/docs
-   Health check: http://127.0.0.1:8000/api/health

### Terminal 2 --- Start the Streamlit frontend

``` powershell
streamlit run frontend/frontend.py
```

Frontend URL:

-   http://localhost:8501

These are local development URLs, not public deployment links.

## API Endpoints

  -------------------------------------------------------------------------
  Method                  Endpoint                  Purpose
  ----------------------- ------------------------- -----------------------
  GET                     `/`                       API service information

  GET                     `/api/health`             Health check

  POST                    `/api/analyze`            Analyze an energy
                                                    scenario

  POST                    `/api/search`             Search historical cases

  POST                    `/api/recommend/llm`      Request an LLM-assisted
                                                    recommendation

  POST                    `/api/recommend/rule`     Request a rule-based
                                                    recommendation

  POST                    `/api/evaluate`           Evaluate a query using
                                                    configured evaluation
                                                    logic

  POST                    `/api/feedback`           Submit user feedback

  GET                     `/api/feedback/summary`   Retrieve feedback
                                                    summary

  GET                     `/api/feedback/records`   Retrieve stored
                                                    feedback records
  -------------------------------------------------------------------------

Use the running Swagger UI at http://127.0.0.1:8000/docs for the exact
request schemas and live response payloads.

## Sample Queries and Expected Outputs

The following are **illustrative examples**, not fixed response
snapshots. Actual results depend on the processed dataset, current
vector-store contents, filters, model configuration, and supplied
scenario inputs.

### Query 1 --- Peak-period investigation

**Sample query**

> How can I reduce high energy consumption during peak hours?

**Expected output**

The application attempts to retrieve relevant cases associated with high
consumption and peak-period context, then returns an analysis or
recommendation through the selected workflow. The exact cases and advice
depend on the current index and implementation.

### Query 2 --- Find high-consumption cases

**Sample query**

> Find historical cases with high energy consumption.

**Expected output**

The search endpoint returns a success indicator, the validated query, a
list of matching case records, and a result count when matching cases
are found. The result list can vary with the query and index.

### Query 3 --- Seasonal patterns

**Sample query**

> What patterns are visible during winter energy consumption?

**Expected output**

The application attempts to retrieve cases matching the seasonal context
where the selected endpoint supports the relevant filter/metadata. The
returned evidence depends on available records.

### Query 4 --- Anomaly-related investigation

**Sample query**

> Find cases marked as anomalous during high consumption.

**Expected output**

The application searches for cases matching available anomaly and
consumption metadata. Any anomaly flag is a project-derived indicator,
not an independently verified incident label.

### Query 5 --- Practical optimization suggestions

**Sample query**

> Suggest practical ways to reduce energy use without disrupting
> essential activities.

**Expected output**

The selected rule-based or LLM-assisted recommendation path returns
suggestions based on the current workflow and retrieved context. Cost,
carbon, or reduction values should be treated as estimates when they
rely on supplied assumptions.

## Testing

Run the automated test suite from the project root:

``` powershell
python -m pytest tests/ -v
```

Use the latest terminal output to report the actual pass/fail count.
Review live LLM tests before running them because they may make external
API calls or incur usage costs.

Quick health check:

``` powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Expected health response:

``` json
{
  "status": "healthy",
  "service": "Energy Optimization Assistant"
}
```

## Evaluation

The project contains retrieval and recommendation evaluation utilities,
plus optional LLM-based evaluation. Record the metric name, score,
query/test set, model, and evaluation method when reporting results.
Model-judge scores can vary across runs, and rule-derived labels are not
human-verified ground truth.

Do not claim RAGAS scores unless RAGAS has actually been run and its
results recorded. Do not claim measured energy savings without measured
post-intervention results.

## Limitations and Future Work

-   The source dataset represents a single residential setting.
-   Consumption-level and anomaly-related labels are derived by project
    rules.
-   Cost, carbon, and potential-reduction values depend on scenario
    assumptions.
-   Retrieval quality depends on dataset coverage, metadata, and query
    wording.
-   Future work can include broader facility datasets, human-annotated
    evaluation cases, measured intervention outcomes, and deployment
    monitoring.

## Security and GitHub Submission Checklist

Before pushing the project:

-   Ensure `.env` and API keys are excluded from version control.
-   Exclude virtual environments, Python caches, local vector-store
    databases, and generated temporary files where appropriate.
-   Review `git status` and inspect the files staged for commit.
-   Include the required architecture, design, API, dataset,
    presentation, and evaluation documents in their specified folders.
-   Verify that the README's file tree matches the final project folder.
