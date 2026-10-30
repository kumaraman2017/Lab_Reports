# 🩺 MedCheck AI — Multimodal Health Report Agent

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Google Gemini](https://img.shields.io/badge/Google%20GenAI-2.24%2B-orange.svg)
![Database](https://img.shields.io/badge/Database-SQLite-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**MedCheck AI** is an intelligent, privacy-preserving clinical lab report analyzer powered by **Google Gemini** and **Streamlit**. It ingests multimodal health reports (PDFs, scans, and mobile photos), structures and normalizes biomarker data into a local SQLite database, and provides a grounded conversational AI agent equipped with deterministic mathematical tools and medical safety tripwires.

---

## 📑 Table of Contents
- [✨ Core Capabilities](#-core-capabilities)
- [🏗️ System Architecture](#️-system-architecture)
- [🛠️ Deterministic Agent Tools](#️-deterministic-agent-tools)
- [🛡️ Safety & Guardrails](#️-safety--guardrails)
- [🚀 Quickstart (Local Setup)](#-quickstart-local-setup)
- [☁️ Deploying to Streamlit Community Cloud](#️-deploying-to-streamlit-community-cloud)
- [📂 Project Structure](#-project-structure)
- [🧪 Running Tests](#-running-tests)
- [⚠️ Medical Disclaimer](#️-medical-disclaimer)

---

## ✨ Core Capabilities

### 1. 📄 Multimodal Document Ingestion
- Upload clinical diagnostic reports as **multi-page PDFs** or image formats (**PNG, JPG, JPEG**).
- Uses `pypdfium2` for PDF rasterization and passes high-resolution page buffers directly to Gemini Vision.
- Extracts structured test metadata: `raw_test_name`, `value`, `unit`, `reference_range`, and test dates.

### 2. 🧹 Defensive Data Cleaning & Biomarker Normalization
- Normalizes disparate laboratory test aliases into canonical names (e.g., `FBS`, `Fasting Blood Sugar`, and `Fasting Glucose` map to `Glucose, Fasting`).
- Evaluates extraction quality and tags each record with confidence metrics (`HIGH`, `MEDIUM`, `LOW`).

### 3. 📊 Interactive Record Correction & Storage
- Structured records are persisted to a local **SQLite** database (`health_reports.db`).
- Embedded `st.data_editor` allows users to review, edit, and validate extracted values and reference ranges before querying the agent.

### 4. 🤖 Grounded Conversational AI Agent
- Powered by Google Gemini (`gemini-3.6-flash` / `gemini-3.8-flash` / `gemini-2.5-flash`).
- Uses native function calling to retrieve longitudinal records, evaluate trends, and compare out-of-range biomarkers.
- Transparent execution logs show exact tool arguments and deterministic returns in an expander accordion.

---

## 🏗️ System Architecture

```text
[User Upload: PDF / Image]
          │
          ▼
   [pypdfium2 / PIL]  ──►  Render Page Images
          │
          ▼
   [Gemini Multimodal]  ──►  Structured JSON Extraction
          │
          ▼
[Normalization & Cleaning] ──► Alias Resolution & Range Parsing
          │
          ▼
    [SQLite DB]        ──► Stored Historical Biomarkers
          │
          ▼
   [Streamlit UI]      ──► Editable Data Table & Agent Chat
          │
          ▼
[Grounded Agent Loop]  ──► Deterministic Python Tools & Safety Tripwire
```

---

## 🛠️ Deterministic Agent Tools

To eliminate LLM hallucinations on clinical mathematics and range checks, the agent delegates calculations to deterministic Python functions:

| Tool Name | Purpose | Output |
| :--- | :--- | :--- |
| `get_latest_result` | Retrieves the most recent measurement for a specific test | Canonical name, date, value, unit, status |
| `get_test_history` | Analyzes historical values across dates | Chronological values, absolute delta, % change, and trend direction |
| `list_out_of_range_results` | Scans records for values exceeding reference ranges | Out-of-bounds biomarkers, low/high flags |
| `get_all_test_names` | Discovers available tests currently stored in database | List of unique normalized test names |

---

## 🛡️ Safety & Guardrails

1. **Non-Diagnostic Grounding:** The assistant explains findings in plain language but is strictly forbidden from diagnosing conditions or recommending prescription medications.
2. **Deterministic Calculations:** Numerical percentage changes and trends are computed via Python math, not guessed by the model.
3. **Critical Value Tripwires:** High-risk biomarker results (e.g., severe hypoglycemia, dangerously elevated potassium) automatically trigger a highlighted medical safety notice directing the user to seek immediate professional care.

---

## 🚀 Quickstart (Local Setup)

### Prerequisites
- Python 3.10, 3.11, or 3.12
- A Google Gemini API Key ([Get one from Google AI Studio](https://aistudio.google.com/))

### 1. Clone the repository
```bash
git clone https://github.com/kumaraman2017/Lab_Reports.git
cd Lab_Reports
```

### 2. Create and activate a virtual environment
```bash
# On Windows:
python -m venv .venv
.\.venv\Scripts\activate

# On macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API Key
Create a `.streamlit/secrets.toml` file in the project root:
```toml
GEMINI_API_KEY = "your-actual-gemini-api-key"
```
*(Alternatively, set the `GEMINI_API_KEY` system environment variable or enter it directly into the sidebar).*

### 5. Launch the application
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

---

## ☁️ Deploying to Streamlit Community Cloud

Streamlit Community Cloud hosts your application directly from your GitHub repository:

1. **Push your code to GitHub:**
   ```bash
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
   git push -u origin main
   ```
2. **Connect to Streamlit Cloud:**
   - Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
   - Click **Create app** and select your repository, branch (`main`), and main file (`app.py`).
3. **Add Secrets:**
   - In **Advanced settings...** > **Secrets**, enter:
     ```toml
     GEMINI_API_KEY = "your-gemini-api-key"
     ```
   - Click **Deploy!**

---

## 📂 Project Structure

```
LLM_Lab_Reports/
├── .streamlit/
│   └── secrets.toml          # Local secrets (ignored by Git)
├── app.py                    # Main Streamlit application & agent engine
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Protects secrets, venv, and local SQLite DB
├── health_reports.db         # Local SQLite database (auto-generated)
└── test_app_core.py          # Core verification test suite
```

---

## 🧪 Running Tests

Verify database initialization, schema integrity, alias resolution, deterministic tool math, and safety tripwires:

```bash
python test_app_core.py
```

---

## ⚠️ Medical Disclaimer

> **IMPORTANT:** MedCheck AI is designed solely for informational and educational purposes. It is **not** a certified medical device and does **not** provide clinical diagnosis, medical advice, or treatment plans. Laboratory results should always be interpreted by a qualified, licensed healthcare professional in the context of your complete medical history.
