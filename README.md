# 🩺 MedCheck AI — Multimodal Health Report Agent

MedCheck AI is an intelligent health report analyzer powered by **Google Gemini** and **Streamlit**. It extracts clinical data from medical lab reports (PDFs and images), tracks historical test trends in SQLite, and provides an agentic conversational assistant with built-in medical safety tripwires.

---

## ✨ Features

- 📄 **Multimodal Extraction:** Upload lab reports in PDF or image (PNG/JPG) format to automatically extract lab test names, values, units, reference ranges, and confidence levels.
- 🗄️ **Persistent Health Tracking:** Stores and organizes lab history in a local SQLite database with interactive inline table editing.
- 🛠️ **Deterministic Health Tools:** Calculates precise numeric changes, percentage trends, and flags out-of-range or critical biomarker levels.
- 🤖 **Agentic AI Assistant:** Explains lab results in clear, natural language with strict grounding and verifiable tool execution logs.
- 🛡️ **Built-in Safety Tripwires:** Refuses medical diagnoses or prescriptions, directs users to healthcare professionals, and flags critical physiological thresholds.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))


## ⚠️ Disclaimer
*MedCheck AI is for informational and educational purposes only. It does not provide medical advice, diagnosis, or treatment recommendations. Always consult a licensed healthcare professional for medical concerns.*

