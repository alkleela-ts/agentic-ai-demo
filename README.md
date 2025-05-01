# Local Agentic AI CDC Health Mortality Tracker & Analyzer

## Description
A fully local agentic AI system that fetches real-time CDC mortality data, cleans & analyzes it, and provides human-friendly health recommendations. Built with CrewAI + TinyLlama running locally (no external APIs).

---

## Project Overview
- **Data Fetcher Agent:** Retrieves and cleans public CDC health data.
- **Trend Analyzer Agent:** Analyzes trends (e.g., changes in deaths & AADR).
- **Advisor Agent:** Crafts actionable health advice based on trends.

Uses:
- 📊 Public CDC dataset (open data)
- 🤖 Local LLM: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- 🔗 CrewAI orchestration

**Fully local execution:** No API calls beyond downloading the public dataset.
**Ethical & governed:** Uses open data + governed model.

---

## 🔧 Full Setup Guide

### 1️⃣ Install Python 3.11

- **macOS:**
```bash
brew install python@3.11
```

- **Ubuntu/Linux:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

- **Windows:**  
Download & install from: https://www.python.org/downloads/release/python-3110/

👉 **Check version:**
```bash
python3.11 --version
```

---

### 2️⃣ Create & Activate a Virtual Environment

```bash
python3.11 -m venv agentic-env
```

- **macOS/Linux:**
```bash
source agentic-env/bin/activate
```

- **Windows:**
```cmd
agentic-env\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the App

```bash
python3 agenticai.py
```


## 🔒 License
_This project is for demo purposes only (no license applied)._

