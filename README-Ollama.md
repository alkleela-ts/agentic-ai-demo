
# 🦙 Local Agentic AI CDC Health Mortality Tracker & Analyzer (Ollama Version)

A **fully local agentic AI system** that:

- ✅ Fetches **real-time CDC mortality data**
- ✅ Cleans & analyzes trends (e.g., deaths, age-adjusted death rates)
- ✅ Generates **human-friendly health recommendations**

**Powered by:**

- 📊 Public CDC dataset (open data)
- 🦙 Local LLM: **LLaMA 2 (7B Chat)** via [Ollama](https://ollama.com/)
- 🔗 CrewAI orchestration

💡 **Fully offline:** No external API calls beyond downloading the dataset. Governed & ethical AI.

---

## 🚀 Why Ollama?

We recommend Ollama because:

- ✅ **Better accuracy & consistency** than smaller models like TinyLlama
- ✅ Easy local deployment
- ✅ Supports **M1/M2 Macs, Windows, and Linux**

---

## 🔧 Full Setup Guide (Ollama + Python)

### 1️⃣ Install Ollama

**macOS / Windows / Linux:**

- 👉 [Download Ollama](https://ollama.com/download)

After installing, confirm it’s running:

```bash
ollama list
```

If you don’t see any models yet, download the LLaMA 2 Chat model:

```bash
ollama pull llama2:7b-chat
```

---

### 2️⃣ Install Python 3.11

#### macOS:

```bash
brew install python@3.11
```

#### Ubuntu/Linux:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

#### Windows:

Download & install from:  
👉 https://www.python.org/downloads/release/python-3110/

Check your version:

```bash
python3.11 --version
```

---

### 3️⃣ Set Up the Virtual Environment

```bash
python3.11 -m venv agentic-env
```

- **macOS/Linux:**

```bash
source agentic-env/bin/activate
```

- **Windows:**

```bash
agentic-env\Scripts\activate
```

---

### 4️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Application

➡️ **First, ensure Ollama is running in the background.**
ollama serve

Then, run:

```bash
python3 agenticai_ollama2.py
```

---

## ⚠️ Notes

- Make sure **Ollama** is running **before** starting your Python script.
- The app connects to Ollama locally at `http://localhost:11434`.
- Tested with **llama2:7b-chat**. You can substitute with other models via Ollama if needed.

---

## 🔒 License

This project is for **demo purposes only** (no license applied).
