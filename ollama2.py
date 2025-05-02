import requests
from crewai import Agent, Task, Crew
from crewai.llm import LLM
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

# ── 1) Fetch & filter your data once ──
def fetch_data(limit: int = 50):
    url = "https://data.cdc.gov/resource/bi63-dtpu.json"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()[:limit]

def sort_and_filter_by_year(records, years):
    sorted_records = sorted(records, key=lambda r: int(r.get("year", 0)))
    return [r for r in sorted_records if int(r.get("year", 0)) in years]

raw = fetch_data(100)
filtered = sort_and_filter_by_year(raw, years=[2016, 2017])
if not filtered:
    raise RuntimeError("No records for 2016–17 found!")
sample = filtered[:5]
print(f"Fetched {len(raw)}, filtered to {len(filtered)}, sampling {len(sample)}.")


model_name = "ollama/llama2:7b-chat"

# ── 2) Custom Ollama LLM ──
class OllamaLLM(LLM):
    def __init__(self, model="ollama/llama2:7b-chat"):
        super().__init__(model=model)
        self.model_name = model
        self.api_url = "http://localhost:11434/api/generate"  # default Ollama endpoint

def call(self, prompt: str, **kwargs) -> str:
    payload = {
        "model": self.model_name,
        "prompt": prompt,
        "stream": False,
        "options": {  # OPTIONAL: max tokens or other tweaks
            "temperature": 0.2,
            "num_predict": kwargs.get("max_new_tokens", 150)
        }
    }
    try:
        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()
        output = response.json()
        return output.get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"

    
# Instantiate the Ollama LLM
llm = OllamaLLM()

# ── 3) Define your three Agents ──
data_fetcher = Agent(
    role="CDC Data Fetcher",
    goal="Clean & standardize CDC sample records.",
    backstory="Expert at sanitizing public-health mortality data.",
    llm=llm
)
analyzer = Agent(
    role="Health Data Analyst",
    goal="Compute year-over-year changes in deaths and AADR.",
    backstory="Epidemiologist skilled at numeric trend analysis.",
    llm=llm
)
advisor = Agent(
    role="Health Advisor",
    goal="Produce actionable bullet-point recommendations.",
    backstory="Advisor focused on clear public-health guidance.",
    llm=llm
)

# ── 4) Create Tasks using your sample ──
task1 = Task(
    description=(
        f"Sample records for {', '.join(str(r['year']) for r in sample)}:\n{sample}\n\n"
        "Please output a JSON array of objects with keys:\n"
        "`year`(int),`cause_name`(str),`state`(str),`deaths`(int),`aadr`(float).\n"
        "Example:\n```json\n"
        "[{\"year\":2016,\"cause_name\":\"Kidney disease\",\"state\":\"Vermont\",\"deaths\":30,\"aadr\":3.7}]\n```"
    ),
    agent=data_fetcher,
    expected_output=""
)

task2 = Task(
    description=(
        "Using the cleaned JSON from Task 1, compute for each cause/state:\n"
        "1. `deaths_change` = deaths(2017) – deaths(2016)\n"
        "2. `aadr_change` = aadr(2017) – aadr(2016)\n\n"
        "Output a JSON array of objects:\n"
        "`cause_name`, `state`, `deaths_change`(int), `aadr_change`(float).\n"
        "Example:\n```json\n"
        "[{\"cause_name\":\"Kidney disease\",\"state\":\"Vermont\",\"deaths_change\":-1,\"aadr_change\":-0.4}]\n```"
    ),
    agent=analyzer,
    expected_output=""
)

task3 = Task(
    description=(
        "Based on Task 2’s JSON results, write **2–3 bullet points** of recommendations:\n"
        "- One for public-health officials\n"
        "- One for at-risk patients\n\n"
        "Example:\n"
        "- Increase screening in states with rising death rates.\n"
        "- …"
    ),
    agent=advisor,
    expected_output=""
)

# ── 5) Run the Crew ──
crew = Crew(
    agents=[data_fetcher, analyzer, advisor],
    tasks=[task1, task2, task3],
    verbose=True
)

if __name__ == "__main__":
    print("\n Starting Agentic Llama-2-7B-Chat 4-bit Workflow…\n")
    result = crew.kickoff()
    print("\n Final Agentic AI Output:\n", result)
