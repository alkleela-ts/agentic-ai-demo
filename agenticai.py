import requests
from crewai import Agent, Task, Crew
from crewai.llm import LLM
from transformers import pipeline

# ── 1) Fetch & filter CDC data once ──
def fetch_data(limit: int = 50):
    url = "https://data.cdc.gov/resource/bi63-dtpu.json"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()[:limit]

def sort_and_filter_by_year(records, years):
    # sort by numeric year
    sorted_records = sorted(records, key=lambda r: int(r.get("year", 0)))
    # keep only the target years
    return [r for r in sorted_records if int(r.get("year", 0)) in years]

# fetch & filter
raw   = fetch_data(50)
filt  = sort_and_filter_by_year(raw, years=[2016, 2017])
if not filt:
    raise RuntimeError("No records found for 2016–2017.")

# sample up to 5 for brevity
sample = filt[:5]
# Detect the unique years in your sample
years_present = sorted({ int(r["year"]) for r in sample })

# Turn it into a human-friendly string
years_str = ", ".join(str(y) for y in years_present)

# ── 2) Custom in-process TinyLlama Chat LLM ──
class TinyLlamaChatLLM(LLM):
    def __init__(self):
        super().__init__(model="TinyLlama-Chat")
        self.generator = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device_map="auto",
            torch_dtype="auto",
            return_full_text=False,
            do_sample=False,
            max_new_tokens=500,
        )

    def call(self, prompt: str, **kwargs) -> str:
        chat = (
            "<|system|>\n"
            "You are a data-cleaning and analysis assistant.\n"
            "<|end|>\n\n"
            "<|user|>\n"
            f"{prompt}\n"
            "<|end|>\n\n"
            "<|assistant|>\n"
        )
        out = self.generator(chat, max_new_tokens=kwargs.get("max_new_tokens",500))[0]["generated_text"]
        return out.strip()

llm = TinyLlamaChatLLM()

# ── 3) Define Agents ──
data_fetcher = Agent(
    role="CDC Data Fetcher",
    goal="Clean and standardize the sample vital-stats records (2016–2017).",
    backstory="Expert in sanitizing public-health mortality datasets.",
    llm=llm
)

analyzer = Agent(
    role="Health Data Analyst",
    goal="Identify key year-over-year changes in deaths and age-adjusted death rate.",
    backstory="Epidemiologist skilled at deriving insights from numeric trends.",
    llm=llm
)

advisor = Agent(
    role="Health Advisor",
    goal="Formulate actionable public-health recommendations from those trends.",
    backstory="Advisor focused on clear, practical health guidance.",
    llm=llm
)

# ── 4) Create Tasks, feeding the SAME sample into Task 1 ──
# TASK 1: Clean & normalize
task1 = Task(
    description=(
         f"Here are 5 sample records for the years: {years_str}.\n{sample}\n\n"
        "**Please output** a JSON array of objects with these keys:\n"
        "`year` (int), `cause_name` (str), `state` (str), `deaths` (int), `aadr` (float).\n\n"
        "Example:\n```json\n"
        "[{\"year\":2020,\"cause_name\":\"X\",\"state\":\"Y\",\"deaths\":123,\"aadr\":4.5}]\n```"

    ),
    agent=data_fetcher,
    expected_output=""
)


# TASK 2: Trend analysis
task2 = Task(
    description=(
       "Given the **cleaned JSON** from Task 1, calculate:\n"
        "1. The year-over-year difference in `deaths` for each cause and state.\n"
        "2. The year-over-year change in `aadr` for each cause and state.\n\n"
        "**Please output** a JSON array of objects, each with:\n"
        "`cause_name`, `state`, `deaths_change` (int), `aadr_change` (float).\n\n"
        "Example:\n```json\n"
        "[{\"cause_name\":\"Kidney disease\",\"state\":\"VT\",\"deaths_change\":-1,\"aadr_change\":-0.4}]\n```"

    ),
    agent=analyzer,
    expected_output=""
)


# TASK 3: Public-health advice
task3 = Task(
    description=(
         "Based on the JSON results from Task 2, write **2–3 bullet-point recommendations**\n"
        "for public-health officials and at-risk patients.\n\n"
        "**Please output** plain text bullets, e.g.:\n"
        "- Increase vaccination campaigns in states with rising death rates.\n"
        "- …"
    ),
    agent=advisor,
    expected_output=""
)


# ── 5) Assemble & run the Crew ──
crew = Crew(
    agents=[data_fetcher, analyzer, advisor],
    tasks=[task1, task2, task3],
    verbose=True
)

if __name__ == "__main__":
    print("\n Starting Agentic TinyLlama Chat Workflow…\n")
    result = crew.kickoff()
    print("\n Final Agentic AI Output:\n", result)
