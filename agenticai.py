# agentic_custom_llm.py

from crewai import Agent, Task, Crew
from crewai.llm import LLM
from transformers import pipeline
import requests

# ── 1) Define a custom in-process LLM by subclassing crewai.llm.LLM ──
class TinyLlamaLLM(LLM):
    def __init__(self):
        super().__init__(model="TinyLlama-Chat")
        # initialize the Hugging Face pipeline once
        self.generator = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device_map="auto",      # use MPS on Apple Silicon
            torch_dtype="auto",     # float16 on MPS
            return_full_text=False,
            do_sample=False,
            max_new_tokens=100,
        )

    def call(self, prompt: str, **kwargs) -> str:
        # wrap in Q/A format so base model knows to answer
        qa = f"Q: {prompt}\nA:"
        gen = self.generator(
            qa,
            max_new_tokens=kwargs.get("max_new_tokens", 100)
        )[0]["generated_text"]
        # strip off the "A:" prefix
        return gen.split("A:")[-1].strip()

# ── 2) Helper to fetch CDC flu hospitalization data ──
def fetch_cdc_data():
    url = "https://data.cdc.gov/resource/bi63-dtpu.json"
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.json()[:100]    # limit to first three for brevity
    raise RuntimeError(f"Failed to fetch CDC data: {resp.status_code}")

# ── 3) Instantiate the LLM ──
llm = TinyLlamaLLM()

# ── 4) Define Agents ──
data_fetcher = Agent(
    role="CDC Data Fetcher",
    goal="Collect and clean the latest CDC flu hospitalization data.",
    backstory="An expert in public-health data cleaning and formatting.",
    llm=llm
)

analyzer = Agent(
    role="Health Data Analyst",
    goal="Analyze cleaned flu data and identify key trends.",
    backstory="Epidemiologist skilled at spotting patterns in time-series data.",
    llm=llm
)

advisor = Agent(
    role="Health Advisor",
    goal="Provide simple, actionable health advice based on trends.",
    backstory="A public-health advisor focused on clear recommendations.",
    llm=llm
)

# ── 5) Define Tasks (with empty expected_output to satisfy schema) ──
raw = fetch_cdc_data()

task1 = Task(
    description=(
        f"Here are the first 3 records of CDC flu hospitalization data:\n{raw}\n\n"
        "Please clean it (format dates, remove nulls) and summarize the key fields."
    ),
    agent=data_fetcher,
    expected_output=""
)

task2 = Task(
    description=(
        "Based on the cleaned data, identify which weeks saw increases "
        "or decreases in hospitalizations and summarize the trends."
    ),
    agent=analyzer,
    expected_output=""
)

task3 = Task(
    description=(
        "Given the trend analysis, write clear health advice for the public, "
        "e.g., who should consider extra precautions or vaccination."
    ),
    agent=advisor,
    expected_output=""
)

# ── 6) Create and run the Crew ──
crew = Crew(
    agents=[data_fetcher, analyzer, advisor],
    tasks=[task1, task2, task3],
    verbose=True
)

if __name__ == "__main__":
    print("\n Starting Agentic TinyLlama Workflow…\n")
    result = crew.kickoff()
    print("\n Final Agentic AI Output:\n", result)
