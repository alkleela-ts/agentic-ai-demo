from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "TinyLlama/TinyLlama_v1.1"

print("Loading TinyLlama…")
generator = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=model_name,
    device_map="auto",      # will use MPS if available
    torch_dtype="auto" ,     # float16 on MPS
    return_full_text=False
)

out = generator("Q:Hello, what is the capital of France?\nA:", max_new_tokens=30, num_return_sequences=1,do_sample=False)[0]["generated_text"]
print("→", out)
