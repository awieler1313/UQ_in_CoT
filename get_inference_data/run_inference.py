import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm import tqdm

"""
This code is used for generating 10 responses to each prompt in the GSM8K data set
Run this for the training set and test set, I split the trainign set into two to run
in parallel.
"""

## paths
model_name = "models/Mistral-7B-Instruct-v0.2"
print(model_name)
data_path = "data/gsm8k_local_copy"
save_path = "data/gsm8k_inference_results_train1.jsonl"

# hyper parameters
num_gen = 10
max_new_tokens = 500

## load data
gsm8k = load_from_disk(data_path)
full_test_data = gsm8k["train"].select(range(4000))
print(f"Loaded {len(full_test_data)} test samples")

# Load model and atokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True
)
model.eval()

## utils

def make_prompt(question):
    return f"<s>[INST] Solve this problem step by step, and give the final numeric answer after '####', stop after generating the final numeric answer.\n\nQuestion: {question} [/INST]"

def generate_k_responses(model, tokenizer, question, k=10, max_new_tokens=500):
    prompt = make_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    decoded = []
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
            num_return_sequences=k,
        )
    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return decoded

def extract_answer(text: str):
    text = text.strip().lower()

    # Match times, comma numbers, decimals, and integers
    pattern = r"(\d{1,2}:\d{2}\s?(?:am|pm)?)|(\d{1,3}(?:,\d{3})+)|(\d+\.\d+)|(\d+)"
    matches = re.findall(pattern, text)

    if matches:
        # Flatten match groups
        all_matches = [m for group in matches for m in group if m]
        if all_matches:
            raw = all_matches[-1]

            # If it contains commas, remove them
            raw_no_commas = raw.replace(",", "")

            # If it's a decimal, normalize
            if re.match(r"^\d+\.\d+$", raw_no_commas):
                val = float(raw_no_commas)
                # If it's actually an integer like 18.00 -> "18"
                if val.is_integer():
                    return str(int(val))
                return raw_no_commas  # real decimal, keep as-is

            return raw_no_commas  # simple integer or time

    return None




print("running inference...")
results = []
samples = full_test_data  # for testing; remove for full run

save_interval = 100  # save every 100 examples

for i, ex in enumerate(tqdm(samples, desc="Generating"), 1):
    question = ex["question"]
    gold = extract_answer(ex["answer"])
    responses = generate_k_responses(model, tokenizer, question, k=num_gen, max_new_tokens=max_new_tokens)

    # compute correctness
    entry = {
        "question": question,
        "gold_answer": gold,
        "responses": []
    }

    for response in responses:
        pred = extract_answer(response)
        correct = int(pred == gold)
        entry["responses"].append({
            "raw_response": response,
            "pred_answer": pred,
            "correct": correct
        })

    results.append(entry)

    # Save every save_interval examples
    if i % save_interval == 0:
        print(f"Saving intermediate results at example {i}...")
        with open(save_path, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        results = []  # clear memory

# Save any remaining results
if results:
    print("Saving remaining results...")
    with open(save_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Saved results to {save_path}")
