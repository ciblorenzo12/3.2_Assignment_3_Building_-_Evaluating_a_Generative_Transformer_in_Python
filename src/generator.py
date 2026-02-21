# I used this function to run text generation and track speed numbers.
# It returns generated text, latency, new token count, and tokens/sec.
import time
import torch

def generateText(model, tokenizer, promptText: str, genConfig: dict) -> dict:
    inputs = tokenizer(promptText, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(genConfig["maxNewTokens"]),
            do_sample=bool(genConfig["doSample"]),
            temperature=float(genConfig["temperature"]),
            top_p=float(genConfig["topP"]),
            pad_token_id=tokenizer.eos_token_id
        )
    end = time.perf_counter()

    fullText = tokenizer.decode(out[0], skip_special_tokens=True)
    generatedPart = fullText[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    tokenCount = int(out.shape[-1] - inputs["input_ids"].shape[-1])
    seconds = max(end - start, 1e-9)

    return {
        "text": generatedPart.strip(),
        "latencySec": seconds,
        "newTokens": tokenCount,
        "tokensPerSec": tokenCount / seconds
    }