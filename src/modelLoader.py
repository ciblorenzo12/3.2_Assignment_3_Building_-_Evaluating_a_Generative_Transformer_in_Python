# I handled model and tokenizer loading here to keep setup in one place.
# It picks dtype/device automatically and ensures a pad token is available.
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def loadModelAndTokenizer(modelId: str):
    tokenizer = AutoTokenizer.from_pretrained(modelId)
    model = AutoModelForCausalLM.from_pretrained(
        modelId,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer