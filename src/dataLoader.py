# I used this helper to load summarization data from Hugging Face or local JSONL.
# It also applies the row limit from config so experiments run at the right size.
from datasets import load_dataset

def loadSummarizationData(datasetConfig: dict):
    if datasetConfig["source"] == "hf":
        ds = load_dataset(datasetConfig["name"], split=datasetConfig["split"])
        limit = int(datasetConfig.get("limit", len(ds)))
        return ds.select(range(min(limit, len(ds))))
    if datasetConfig["source"] == "localJsonl":
        ds = load_dataset("json", data_files=datasetConfig["path"], split="train")
        limit = int(datasetConfig.get("limit", len(ds)))
        return ds.select(range(min(limit, len(ds))))
    raise ValueError("Unsupported dataset source")