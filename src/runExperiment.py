# I used this as the main experiment runner.
# It loads config/data/model, generates outputs, scores them, and saves results.
import json
import os
from tqdm import tqdm
from transformers import __version__ as transformersVersion
import torch

try:
    from .configLoader import loadConfig
    from .seedUtils import setGlobalSeed
    from .dataLoader import loadSummarizationData
    from .modelLoader import loadModelAndTokenizer
    from .promptBuilder import buildFewShotPrompt, buildStructuredFewShotPrompt
    from .generator import generateText
    from .evaluator import computeMetrics
except ImportError:
    from configLoader import loadConfig
    from seedUtils import setGlobalSeed
    from dataLoader import loadSummarizationData
    from modelLoader import loadModelAndTokenizer
    from promptBuilder import buildFewShotPrompt, buildStructuredFewShotPrompt
    from generator import generateText
    from evaluator import computeMetrics

def ensureDir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def pickShotPairs(dataset, numShots: int) -> list[dict]:
    pairs = []
    for i in range(numShots):
        ex = dataset[i]
        pairs.append({"article": ex["document"], "summary": ex["summary"]})
    return pairs

def getArticleAndRef(example: dict) -> tuple[str, str]:
    if "document" in example and "summary" in example:
        return example["document"], example["summary"]
    if "article" in example and "highlights" in example:
        return example["article"], example["highlights"]
    raise ValueError("Unsupported dataset fields")

def main(configPath: str) -> None:
    cfg = loadConfig(configPath)
    setGlobalSeed(int(cfg["seed"]))

    ds = loadSummarizationData(cfg["dataset"])
    model, tokenizer = loadModelAndTokenizer(cfg["modelId"])

    shotPairs = pickShotPairs(ds, int(cfg["prompt"]["numShots"]))
    predictions = []
    references = []
    sampleRows = []
    emptyPredictionCount = 0

    for ex in tqdm(ds, desc=cfg["experimentName"]):
        articleText, refSummary = getArticleAndRef(ex)

        if cfg["prompt"]["mode"] == "fewShot":
            promptText = buildFewShotPrompt(articleText, shotPairs)
        elif cfg["prompt"]["mode"] == "structuredFewShot":
            promptText = buildStructuredFewShotPrompt(articleText, shotPairs)
        else:
            raise ValueError("Unsupported prompt mode")

        genOut = generateText(model, tokenizer, promptText, cfg["generation"])
        if not genOut["text"].strip():
            emptyPredictionCount += 1
        predictions.append(genOut["text"])
        references.append(refSummary)

        sampleRows.append({
            "article": articleText,
            "prediction": genOut["text"],
            "reference": refSummary,
            "isEmptyPrediction": not genOut["text"].strip(),
            "latencySec": genOut["latencySec"],
            "tokensPerSec": genOut["tokensPerSec"]
        })

    metrics = computeMetrics(
        predictions,
        references,
        useRouge=bool(cfg["eval"]["rouge"]),
        useBertScore=bool(cfg["eval"]["bertScore"])
    )

    latencyAvg = sum(r["latencySec"] for r in sampleRows) / len(sampleRows)
    tpsAvg = sum(r["tokensPerSec"] for r in sampleRows) / len(sampleRows)

    runInfo = {
        "experimentName": cfg["experimentName"],
        "modelId": cfg["modelId"],
        "torchCudaAvailable": bool(torch.cuda.is_available()),
        "transformersVersion": transformersVersion,
        "dataQuality": {
            "emptyPredictions": emptyPredictionCount,
            "emptyPredictionsPct": (emptyPredictionCount / len(predictions)) if predictions else 0.0
        },
        "qualityMetrics": metrics,
        "efficiency": {"latencyAvgSec": latencyAvg, "tokensPerSecAvg": tpsAvg}
    }

    outDir = cfg.get("outputDir", "results")
    ensureDir(os.path.join(outDir, "metrics"))
    ensureDir(os.path.join(outDir, "samples"))

    metricsPath = os.path.join(outDir, "metrics", f"{cfg['experimentName']}.json")
    samplesPath = os.path.join(outDir, "samples", f"{cfg['experimentName']}.txt")

    with open(metricsPath, "w", encoding="utf-8") as f:
        json.dump(runInfo, f, indent=2)

    with open(samplesPath, "w", encoding="utf-8") as f:
        for idx, row in enumerate(sampleRows, start=1):
            f.write(f"=== Sample {idx} ===\n")
            f.write("ARTICLE:\n")
            f.write(row["article"].strip() + "\n\n")
            f.write("REFERENCE SUMMARY:\n")
            f.write(row["reference"].strip() + "\n\n")
            f.write("GENERATED SUMMARY:\n")
            f.write(row["prediction"].strip() + "\n\n")
            f.write(
                f"Latency (sec): {row['latencySec']:.4f}\n"
                f"Tokens/sec: {row['tokensPerSec']:.2f}\n"
            )
            f.write("\n")

    print(
        f"Empty predictions: {emptyPredictionCount}/{len(predictions)} "
        f"({(emptyPredictionCount / len(predictions) * 100) if predictions else 0.0:.2f}%)"
    )

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)