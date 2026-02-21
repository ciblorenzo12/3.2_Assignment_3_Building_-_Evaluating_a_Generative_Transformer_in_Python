# I put metric scoring here so quality evaluation is easy to reuse.
# It computes ROUGE and BERTScore, then returns the results in one dictionary.
import evaluate

def computeMetrics(predictions: list[str], references: list[str], useRouge: bool, useBertScore: bool) -> dict:
    metrics = {}
    if useRouge:
        rouge = evaluate.load("rouge")
        metrics["rouge"] = rouge.compute(predictions=predictions, references=references)
    if useBertScore:
        bert = evaluate.load("bertscore")
        out = bert.compute(predictions=predictions, references=references, lang="en")
        metrics["bertscore"] = {
            "precisionMean": sum(out["precision"]) / len(out["precision"]),
            "recallMean": sum(out["recall"]) / len(out["recall"]),
            "f1Mean": sum(out["f1"]) / len(out["f1"])
        }
    return metrics