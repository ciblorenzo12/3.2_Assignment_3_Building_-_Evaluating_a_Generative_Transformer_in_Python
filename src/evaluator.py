# I put metric scoring here so quality evaluation is easy to reuse.
# It computes ROUGE and BERTScore, then returns the results in one dictionary.
import evaluate

def computeMetrics(predictions: list[str], references: list[str], useRouge: bool, useBertScore: bool) -> dict:
    metrics = {}
    safePredictions = [p if isinstance(p, str) and p.strip() else "[EMPTY]" for p in predictions]
    safeReferences = [r if isinstance(r, str) and r.strip() else "[EMPTY]" for r in references]

    if useRouge:
        rouge = evaluate.load("rouge")
        metrics["rouge"] = rouge.compute(predictions=safePredictions, references=safeReferences)
    if useBertScore:
        bert = evaluate.load("bertscore")
        try:
            out = bert.compute(
                predictions=safePredictions,
                references=safeReferences,
                lang="en",
                use_fast_tokenizer=True
            )
        except TypeError:
            out = bert.compute(predictions=safePredictions, references=safeReferences, lang="en")
        metrics["bertscore"] = {
            "precisionMean": sum(out["precision"]) / len(out["precision"]),
            "recallMean": sum(out["recall"]) / len(out["recall"]),
            "f1Mean": sum(out["f1"]) / len(out["f1"])
        }
    return metrics