# I built prompt templates here so I could swap prompting styles easily.
# These functions create the final prompt string from examples and article text.
def buildFewShotPrompt(articleText: str, shotPairs: list[dict]) -> str:
    parts = []
    parts.append("You are a helpful assistant that writes faithful summaries.")
    parts.append("Write a summary in 1-2 sentences. Do not invent facts.")
    for p in shotPairs:
        parts.append("Article:")
        parts.append(p["article"].strip())
        parts.append("Summary:")
        parts.append(p["summary"].strip())
    parts.append("Article:")
    parts.append(articleText.strip())
    parts.append("Summary:")
    return "\n".join(parts)

def buildStructuredFewShotPrompt(articleText: str, shotPairs: list[dict]) -> str:
    parts = []
    parts.append("Task: Summarize the article faithfully.")
    parts.append("Rules: 1) Use 1-2 sentences. 2) Only use article facts. 3) No extra names or numbers.")
    for p in shotPairs:
        parts.append("[ARTICLE]")
        parts.append(p["article"].strip())
        parts.append("[SUMMARY]")
        parts.append(p["summary"].strip())
    parts.append("[ARTICLE]")
    parts.append(articleText.strip())
    parts.append("[SUMMARY]")
    return "\n".join(parts)