# I kept config loading in one place so the rest of the code stays clean.
# This reads a YAML file and returns it as a normal Python dictionary.
import yaml

def loadConfig(configPath: str) -> dict:
    with open(configPath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)