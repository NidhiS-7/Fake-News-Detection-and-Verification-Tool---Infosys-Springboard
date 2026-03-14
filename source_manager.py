import json
from pathlib import Path
from typing import List

TRUSTED_SOURCE_PATH = Path("data/trusted_sources.json")


def load_trusted_sources(path: Path = TRUSTED_SOURCE_PATH) -> List[str]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return [item.lower().strip() for item in data.get("trusted_sources", [])]


def is_source_trusted(source_name: str) -> bool:
    if not source_name or not source_name.strip():
        return False
    return source_name.lower().strip() in load_trusted_sources()
