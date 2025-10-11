"""Persona fetching/cleaning and prompt generation."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import requests
import os
from datetime import datetime
from pathlib import Path


def _logs_dir() -> str:
    """Return absolute path to the project's logs directory."""
    # duplib/ -> duplicity/
    project_root = Path(__file__).resolve().parent.parent
    logs_path = project_root / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    return str(logs_path)


def save_personas_log(personas_dict: Dict, offset: int = 0, length: int = 100, tag: Optional[str] = None) -> str:
    """Persist personas to logs as JSON and return the saved file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    filename = f"personas_{timestamp}_off{offset}_len{length}{tag_part}.json"
    path = os.path.join(_logs_dir(), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(personas_dict, f, indent=2)
    return path


def list_cached_personas() -> List[str]:
    """List cached persona JSON files (sorted newest first)."""
    logs_path = Path(_logs_dir())
    files = sorted(logs_path.glob("personas_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


def load_personas_from_log(filepath: str) -> Dict:
    """Load personas JSON from a specific log file path."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_latest_personas() -> Optional[Dict]:
    """Load the most recent cached personas if present; otherwise return None."""
    files = list_cached_personas()
    if not files:
        return None
    return load_personas_from_log(files[0])


def get_and_clean_personas(offset: int = 0, length: int = 100, cache: bool = True, tag: Optional[str] = None) -> Dict:
    """Fetch and clean persona data from the Hugging Face dataset.

    If cache=True, persist the cleaned personas into `duplicity/logs`.
    """

    url = (
        "https://datasets-server.huggingface.co/rows?dataset=nvidia%2FNemotron-Personas&config=default&split=train"
        f"&offset={offset}&length={length}"
    )
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:  # noqa: BLE001
        print(f"Error fetching data: {e}")
        return {"rows": []}

    all_personas = data.get("rows", [])
    final_personas_dict = {"rows": []}

    keys_to_remove = [
        "hobbies_and_interests_list",
        "sports_persona",
        "arts_persona",
        "travel_persona",
        "culinary_persona",
        "skills_and_expertise",
        "skills_and_expertise_list",
        "hobbies_and_interests",
    ]

    idx = 1
    for persona_entry in all_personas:
        if "row" in persona_entry:
            current_row = persona_entry["row"]
            for key in keys_to_remove:
                current_row.pop(key, None)
            current_row["uuid"] = idx
            idx += 1
            final_personas_dict["rows"].append(persona_entry)
        else:
            final_personas_dict["rows"].append(persona_entry)

    if cache:
        save_personas_log(final_personas_dict, offset=offset, length=length, tag=tag)

    return final_personas_dict


def generate_queries_for_personas(personas_dict: Dict, topics_list: List[str]) -> Dict[str, Dict[str, str]]:
    """Generate personalized queries for personas based on topics."""

    generated_queries: Dict[str, Dict[str, str]] = {topic: {} for topic in topics_list}
    personas = personas_dict.get("rows", [])

    query_template = (
        "Here's a bit about me.\n\n{persona_details}\n\n"
        "I'm interested in learning about the following topic: {topic}. For this topic, could you do the following:\n\n"
        "Could you give me five facts about this topic, as well as the source you got them from? The sources shouldn't be links, just org name and article title. Present them as:\n"
        "Section: TOPIC NAME\nSection: Facts\n"
        "Fact 1\nFact 2\nFact 3\nFact 4\nFact 5\n\n"
        "Section: Sources\n"
        "Source 1 for Fact 1\nSource 2 for Fact 2\nSource 3 for Fact 3\nSource 4 for Fact 4\nSource 5 for Fact 5\n\n"
        "Even if there's duplicates, just write out all the sources.\n\n"
        "Then, at the end, make a final section summarizing the current situation.\nSection: Final Verdict\n\n"
        "Don't include any introduction or conclusion - all I want is the facts, sources, and final verdict exactly as mentioned above.\n"
    )

    for entry in personas:
        if "row" in entry and "persona" in entry["row"] and "professional_persona" in entry["row"]:
            row_data = entry["row"]
            sex = row_data.get("sex", "person")
            persona_details = (
                f"I am a {row_data['age']}-year-old {sex}. "
                f"{row_data['persona']} {row_data['professional_persona']}"
            )
            persona_uuid = row_data["uuid"]

            for topic in topics_list:
                query = query_template.format(persona_details=persona_details, topic=topic)
                generated_queries[topic][persona_uuid] = query

    return generated_queries


