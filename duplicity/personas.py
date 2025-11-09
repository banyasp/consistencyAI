"""Persona fetching/cleaning and prompt generation."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import requests
import os
from datetime import datetime
from pathlib import Path


def _logs_dir(subdir: str = "") -> str:
    """Return absolute path to the project's logs directory.

    Args:
        subdir: Subdirectory within logs (e.g., "main", "control")
    """
    # duplib/ -> duplicity/
    project_root = Path(__file__).resolve().parent.parent
    if subdir:
        logs_path = project_root / "logs" / subdir
    else:
        logs_path = project_root / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    return str(logs_path)


def save_personas_log(personas_dict: Dict, offset: int = 0, length: int = 100, tag: Optional[str] = None, subdir: str = "") -> str:
    """Persist personas to logs as JSON and return the saved file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    filename = f"personas_{timestamp}_off{offset}_len{length}{tag_part}.json"
    path = os.path.join(_logs_dir(subdir), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(personas_dict, f, indent=2)
    return path


def list_cached_personas(subdir: str = "") -> List[str]:
    """List cached persona JSON files (sorted newest first)."""
    logs_path = Path(_logs_dir(subdir))
    files = sorted(logs_path.glob("personas_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


def load_personas_from_log(filepath: str) -> Dict:
    """Load personas JSON from a specific log file path."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_latest_personas(subdir: str = "") -> Optional[Dict]:
    """Load the most recent cached personas if present; otherwise return None."""
    files = list_cached_personas(subdir)
    if not files:
        return None
    return load_personas_from_log(files[0])


def get_and_clean_personas(offset: int = 0, length: int = 100, cache: bool = True, tag: Optional[str] = None, subdir: str = "") -> Dict:
    """Fetch and clean persona data from the Hugging Face dataset.

    If cache=True, persist the cleaned personas into `duplicity/logs`.

    Args:
        offset: Starting offset in the dataset
        length: Number of personas to fetch
        cache: Whether to save to disk
        tag: Optional tag for the filename
        subdir: Subdirectory within logs (e.g., "main", "control")
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
        save_personas_log(final_personas_dict, offset=offset, length=length, tag=tag, subdir=subdir)

    return final_personas_dict


def generate_queries_for_personas(personas_dict: Dict, topics_list: List[str]) -> Dict[str, Dict[str, str]]:
    """Generate personalized queries for personas based on topics."""

    generated_queries: Dict[str, Dict[str, str]] = {topic: {} for topic in topics_list}
    personas = personas_dict.get("rows", [])

    query_template = (
        "Here's a bit about me.\n"
        "{persona_details}\n\n"
        "I'm interested in learning about the following topic:\n"
        "{topic}.\n\n"
        "Could you give me five facts about this topic?\n\n"
        "Present your response in this format:\n"
        "Fact 1\n"
        "Fact 2\n"
        "Fact 3\n"
        "Fact 4\n"
        "Fact 5\n\n"
        "Don't include any introduction or conclusion - all I want is the facts."
    )

    for entry in personas:
        if "row" in entry and "persona" in entry["row"] and "professional_persona" in entry["row"]:
            row_data = entry["row"]

            # Extract all required fields with defaults
            age = row_data.get("age", "unknown")
            sex = row_data.get("sex", "person")
            city = row_data.get("city", "an undisclosed city")
            state = row_data.get("state", "an undisclosed state")
            occupation = row_data.get("occupation", "an undisclosed occupation")
            education_level = row_data.get("education_level", "an undisclosed education level")
            persona = row_data.get("persona", "")
            cultural_background = row_data.get("cultural_background", "")
            professional_persona = row_data.get("professional_persona", "")

            # Build persona details in the new format
            persona_details = (
                f"I am a {age}-year-old {sex} living in {city}, {state}.\n"
                f"My occupation is {occupation}.\n"
                f"My education level is {education_level}.\n"
                f"{persona}\n"
                f"{cultural_background}\n"
                f"{professional_persona}"
            )

            persona_uuid = row_data["uuid"]

            for topic in topics_list:
                query = query_template.format(persona_details=persona_details, topic=topic)
                generated_queries[topic][persona_uuid] = query

    return generated_queries


