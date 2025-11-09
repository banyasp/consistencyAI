"""High-level query orchestration helpers (sync and async)."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
from pathlib import Path

from .llm_tool import (
    LLMComparisonTool,
    get_model_name_from_spec,
    get_provider_from_model,
)


def _logs_dir(subdir: str = "") -> str:
    """Return absolute path to the project's logs directory.

    Args:
        subdir: Subdirectory within logs (e.g., "main", "control")
    """
    project_root = Path(__file__).resolve().parent.parent
    if subdir:
        logs_path = project_root / "logs" / subdir
    else:
        logs_path = project_root / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    return str(logs_path)


def save_results_log(results: Dict[str, Dict[str, Dict[str, str]]], tag: Optional[str] = None, subdir: str = "") -> str:
    """Persist model results to logs as JSON and return the saved file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    filename = f"results_{timestamp}{tag_part}.json"
    path = os.path.join(_logs_dir(subdir), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path


def list_cached_results(subdir: str = "") -> List[str]:
    """List cached results JSON files (sorted newest first)."""
    logs_path = Path(_logs_dir(subdir))
    files = sorted(logs_path.glob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


def load_results_from_log(filepath: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load results JSON from a specific log file path."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_latest_results(subdir: str = "") -> Optional[Dict[str, Dict[str, Dict[str, str]]]]:
    """Load the most recent cached results if present; otherwise return None."""
    files = list_cached_results(subdir)
    if not files:
        return None
    return load_results_from_log(files[0])


async def query_llm(nested_queries: Dict[str, Dict[str, str]], list_of_models: List[str], all_open_router: bool = False) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Query multiple LLMs with the given queries and return responses organized by model/topic/persona."""

    result_dict: Dict[str, Dict[str, Dict[str, str]]] = {}
    async with LLMComparisonTool() as tool:
        for model_spec in list_of_models:
            model_responses: Dict[str, Dict[str, str]] = {}

            # Access provider/model name if needed
            _provider = get_provider_from_model(model_spec)
            _model_name = get_model_name_from_spec(model_spec)

            for topic, id_queries in nested_queries.items():
                model_responses[topic] = {}
                for id_key, prompt in id_queries.items():
                    try:
                        responses = await tool.run_comparison(prompt, [model_spec], all_open_router)
                        if responses:
                            response = responses[0]
                            if response.error:
                                model_responses[topic][id_key] = f"Error: {response.error}"
                            else:
                                model_responses[topic][id_key] = response.response_text
                        else:
                            model_responses[topic][id_key] = "No response received"
                    except Exception as e:  # noqa: BLE001
                        model_responses[topic][id_key] = f"Error: {str(e)}"

            result_dict[model_spec] = model_responses

    return result_dict


def query_llm_sync(nested_queries: Dict[str, Dict[str, str]], list_of_models: List[str], all_open_router: bool = False) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Synchronous wrapper for `query_llm` that also works in notebooks."""

    try:
        return asyncio.run(query_llm(nested_queries, list_of_models, all_open_router))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(query_llm(nested_queries, list_of_models, all_open_router))
        raise


async def process_single_query(
    tool: LLMComparisonTool, model_spec: str, topic: str, id_key: str, prompt: str, all_open_router: bool = False
) -> Tuple[str, str, str, str]:
    """Process a single query and return (model_spec, topic, id, response_text)."""

    responses = await tool.run_comparison(prompt, [model_spec], all_open_router)
    if responses:
        response = responses[0]
        if response.error:
            return model_spec, topic, id_key, f"Error: {response.error}"
        return model_spec, topic, id_key, response.response_text
    return model_spec, topic, id_key, "No response received"


async def query_llm_multithreaded(
    nested_queries: Dict[str, Dict[str, str]], list_of_models: List[str], all_open_router: bool = False
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Parallelize all API calls across models/topics/personas using asyncio.gather."""

    result_dict: Dict[str, Dict[str, Dict[str, str]]] = {
        model_spec: {topic: {} for topic in nested_queries.keys()} for model_spec in list_of_models
    }

    tasks = []
    async with LLMComparisonTool() as tool:
        for model_spec in list_of_models:
            for topic, id_queries in nested_queries.items():
                for id_key, prompt in id_queries.items():
                    tasks.append(process_single_query(tool, model_spec, topic, id_key, prompt, all_open_router))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for item in responses:
            if isinstance(item, Exception):
                # Generic fallback; since we do not have context here, skip.
                continue
            model_spec, topic, id_key, response_text = item
            result_dict[model_spec][topic][id_key] = response_text

    return result_dict


def query_llm_multithreaded_sync(
    nested_queries: Dict[str, Dict[str, str]], list_of_models: List[str], all_open_router: bool = False
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Synchronous wrapper for `query_llm_multithreaded`. Safe for notebooks."""

    try:
        return asyncio.run(query_llm_multithreaded(nested_queries, list_of_models, all_open_router))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(query_llm_multithreaded(nested_queries, list_of_models, all_open_router))
        raise


