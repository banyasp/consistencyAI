"""Embedding and similarity computations."""

from __future__ import annotations

from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from datetime import datetime
from pathlib import Path
import warnings

# Note: sentence_transformers import is deferred to avoid breaking package
# if it's not installed. Import happens in _get_sbert_model() function.
_SBERT_MODEL_CACHE: Optional[object] = None


def _get_sbert_model():
    """Get the SBERT model with caching and offline fallback."""
    global _SBERT_MODEL_CACHE
    
    if _SBERT_MODEL_CACHE is not None:
        return _SBERT_MODEL_CACHE
    
    # Check if user wants to force TF-IDF fallback
    if os.getenv("DUPLICITY_FORCE_TFIDF", "false").lower() == "true":
        print("Using TF-IDF similarity (forced via DUPLICITY_FORCE_TFIDF env var)")
        return None
    
    try:
        # Import here to avoid breaking package if not installed
        from sentence_transformers import SentenceTransformer
        
        # Try to load from cache first
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(cache_dir):
            print("Loading sentence-transformers model from cache...")
            _SBERT_MODEL_CACHE = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
        else:
            print("Downloading sentence-transformers model...")
            _SBERT_MODEL_CACHE = SentenceTransformer("all-MiniLM-L6-v2")
        
        return _SBERT_MODEL_CACHE
    
    except ImportError:
        warnings.warn("sentence-transformers not installed. Falling back to TF-IDF similarity")
        warnings.warn("Install with: pip install sentence-transformers")
        return None
    except Exception as e:
        warnings.warn(f"Failed to load sentence-transformers model: {e}")
        warnings.warn("Falling back to simple TF-IDF similarity")
        return None


def compute_persona_similarity(
    results: Dict[str, Dict[str, Dict[str, str]]], model_name: str, topic_name: str
) -> Tuple[np.ndarray, pd.DataFrame, List[int], np.ndarray]:
    """Compute embeddings and cosine similarity for a model/topic."""

    topic_responses = results[model_name][topic_name]
    sorted_personas = dict(sorted(topic_responses.items()))
    persona_ids = list(sorted_personas.keys())
    responses = list(sorted_personas.values())

    sbert_model = _get_sbert_model()
    
    if sbert_model is not None:
        # Use SBERT embeddings
        embeddings = sbert_model.encode(responses, show_progress_bar=False)
        similarity_matrix = cosine_similarity(embeddings)
    else:
        # Fallback to simple TF-IDF similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as tfidf_cosine
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(responses)
        similarity_matrix = tfidf_cosine(tfidf_matrix)
        embeddings = tfidf_matrix.toarray()  # Convert to numpy array for consistency
    
    similarity_df = pd.DataFrame(similarity_matrix, index=persona_ids, columns=persona_ids)

    return similarity_matrix, similarity_df, persona_ids, embeddings


def compute_all_similarity_matrices(results: Dict[str, Dict[str, Dict[str, str]]]):
    """Compute similarity matrices/DFs for every model/topic combination."""

    all_similarity_matrices: Dict[str, Dict[str, np.ndarray]] = {}
    all_similarity_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}

    for model_name, topics in results.items():
        all_similarity_matrices[model_name] = {}
        all_similarity_dfs[model_name] = {}
        for topic_name in topics:
            similarity_matrix, similarity_df, _persona_ids, _emb = compute_persona_similarity(
                results, model_name, topic_name
            )
            all_similarity_matrices[model_name][topic_name] = similarity_matrix
            all_similarity_dfs[model_name][topic_name] = similarity_df

    return all_similarity_matrices, all_similarity_dfs


def supercompute_similarities(results: Dict[str, Dict[str, Dict[str, str]]]):
    """Return similarity matrices/dfs, sorted persona IDs, and embeddings for all combinations."""

    all_similarity_matrices: Dict[str, Dict[str, np.ndarray]] = {}
    all_similarity_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}
    all_sorted_personas: Dict[str, Dict[str, List[int]]] = {}
    all_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

    # Calculate total combinations for progress tracking
    total_combinations = sum(len(topics) for topics in results.values())
    
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_combinations, desc="Computing similarities", unit="topic")
    except ImportError:
        pbar = None
        print(f"Processing {total_combinations} model/topic combinations...")

    for model_name, topics in results.items():
        all_similarity_matrices[model_name] = {}
        all_similarity_dfs[model_name] = {}
        all_sorted_personas[model_name] = {}
        all_embeddings[model_name] = {}

        for topic_name in topics:
            similarity_matrix, similarity_df, persona_ids, embeddings = compute_persona_similarity(
                results, model_name, topic_name
            )
            
            all_similarity_matrices[model_name][topic_name] = similarity_matrix
            all_similarity_dfs[model_name][topic_name] = similarity_df
            all_sorted_personas[model_name][topic_name] = persona_ids
            all_embeddings[model_name][topic_name] = embeddings
            
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"current": f"{model_name}/{topic_name}"})

    if pbar:
        pbar.close()
    
    return all_similarity_matrices, all_similarity_dfs, all_sorted_personas, all_embeddings


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


def save_similarity_results(
    all_similarity_matrices: Dict[str, Dict[str, np.ndarray]],
    all_similarity_dfs: Dict[str, Dict[str, pd.DataFrame]],
    all_sorted_personas: Dict[str, Dict[str, List[int]]],
    all_embeddings: Dict[str, Dict[str, np.ndarray]],
    tag: Optional[str] = None,
    subdir: str = ""
) -> str:
    """Save the full similarity computation bundle to a pickle file in logs/ and return the path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    filename = f"similarities_{timestamp}{tag_part}.pkl"
    path = os.path.join(_logs_dir(subdir), filename)
    bundle = {
        "all_similarity_matrices": all_similarity_matrices,
        "all_similarity_dfs": all_similarity_dfs,
        "all_sorted_personas": all_sorted_personas,
        "all_embeddings": all_embeddings,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return path


def list_cached_similarity_results(subdir: str = "") -> List[str]:
    logs_path = Path(_logs_dir(subdir))
    files = sorted(logs_path.glob("similarities_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


def load_similarity_results(filepath: str):
    with open(filepath, "rb") as f:
        bundle = pickle.load(f)
    return (
        bundle["all_similarity_matrices"],
        bundle["all_similarity_dfs"],
        bundle["all_sorted_personas"],
        bundle["all_embeddings"],
    )


def load_latest_similarity_results(subdir: str = ""):
    files = list_cached_similarity_results(subdir)
    if not files:
        return None
    return load_similarity_results(files[0])


def compute_avg_similarity(similarity_matrix: Union[List[List[float]], np.ndarray]) -> float:
    """Return average non-diagonal similarity for a square matrix."""

    matrix = np.array(similarity_matrix, dtype=float)
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square."
    np.fill_diagonal(matrix, np.nan)
    return float(np.nanmean(matrix))


def collect_avg_scores_by_model(all_similarity_matrices: Dict[str, Dict[str, np.ndarray]]):
    """Collect average similarity per model per topic."""

    avg_scores: Dict[str, Dict[str, float]] = {}
    for model, spheres in all_similarity_matrices.items():
        avg_scores[model] = {}
        for sphere, matrix in spheres.items():
            avg_scores[model][sphere] = compute_avg_similarity(matrix)
    return avg_scores


