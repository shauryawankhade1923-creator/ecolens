"""
Encyclopedia & search utilities.

Search order:
  1. Local curated database (data/species_db.json) — fast, offline, trusted.
  2. Wikipedia summary fallback — for anything not yet in the local DB.

Expand data/species_db.json over time with your own verified entries;
the local DB should always be preferred for accuracy over the Wikipedia
fallback, which is convenient but not curated.
"""

import json
import os

import streamlit as st

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "species_db.json")


@st.cache_data
def load_local_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_db(db):
    """Turn the categorized DB into a flat list of searchable records."""
    flat = []
    for category, entries in db.items():
        for entry in entries:
            record = dict(entry)
            record["_category"] = category
            flat.append(record)
    return flat


def search_local(query, db):
    query_lower = query.strip().lower()
    if not query_lower:
        return []
    results = []
    for record in _flatten_db(db):
        haystack = " ".join(str(v) for v in record.values()).lower()
        if query_lower in haystack:
            results.append(record)
    return results


def search_wikipedia(query):
    """
    Fallback lookup for anything not in the local DB.
    Requires the `wikipedia` package (pip install wikipedia).
    Returns None if not found or the package/network isn't available.
    """
    try:
        import wikipedia
    except ImportError:
        return {"error": "The 'wikipedia' package isn't installed. Run: pip install wikipedia"}

    try:
        summary = wikipedia.summary(query, sentences=4, auto_suggest=True)
        page = wikipedia.page(query, auto_suggest=True)
        return {
            "name": page.title,
            "summary": summary,
            "url": page.url,
            "_category": "wikipedia",
        }
    except wikipedia.exceptions.DisambiguationError as e:
        return {"error": f"That's ambiguous — did you mean: {', '.join(e.options[:5])}?"}
    except wikipedia.exceptions.PageError:
        return None
    except Exception as e:
        return {"error": f"Lookup failed: {e}"}


def search(query):
    """
    Unified search: checks local DB first, falls back to Wikipedia.
    Returns a dict: {"source": "local"|"wikipedia"|"none", "results": [...]}
    """
    db = load_local_db()
    local_results = search_local(query, db)
    if local_results:
        return {"source": "local", "results": local_results}

    wiki_result = search_wikipedia(query)
    if wiki_result and "error" not in wiki_result:
        return {"source": "wikipedia", "results": [wiki_result]}
    elif wiki_result and "error" in wiki_result:
        return {"source": "error", "results": [], "message": wiki_result["error"]}

    return {"source": "none", "results": []}
