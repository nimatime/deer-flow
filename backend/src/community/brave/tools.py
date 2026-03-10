import json
import os

import httpx
from langchain.tools import tool

from src.config import get_app_config


def _get_brave_api_key() -> str:
    """Get the Brave Search API key from config or environment."""
    config = get_app_config().get_tool_config("web_search")
    api_key = None
    if config is not None and "api_key" in config.model_extra:
        api_key = config.model_extra.get("api_key")
    if not api_key:
        api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError(
            "Brave Search API key not configured. "
            "Set BRAVE_SEARCH_API_KEY environment variable or add api_key to tool config."
        )
    return api_key


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str) -> str:
    """Search the web using Brave Search.

    Args:
        query: The query to search for.
    """
    config = get_app_config().get_tool_config("web_search")
    max_results = 5
    if config is not None and "max_results" in config.model_extra:
        max_results = config.model_extra.get("max_results")

    api_key = _get_brave_api_key()

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    params = {
        "q": query,
        "count": str(max_results),
    }

    with httpx.Client(timeout=10) as client:
        response = client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        data = response.json()

    results = data.get("web", {}).get("results", [])
    normalized_results = [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("description", ""),
        }
        for result in results
    ]
    json_results = json.dumps(normalized_results, indent=2, ensure_ascii=False)
    return json_results
