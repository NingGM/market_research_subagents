import os, json, urllib.request, logging
if __package__ == "src":
    from . import config  # noqa: F401  # Load the project-root .env before reading keys.
    from .search_processing import process_search_response
else:
    import config  # noqa: F401  # Load the project-root .env before reading keys.
    from search_processing import process_search_response
logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


def tavily_search_raw(query: str, max_results: int = 5) -> dict:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"error": "未设置 TAVILY_API_KEY"}
    payload = {
        "api_key": key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
    }
    try:
        req = urllib.request.Request(
            TAVILY_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # Preserve the response for controlled cleaning/replay experiments.
        # Default content is a snippet; HTML/raw_content is not guaranteed.
        return data
    except Exception as e:
        logger.warning("Tavily search failed: %s", type(e).__name__)
        return {"error": type(e).__name__}


def tavily_search(query: str, max_results: int = 5) -> dict:
    return process_search_response(tavily_search_raw(query, max_results), query)


def format_search_result(r: dict) -> str:
    if "error" in r:
        return f"搜索失败: {r['error']}"
    parts = []
    if r.get("answer"):
        parts.append(f"摘要: {r['answer']}")
    for i, res in enumerate(r.get("results", []), 1):
        sid = res.get("source_id", str(i))
        parts.append(f"[{sid}] {res['title']}\nURL: {res.get('url', '')}\n{res['content']}")
    return "\n".join(parts) if parts else "无结果"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = tavily_search("中国新能源汽车2024年销量")
    print(format_search_result(r)[:400])
