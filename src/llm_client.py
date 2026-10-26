import os, time, logging
from openai import OpenAI
import config  # noqa: F401  # Load the project-root .env before reading keys.
logger = logging.getLogger(__name__)
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
_client = None
def get_client():
    global _client
    if _client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key: raise EnvironmentError("请设置 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_URL, timeout=60, max_retries=0)
    return _client
def llm_chat(system, user, *, temperature=0.0, max_tokens=1024, stop=None, retries=3,
             response_format=None, model=None):
    for attempt in range(retries):
        try:
            extra = {"response_format": response_format} if response_format else {}
            resp = get_client().chat.completions.create(
                model=model or DEEPSEEK_MODEL,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=temperature, max_tokens=max_tokens, stop=stop, **extra)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries-1: raise
            time.sleep(2**attempt); logger.warning("LLM retry %s: %s", attempt+1, type(e).__name__)
