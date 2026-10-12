"""OpenAI 兼容的大模型客户端（默认 DeepSeek）。只用标准库，Windows 版也能直接用。

密钥不进代码仓库：先看环境变量（如 DEEPSEEK_API_KEY），再看本机配置文件
（环境变量 SEKAI_SECRETS 指定的路径，默认 ~/.config/sekai/secrets.json，形如 {"deepseek_api_key": "sk-…"}）。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "env": "DEEPSEEK_API_KEY",
        "secret": "deepseek_api_key",
    },
}


class LLMError(RuntimeError):
    pass


def _secrets() -> dict:
    path = Path(
        os.environ.get("SEKAI_SECRETS") or Path.home() / ".config" / "sekai" / "secrets.json"
    )
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def provider(name: str) -> dict:
    cfg = PROVIDERS.get(name)
    if cfg is None:
        raise LLMError(f"没有这个大模型服务：{name}")
    return cfg


def api_key(name: str) -> str | None:
    cfg = provider(name)
    return os.environ.get(cfg["env"]) or _secrets().get(cfg["secret"])


def available(name: str = "deepseek") -> bool:
    try:
        return bool(api_key(name))
    except LLMError:
        return False


def chat_json(
    name: str,
    model: str | None,
    system: str,
    user: str,
    *,
    timeout: float = 120,
    max_tokens: int = 2000,
    temperature: float = 0.6,
    retries: int = 1,
) -> tuple[dict, dict]:
    """发一轮对话，要求模型只回 JSON。返回（解析后的 JSON，用量信息）。"""
    cfg = provider(name)
    key = api_key(name)
    if not key:
        raise LLMError(
            f"没有配置 {name} 的 API key（环境变量 {cfg['env']} 或 ~/.config/sekai/secrets.json）"
        )
    body = {
        "model": model or cfg["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    last = None
    for attempt in range(retries + 1):
        started = time.time()
        try:
            raw = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            text = raw["choices"][0]["message"]["content"]
            data = json.loads(text)
            usage = dict(raw.get("usage") or {})
            usage["seconds"] = round(time.time() - started, 1)
            usage["model"] = raw.get("model")
            return data, usage
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}：{e.read()[:300].decode('utf-8', 'replace')}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = f"连接失败：{e}"
        except (KeyError, IndexError, ValueError) as e:
            last = f"回答格式不对：{e}"
        if attempt < retries:
            time.sleep(1.5)
    raise LLMError(last or "大模型调用失败")
