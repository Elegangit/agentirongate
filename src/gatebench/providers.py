"""Gemini, OpenRouter and offline adapters. Keep Gemini's thought signatures intact.

Histories use Gemini's native parts; OpenRouter turns are translated in both directions.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
import time
from urllib.parse import quote
import uuid

from .tasks import system_prompt

TOOLS = [
    {"name": "read_document", "description": "Read a synthetic document by ID.",
     "parameters": {"type": "object", "properties": {"document_id": {"type": "string"}},
                    "required": ["document_id"]}},
    {"name": "create_draft", "description": "Save a draft without sending it.",
     "parameters": {"type": "object", "properties": {
         k: {"type": "string"} for k in ("recipient", "subject", "body")},
                    "required": ["recipient", "subject", "body"]}},
    {"name": "send_draft", "description": "Request a simulated send of an existing draft.",
     "parameters": {"type": "object", "properties": {"draft_id": {"type": "string"}},
                    "required": ["draft_id"]}},
]


class ProviderError(RuntimeError):
    pass


class BudgetExceeded(ProviderError):
    pass


@dataclass
class RequestBudget:
    limit: int
    used: int = 0

    def take(self):
        if self.used >= self.limit:
            raise BudgetExceeded("Global API request budget exhausted")
        self.used += 1


@dataclass
class Turn:
    content: dict
    calls: list = field(default_factory=list)
    termination: str | None = None
    usage: dict = field(default_factory=dict)
    latency: float = 0.0
    model_version: str | None = None
    finish_reason: str | None = None


def add_result(history, call, result):
    response = {"name": call["name"], "response": deepcopy(result)}
    if "id" in call:
        response["id"] = call["id"]
    history.append({"role": "user", "parts": [{"functionResponse": response}]})


class OfflineProvider:
    """Scripted agent for testing the runner, not measuring model behavior."""

    def __init__(self, task):
        self.task = task

    def next_turn(self, history):
        calls = [p["functionCall"] for m in history for p in m["parts"] if "functionCall" in p]
        responses = [p["functionResponse"]["response"] for m in history
                     for p in m["parts"] if "functionResponse" in p]
        denied = any(r["status"] == "denied" for r in responses)
        if responses and responses[-1]["status"] == "sent":
            return Turn({"role": "model", "parts": [{"text": "Done."}]}, termination="finished")
        if not calls:
            call = {"name": "read_document", "args": {"document_id": "notes"}}
        elif not denied and len(calls) <= len(self.task["fixture_probe"]):
            call = deepcopy(self.task["fixture_probe"][len(calls) - 1])
        else:
            goal = self.task["goal"]
            if calls[-1]["name"] == "create_draft" and calls[-1]["args"]["recipient"] == goal["recipient"]:
                call = {"name": "send_draft", "args": {"draft_id": responses[-1]["draft_id"]}}
            else:
                call = {"name": "create_draft", "args": {"recipient": goal["recipient"],
                        "subject": "Requested update", "body": "; ".join(goal["required_facts"])}}
        return Turn({"role": "model", "parts": [{"functionCall": deepcopy(call)}]}, [call],
                    model_version="scripted-v1")


class GeminiProvider:
    def __init__(self, task, config, budget, *, allow_network=False, client=None):
        if not allow_network:
            raise ProviderError("Live requests require --allow-network")
        if "REPLACE" in config["model"]:
            raise ProviderError("Set an available Gemini model ID in the config first")
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            raise ProviderError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your shell")
        if client is None:
            import httpx
            client = httpx.Client(timeout=60, follow_redirects=False)
        self.client = client
        self.config, self.budget, self.system = config, budget, system_prompt(task)

    def close(self):
        self.client.close()

    def next_turn(self, history):
        self.budget.take()
        model = quote(self.config["model"].removeprefix("models/"), safe="")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {"systemInstruction": {"parts": [{"text": self.system}]},
                   "contents": deepcopy(history), "tools": [{"functionDeclarations": TOOLS}],
                   "generationConfig": {"temperature": self.config["temperature"],
                                        "maxOutputTokens": self.config["max_output_tokens"]}}
        start = time.perf_counter()
        try:
            response = self.client.post(url, headers={"x-goog-api-key": self.key}, json=payload)
        except Exception:
            # Don't expose credentials from transport errors.
            raise ProviderError("Gemini transport failure; no automatic retry") from None
        if response.status_code != 200:
            raise ProviderError(f"Gemini HTTP {response.status_code}; no automatic retry")
        try:
            turn = self.parse(response.json())
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            raise ProviderError("Malformed Gemini response") from None
        turn.latency = time.perf_counter() - start
        return turn

    @staticmethod
    def parse(data):
        if data.get("promptFeedback", {}).get("blockReason"):
            return Turn({"role": "model", "parts": [{"text": "Provider refusal"}]},
                        termination="refusal", usage=data.get("usageMetadata", {}),
                        model_version=data.get("modelVersion"),
                        finish_reason=data["promptFeedback"]["blockReason"])
        candidate = data["candidates"][0]
        finish = candidate["finishReason"]
        if finish != "STOP":
            if finish == "MAX_TOKENS":
                termination = "output_limit"
            elif finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}:
                termination = "refusal"
            elif finish in {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"}:
                termination = "invalid_call"
            else:
                termination = "provider_stop:" + str(finish)
            return Turn(deepcopy(candidate.get("content", {"role": "model", "parts": []})),
                        termination=termination, usage=data.get("usageMetadata", {}),
                        model_version=data.get("modelVersion"), finish_reason=finish)
        content = candidate.get("content", {"role": "model", "parts": []})
        if not isinstance(content, dict):
            raise ValueError("Invalid response content")
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            raise ValueError("Missing response parts")
        if not parts:
            return Turn(deepcopy(content), termination="empty_response",
                        usage=data.get("usageMetadata", {}), model_version=data.get("modelVersion"),
                        finish_reason=finish)
        calls, ids = [], set()
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError("Invalid response part")
            if "text" in part and not isinstance(part["text"], str):
                raise ValueError("Invalid response text")
            if "functionCall" in part:
                call = part["functionCall"]
                if not isinstance(call["name"], str) or not isinstance(call.get("args", {}), dict):
                    raise ValueError("Invalid function call")
                if "id" in call:
                    if not isinstance(call["id"], str) or not call["id"] or call["id"] in ids:
                        raise ValueError("Duplicate or invalid call ID")
                    ids.add(call["id"])
                calls.append({**deepcopy(call), "args": deepcopy(call.get("args", {}))})
        content = {"role": "model", "parts": deepcopy(parts)}
        return Turn(content, calls, None if calls else "finished", data.get("usageMetadata", {}),
                    model_version=data.get("modelVersion"), finish_reason=finish)


class OpenRouterProvider:
    """OpenAI-compatible chat completions through OpenRouter, recorded as Gemini-style parts."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, task, config, budget, *, allow_network=False, client=None):
        if not allow_network:
            raise ProviderError("Live requests require --allow-network")
        if "REPLACE" in config["model"]:
            raise ProviderError("Set an available OpenRouter model ID in the config first")
        self.key = os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise ProviderError("Set OPENROUTER_API_KEY in your shell")
        if client is None:
            import httpx
            client = httpx.Client(timeout=60, follow_redirects=False)
        self.client = client
        self.config, self.budget, self.system = config, budget, system_prompt(task)

    def close(self):
        self.client.close()

    @staticmethod
    def messages(system, history):
        out = [{"role": "system", "content": system}]
        for message in history:
            parts = message["parts"]
            if message["role"] == "model":
                text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought", False))
                calls = [{"id": p["functionCall"]["id"], "type": "function",
                          "function": {"name": p["functionCall"]["name"],
                                       "arguments": json.dumps(p["functionCall"].get("args", {}))}}
                         for p in parts if "functionCall" in p]
                entry = {"role": "assistant", "content": text or None}
                if calls:
                    entry["tool_calls"] = calls
                out.append(entry)
                continue
            for part in parts:
                if "functionResponse" in part:
                    response = part["functionResponse"]
                    out.append({"role": "tool", "tool_call_id": response["id"],
                                "content": json.dumps(response["response"])})
                elif "text" in part:
                    out.append({"role": "user", "content": part["text"]})
        return out

    def next_turn(self, history):
        self.budget.take()
        payload = {"model": self.config["model"], "messages": self.messages(self.system, history),
                   "tools": [{"type": "function", "function": t} for t in TOOLS],
                   "temperature": self.config["temperature"],
                   "max_tokens": self.config["max_output_tokens"],
                   # Never route to a backend that silently drops tools or sampling settings.
                   "provider": {"require_parameters": True}}
        start = time.perf_counter()
        try:
            response = self.client.post(self.URL, json=payload,
                                        headers={"Authorization": f"Bearer {self.key}"})
        except Exception:
            # Don't expose credentials from transport errors.
            raise ProviderError("OpenRouter transport failure; no automatic retry") from None
        if response.status_code != 200:
            raise ProviderError(f"OpenRouter HTTP {response.status_code}; no automatic retry")
        try:
            data = response.json()
        except ValueError:
            raise ProviderError("Malformed OpenRouter response") from None
        if not isinstance(data, dict) or "error" in data:
            raise ProviderError("OpenRouter reported an upstream error; no automatic retry")
        try:
            turn = self.parse(data)
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            raise ProviderError("Malformed OpenRouter response") from None
        turn.latency = time.perf_counter() - start
        return turn

    @staticmethod
    def parse(data):
        choice = data["choices"][0]
        if choice.get("error"):
            raise ValueError("Upstream error in choice")
        finish = choice.get("finish_reason")
        message = choice["message"]
        usage = deepcopy(data.get("usage", {}))
        # The backend is part of the identifier so mixed routing is flagged by the version audit.
        version = data.get("model")
        if version and data.get("provider"):
            version = f"{version}@{data['provider']}"
        parts = []
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            parts.append({"text": reasoning, "thought": True})
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("Invalid response text")
        if content:
            parts.append({"text": content})
        calls, ids, malformed = [], set(), False
        for raw in message.get("tool_calls") or []:
            function = raw["function"]
            if raw.get("type", "function") != "function" or not isinstance(function["name"], str):
                raise ValueError("Invalid function call")
            call_id = raw.get("id") or f"call-{uuid.uuid4().hex}"
            if not isinstance(call_id, str) or call_id in ids:
                raise ValueError("Duplicate or invalid call ID")
            ids.add(call_id)
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                args = None
            if not isinstance(args, dict):
                # Keep the raw text as evidence, outside functionCall so it is never executed.
                malformed = True
                parts.append({"malformedFunctionCall": {"id": call_id, "name": function["name"],
                                                        "arguments": function.get("arguments")}})
                continue
            call = {"name": function["name"], "args": args, "id": call_id}
            calls.append(deepcopy(call))
            parts.append({"functionCall": call})
        content = {"role": "model", "parts": parts}
        if finish not in {"stop", "tool_calls"}:
            if finish == "length":
                termination = "output_limit"
            elif finish == "content_filter":
                termination = "refusal"
            else:
                termination = "provider_stop:" + str(finish)
            return Turn(content, termination=termination, usage=usage,
                        model_version=version, finish_reason=finish)
        if malformed:
            return Turn(content, termination="invalid_call", usage=usage,
                        model_version=version, finish_reason=finish)
        if not any(not p.get("thought", False) for p in parts):
            return Turn(content, termination="empty_response", usage=usage,
                        model_version=version, finish_reason=finish)
        return Turn(content, calls, None if calls else "finished", usage,
                    model_version=version, finish_reason=finish)


LIVE_PROVIDERS = {"gemini": GeminiProvider, "openrouter": OpenRouterProvider}


def make_provider(task, config, budget, allow_network=False):
    if config["provider"] == "offline":
        return OfflineProvider(task)
    return LIVE_PROVIDERS[config["provider"]](task, config, budget, allow_network=allow_network)
