from copy import deepcopy

import httpx
import pytest

from gatebench.providers import BudgetExceeded, GeminiProvider, ProviderError, RequestBudget, add_result


def response(parts=None, finish="STOP"):
    return {"candidates": [{"finishReason": finish, "content": {"parts": parts or [
        {"functionCall": {"name": "read_document", "args": {"document_id": "notes"}, "id": "a"},
         "thoughtSignature": "preserve-this"}]}}], "modelVersion": "test-version",
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}


def test_native_parts_and_ids_roundtrip():
    data = response()
    turn = GeminiProvider.parse(data)
    assert turn.content["parts"][0]["thoughtSignature"] == "preserve-this"
    history = [turn.content]
    add_result(history, turn.calls[0], {"status": "ok"})
    assert history[-1]["parts"][0]["functionResponse"]["id"] == "a"
    assert turn.usage["promptTokenCount"] == 10 and turn.model_version == "test-version"


@pytest.mark.parametrize("data", [{}, {"candidates": []},
                                   response([{"functionCall": {"name": 1, "args": {}}}]),
                                   response([{"functionCall": {"name": "x", "args": []}}])])
def test_malformed_response_rejected(data):
    with pytest.raises((ValueError, KeyError, IndexError, TypeError)):
        GeminiProvider.parse(data)


def test_duplicate_ids_rejected():
    part = response()["candidates"][0]["content"]["parts"][0]
    with pytest.raises(ValueError):
        GeminiProvider.parse(response([part, deepcopy(part)]))


@pytest.mark.parametrize("finish,expected", [("MAX_TOKENS", "output_limit"), ("SAFETY", "refusal")])
def test_incomplete_turn_never_executes(finish, expected):
    turn = GeminiProvider.parse(response(finish=finish))
    assert turn.termination == expected and not turn.calls


def test_prompt_block_is_refusal():
    assert GeminiProvider.parse({"promptFeedback": {"blockReason": "SAFETY"}}).termination == "refusal"


@pytest.mark.parametrize("finish,expected", [
    ("MALFORMED_FUNCTION_CALL", "invalid_call"), ("UNEXPECTED_TOOL_CALL", "invalid_call"),
    ("RECITATION", "provider_stop:RECITATION"), ("OTHER", "provider_stop:OTHER")])
def test_stop_classification_preserves_evidence(finish, expected):
    data = response(finish=finish)
    turn = GeminiProvider.parse(data)
    assert turn.termination == expected and not turn.calls
    assert turn.content == data["candidates"][0]["content"]
    assert turn.finish_reason == finish and turn.model_version == "test-version"


def test_empty_response_is_not_infrastructure_error():
    data = response()
    data["candidates"][0]["content"]["parts"] = []
    assert GeminiProvider.parse(data).termination == "empty_response"


def model_cfg():
    return {"model": "test-model", "temperature": 0, "max_output_tokens": 100}


def test_network_is_opt_in(task):
    with pytest.raises(ProviderError, match="allow-network"):
        GeminiProvider(task, model_cfg(), RequestBudget(1))


def test_mocked_request_and_global_cap(task, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["x-goog-api-key"] == "test-only-key"
        assert "test-only-key" not in request.content.decode()
        assert "PRIVATE-ROBOTICS" not in request.content.decode()
        return httpx.Response(200, json=response())

    client = httpx.Client(transport=httpx.MockTransport(handle))
    provider = GeminiProvider(task, model_cfg(), RequestBudget(1), allow_network=True, client=client)
    turn = provider.next_turn([{"role": "user", "parts": [{"text": task["prompt"]}]}])
    assert turn.calls
    with pytest.raises(BudgetExceeded):
        provider.next_turn([])
    assert len(requests) == 1
    provider.close()


@pytest.mark.parametrize("status", [301, 400, 401, 403, 404, 429, 500])
def test_http_failures_do_not_echo_body_or_retry(task, monkeypatch, status):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(status, text="test-only-key secret body")))
    provider = GeminiProvider(task, model_cfg(), RequestBudget(2), allow_network=True, client=client)
    with pytest.raises(ProviderError) as err:
        provider.next_turn([])
    assert "test-only-key" not in str(err.value) and "secret body" not in str(err.value)
    assert provider.budget.used == 1
    provider.close()
