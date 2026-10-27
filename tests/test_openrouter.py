import json

import httpx
import pytest

from gatebench.providers import (BudgetExceeded, OpenRouterProvider, ProviderError, RequestBudget,
                                 add_result, make_provider)


def response(message=None, finish="tool_calls"):
    return {"model": "test/model", "provider": "TestHost",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"finish_reason": finish, "message": message or {
                "role": "assistant", "content": None, "reasoning": "private reasoning",
                "tool_calls": [{"id": "a", "type": "function", "function": {
                    "name": "read_document", "arguments": '{"document_id": "notes"}'}}]}}]}


def test_tool_call_roundtrip_through_native_history():
    turn = OpenRouterProvider.parse(response())
    assert turn.calls == [{"name": "read_document", "args": {"document_id": "notes"}, "id": "a"}]
    assert turn.content["parts"][0] == {"text": "private reasoning", "thought": True}
    assert turn.model_version == "test/model@TestHost" and turn.usage["prompt_tokens"] == 10
    history = [{"role": "user", "parts": [{"text": "task"}]}, turn.content]
    add_result(history, turn.calls[0], {"status": "ok"})
    messages = OpenRouterProvider.messages("system", history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["content"] is None
    assert json.loads(messages[2]["tool_calls"][0]["function"]["arguments"]) == {"document_id": "notes"}
    assert messages[3]["tool_call_id"] == "a" and json.loads(messages[3]["content"]) == {"status": "ok"}


def test_reasoning_is_never_sent_back():
    turn = OpenRouterProvider.parse(response({"role": "assistant", "content": "Done.",
                                              "reasoning": "hidden"}, finish="stop"))
    assert turn.termination == "finished"
    assert OpenRouterProvider.messages("s", [turn.content])[1]["content"] == "Done."


def test_missing_ids_are_generated_uniquely():
    message = response()["choices"][0]["message"]
    call = dict(message["tool_calls"][0])
    del call["id"]
    message["tool_calls"] = [call, dict(call)]
    turn = OpenRouterProvider.parse(response(message))
    assert len({c["id"] for c in turn.calls}) == 2


def test_duplicate_ids_rejected():
    message = response()["choices"][0]["message"]
    message["tool_calls"] = message["tool_calls"] * 2
    with pytest.raises(ValueError):
        OpenRouterProvider.parse(response(message))


@pytest.mark.parametrize("arguments", ["{not json", "[1, 2]"])
def test_malformed_arguments_never_execute(arguments):
    message = response()["choices"][0]["message"]
    message["tool_calls"][0]["function"]["arguments"] = arguments
    turn = OpenRouterProvider.parse(response(message))
    assert turn.termination == "invalid_call" and not turn.calls
    assert not any("functionCall" in p for p in turn.content["parts"])
    assert turn.content["parts"][-1]["malformedFunctionCall"]["arguments"] == arguments


@pytest.mark.parametrize("finish,expected", [("length", "output_limit"), ("content_filter", "refusal"),
                                             ("other", "provider_stop:other")])
def test_incomplete_turn_never_executes(finish, expected):
    turn = OpenRouterProvider.parse(response(finish=finish))
    assert turn.termination == expected and not turn.calls and turn.finish_reason == finish


def test_reasoning_only_is_empty_response():
    turn = OpenRouterProvider.parse(response({"role": "assistant", "content": "",
                                              "reasoning": "thinking"}, finish="stop"))
    assert turn.termination == "empty_response"


@pytest.mark.parametrize("data", [{}, {"choices": []}, response({"content": 3}, finish="stop"),
                                   {"choices": [{"error": {"code": 502}, "message": {}}]}])
def test_malformed_response_rejected(data):
    with pytest.raises((ValueError, KeyError, IndexError, TypeError)):
        OpenRouterProvider.parse(data)


def model_cfg():
    return {"provider": "openrouter", "model": "test/model", "temperature": 0, "max_output_tokens": 100}


def test_network_and_key_are_required(task, monkeypatch):
    with pytest.raises(ProviderError, match="allow-network"):
        make_provider(task, model_cfg(), RequestBudget(1))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        make_provider(task, model_cfg(), RequestBudget(1), allow_network=True)


def test_mocked_request_and_global_cap(task, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    requests = []

    def handle(request):
        requests.append(request)
        body = request.content.decode()
        assert request.headers["authorization"] == "Bearer test-only-key"
        assert "test-only-key" not in body and "PRIVATE-ROBOTICS" not in body
        payload = json.loads(body)
        assert payload["provider"] == {"require_parameters": True}
        assert {t["function"]["name"] for t in payload["tools"]} == {
            "read_document", "create_draft", "send_draft"}
        return httpx.Response(200, json=response())

    client = httpx.Client(transport=httpx.MockTransport(handle))
    provider = OpenRouterProvider(task, model_cfg(), RequestBudget(1), allow_network=True, client=client)
    assert provider.next_turn([{"role": "user", "parts": [{"text": task["prompt"]}]}]).calls
    with pytest.raises(BudgetExceeded):
        provider.next_turn([])
    assert len(requests) == 1
    provider.close()


@pytest.mark.parametrize("status,body", [(401, "test-only-key secret body"), (429, "secret body"),
                                         (200, '{"error": {"message": "secret body"}}'),
                                         (200, "not json secret body")])
def test_failures_do_not_echo_body_or_retry(task, monkeypatch, status, body):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status, text=body)))
    provider = OpenRouterProvider(task, model_cfg(), RequestBudget(2), allow_network=True, client=client)
    with pytest.raises(ProviderError) as err:
        provider.next_turn([])
    assert "test-only-key" not in str(err.value) and "secret body" not in str(err.value)
    assert provider.budget.used == 1
    provider.close()
