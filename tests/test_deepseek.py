import pytest

from app.llm.deepseek import complete_chat


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.choices = [_FakeChoice(content, finish_reason)]


class _FakeCompletions:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        snapshot = dict(kwargs)
        if isinstance(kwargs.get("messages"), list):
            snapshot["messages"] = [dict(message) for message in kwargs["messages"]]
        self.calls.append(snapshot)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = type("_FakeChat", (), {"completions": completions})()


@pytest.mark.asyncio
async def test_complete_chat_continues_when_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = _FakeCompletions(
        [
            _FakeResponse("第一段，", "length"),
            _FakeResponse("第二段。", "stop"),
        ]
    )
    monkeypatch.setattr("app.llm.deepseek.get_deepseek_client", lambda: _FakeClient(completions))

    answer = await complete_chat("sys", "user", max_tokens=100)

    assert answer == "第一段，第二段。"
    assert len(completions.calls) == 2
    second_messages = completions.calls[1]["messages"]
    assert isinstance(second_messages, list)
    assert second_messages[-1]["role"] == "user"
    assert "Continue from exactly where you stopped" in second_messages[-1]["content"]


@pytest.mark.asyncio
async def test_complete_chat_returns_single_response_without_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = _FakeCompletions([_FakeResponse("完整回答。", "stop")])
    monkeypatch.setattr("app.llm.deepseek.get_deepseek_client", lambda: _FakeClient(completions))

    answer = await complete_chat("sys", "user")

    assert answer == "完整回答。"
    assert len(completions.calls) == 1
