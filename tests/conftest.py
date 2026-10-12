import pytest


@pytest.fixture(autouse=True)
def neutral_tide(monkeypatch):
    """测试里天下大势固定为 0（不偏分也不偏合），各概率乘子都是 1；要测大势的测试自己再改。"""
    from engine.systems import tide

    monkeypatch.setattr(tide, "at", lambda seed, turn: 0.0)
