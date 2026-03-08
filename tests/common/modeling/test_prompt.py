from pathlib import Path
from typing import Any, cast

import pytest

from mafc.common.modeling.prompt import (
    Prompt,
    compose_prompt,
    fill_placeholders,
    read_md_file,
)


def test_read_md_and_compose_prompt(tmp_path: Path) -> None:
    file_path = tmp_path / "template.md"
    file_path.write_text("Hello {{X}}", encoding="utf-8")

    assert read_md_file(file_path) == "Hello {{X}}"
    assert compose_prompt(file_path, {"{{X}}": "World"}) == "Hello World"

    with pytest.raises(FileNotFoundError):
        read_md_file(tmp_path / "missing.md")


def test_fill_placeholders_validation() -> None:
    assert fill_placeholders("A {{X}} B", {"{{X}}": "C"}) == "A C B"
    assert fill_placeholders("A {{X}}", {"{{X}}": None}) == "A "
    assert fill_placeholders("abc", cast(dict[str, Any], None)) == "abc"

    with pytest.raises(ValueError):
        fill_placeholders("abc", {"{{X}}": "C"})


def test_prompt_text_template_and_len(tmp_path: Path) -> None:
    p = Prompt(text="abc", name="n")
    assert len(p) == 3
    assert p.extract("resp") == "resp"
    assert p.name == "n"

    file_path = tmp_path / "template.md"
    file_path.write_text("T {{A}}", encoding="utf-8")
    p2 = Prompt(template_file_path=str(file_path), placeholder_targets={"{{A}}": "X"})
    assert str(p2) == "T X"

    with pytest.raises(AssertionError):
        Prompt(text="x", template_file_path=str(file_path))


def test_with_videos_as_frames(monkeypatch) -> None:
    class FakeVideo:
        def sample_frames(self, n_frames: int, format: str = "jpeg"):
            assert n_frames == 2
            assert format == "jpeg"
            return [b"a", b"b"]

    class FakeImage:
        def __init__(self, binary_data: bytes):
            self.binary_data = binary_data

    import ezmm.common.items as items

    monkeypatch.setattr(items, "Video", FakeVideo)
    monkeypatch.setattr(items, "Image", FakeImage)

    prompt = object.__new__(Prompt)
    cast(Any, prompt).data = [FakeVideo(), "txt"]
    prompt.name = "p"

    out = prompt.with_videos_as_frames(n_frames=2)
    assert out is not prompt
    assert len(out.data) == 3
    assert isinstance(out.data[0], FakeImage)
    assert out.data[0].binary_data == b"a"
    assert isinstance(out.data[1], FakeImage)
    assert out.data[1].binary_data == b"b"
    assert out.data[2] == "txt"
