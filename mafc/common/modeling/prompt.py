from pathlib import Path
from typing import Any

from ezmm import MultimodalSequence


def read_md_file(file_path: str | Path) -> str:
    """Reads and returns the contents of the specified Markdown file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"No Markdown file found at '{file_path}'.")
    with open(file_path, "r") as f:
        return f.read()


def fill_placeholders(text: str, placeholder_targets: dict[str, Any]) -> str:
    """Replaces all specified placeholders in placeholder_targets with the respective target content."""
    if placeholder_targets is None:
        return text
    for placeholder, target in placeholder_targets.items():
        if placeholder not in text:
            raise ValueError(f"Placeholder '{placeholder}' not found in prompt template:\n{text}")
        target = str(target) if target else ""
        text = text.replace(placeholder, target)
    return text


def compose_prompt(template_file_path: str | Path, placeholder_targets: dict) -> str:
    """Turns a template prompt into a ready-to-send prompt string."""
    template = read_md_file(template_file_path)
    return fill_placeholders(template, placeholder_targets).strip(" \n")


class Prompt(MultimodalSequence):
    template_file_path: str | None = None
    name: str | None = None
    retry_instruction: str | None = None

    def __init__(
        self,
        text: str | None = None,
        name: str | None = None,
        placeholder_targets: dict = {},
        template_file_path: str | None = None,
    ):
        if template_file_path is not None:
            self.template_file_path = template_file_path
        if self.template_file_path is not None:
            if text is not None:
                raise AssertionError(
                    "A prompt can be specified either by `text`, or by a template (`template_file_path`), not both."
                )
            text = compose_prompt(self.template_file_path, placeholder_targets)
        super().__init__(text)
        self.name = name

    def with_videos_as_frames(self, n_frames: int = 5) -> "Prompt":
        """Returns a new Prompt with all Video items replaced by sampled frames as Images."""
        from ezmm.common.items import Image, Video

        new_data = []
        for item in self.data:
            if isinstance(item, Video):
                frames = item.sample_frames(n_frames, format="jpeg")
                for frame_bytes in frames:
                    new_data.append(Image(binary_data=frame_bytes))
            else:
                new_data.append(item)
        result = object.__new__(self.__class__)
        result.__dict__.update(self.__dict__)
        result.data = new_data
        return result

    def extract(self, response: str) -> dict | str | None:
        """Takes the model's output string and extracts the expected data."""
        return response  # default implementation

    def __len__(self):
        return len(self.__str__())
