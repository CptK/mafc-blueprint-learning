from mafc.utils.console import (
    bold,
    cyan,
    gray,
    green,
    it,
    light_blue,
    magenta,
    num2text,
    red,
    remove_string_formatters,
    sec2hhmmss,
    sec2mmss,
    ul,
    yellow,
)


def test_color_wrappers_are_removable() -> None:
    text = "hello"
    wrapped = [
        gray(text),
        light_blue(text),
        green(text),
        yellow(text),
        red(text),
        magenta(text),
        cyan(text),
        bold(text),
        it(text),
        ul(text),
    ]

    for value in wrapped:
        assert remove_string_formatters(value) == text


def test_num2text_branches() -> None:
    assert num2text(0) == "0"
    assert num2text(0.1234) == "0.12"
    assert num2text(5.66) == "5.7"
    assert num2text(42) == "42"
    assert num2text(1500) == "1.5K"
    assert num2text(10500) == "10K"
    assert num2text(2_300_000) == "2.3M"
    assert num2text(25_000_000) == "25M"


def test_seconds_formatters() -> None:
    assert sec2hhmmss(None) is None
    assert sec2hhmmss(3661) == "1:01:01 h"

    assert sec2mmss(None) is None
    assert sec2mmss(125) == "2:05 min"


def test_remove_string_formatters_keeps_plain_text() -> None:
    assert remove_string_formatters("plain text") == "plain text"
