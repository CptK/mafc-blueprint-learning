from mafc.utils.utils import flatten_dict


def test_flatten_dict_with_nested_structure() -> None:
    source = {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3,
            },
        },
        "f": "x",
    }

    flattened = flatten_dict(source)

    assert flattened == {
        "a": 1,
        "b/c": 2,
        "b/d/e": 3,
        "f": "x",
    }


def test_flatten_dict_empty() -> None:
    assert flatten_dict({}) == {}
