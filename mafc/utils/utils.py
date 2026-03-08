from typing import Any


def flatten_dict(to_flatten: dict[str, Any]) -> dict[str, Any]:
    """Flattens a nested dictionary which has string keys. Renames the keys using
    the scheme "<outer_key>/<inner_key>/..."."""
    flat_dict = {}
    for outer_key, outer_value in to_flatten.items():
        if isinstance(outer_value, dict):
            flat_dict_inner = flatten_dict(outer_value)
            for inner_key, inner_value in flat_dict_inner.items():
                flat_dict[f"{outer_key}/{inner_key}"] = inner_value
        else:
            flat_dict[outer_key] = outer_value
    return flat_dict
