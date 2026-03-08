from datetime import datetime

from mafc.common.claim import Claim


def test_claim_metadata_and_description() -> None:
    claim = Claim(
        "Earth is round.",
        id=123,
        scope=(0, 5),
        dataset="demo",
        author="Alice",
        date=datetime(2025, 1, 2),
        origin="web",
        meta_info="note",
    )

    assert claim.id == "123"
    assert claim.scope == (0, 5)
    assert claim.dataset == "demo"
    assert claim.author == "Alice"
    assert claim.origin == "web"
    assert claim.meta_info == "note"

    description = claim.describe()
    assert 'Claim: "Earth is round."' in description
    assert "Author: Alice" in description
    assert "Date: January 02, 2025" in description
    assert "Origin: web" in description
    assert "Meta info: note" in description

    repr_text = repr(claim)
    assert "Claim(str_len=" in repr_text
    assert "author=Alice" in repr_text


def test_claim_optional_fields_omitted_in_description() -> None:
    claim = Claim("Text only", id=None)
    description = claim.describe()
    assert description == 'Claim: "Text only"'
