from mafc.utils.parsing import get_base_domain, get_domain


def test_get_domain_strips_subdomains() -> None:
    assert get_domain("https://WWW.News.BBC.co.uk/path?q=1") == "co.uk"


def test_get_base_domain_removes_common_prefixes() -> None:
    assert get_base_domain("https://www.facebook.com/some/page") == "facebook.com"
    assert get_base_domain("https://m.example.org/path") == "example.org"


def test_get_base_domain_keeps_other_subdomains() -> None:
    assert get_base_domain("https://api.service.example.com/v1") == "api.service.example.com"
