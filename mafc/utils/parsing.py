from urllib.parse import urlparse


def get_domain(url: str) -> str:
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.lower()  # get the network location (netloc)
    domain = ".".join(netloc.split(".")[-2:])  # remove subdomains
    return domain


def get_base_domain(url: str) -> str:
    """
    Extracts the base domain from a given URL, ignoring common subdomains like 'www' and 'm'.

    Args:
        url (str): The URL to extract the base domain from.

    Returns:
        str: The base domain (e.g., 'facebook.com').
    """
    netloc = urlparse(url).netloc

    # Remove common subdomains like 'www.' and 'm.'
    if netloc.startswith("www.") or netloc.startswith("m."):
        netloc = netloc.split(".", 1)[1]  # Remove the first part (e.g., 'www.', 'm.')

    return netloc
