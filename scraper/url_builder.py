from urllib.parse import quote_plus


BASE_URL = "https://www.pakwheels.com/used-cars/search/-/"


def build_search_url(make: str, model: str):

    query = quote_plus(f"{make} {model}")

    return f"{BASE_URL}?q={query}"