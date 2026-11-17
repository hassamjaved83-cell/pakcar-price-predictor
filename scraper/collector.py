from playwright.sync_api import Page


def collect_listing_urls(page: Page):

    urls = []

    listings = page.locator("a.car-name.ad-detail-path")

    count = listings.count()

    print(f"Found {count} listings")

    for i in range(count):

        try:

            href = listings.nth(i).get_attribute("href")

            if href:

                if href.startswith("/"):

                    href = "https://www.pakwheels.com" + href

                if href not in urls:

                    urls.append(href)

        except Exception as e:

            print(f"Error collecting URL {i + 1}: {e}")

    return urls