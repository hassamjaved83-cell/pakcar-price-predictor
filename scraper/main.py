from playwright.sync_api import sync_playwright

from crawler import Crawler
from exporter import Exporter
from url_builder import build_search_url
from progress import load_progress, clear_progress

from config import *


def main():

    progress = load_progress()

    start_make = 0
    resume_page = 1

    if progress:

        print("=" * 60)
        print("Previous scraping session found")
        print("=" * 60)
        print(f"Last Make Index : {progress['make_index']}")
        print(f"Last Page       : {progress['page_number']}")
        print()

        choice = input("Resume previous scraping? (Y/N): ").strip().upper()

        if choice == "Y":

            start_make = progress["make_index"]
            resume_page = progress["page_number"]

            print("\nResuming...\n")

        else:

            clear_progress()

            start_make = 0
            resume_page = 1

            print("\nStarting New Scraping...\n")

    all_cars = []

    exporter = Exporter()

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=HEADLESS
        )

        page = browser.new_page()

        crawler = Crawler(page)

        for i in range(start_make, len(CARS)):

            make, model = CARS[i]

            print("\n")
            print("=" * 80)
            print(f"SCRAPING : {make} {model}")
            print("=" * 80)

            url = build_search_url(make, model)

            if i == start_make:

                cars = crawler.crawl(
                    search_url=url,
                    make_index=i,
                    start_page=resume_page
                )

            else:

                cars = crawler.crawl(
                    search_url=url,
                    make_index=i,
                    start_page=1
                )

            all_cars.extend(cars)

            print(f"Collected So Far : {len(all_cars)}")

        browser.close()

    exporter.save(all_cars)

    clear_progress()

    print("\nScraping Finished Successfully!")


if __name__ == "__main__":
    main()