from playwright.sync_api import Page
from collector import collect_listing_urls
from parser import extract_car_data
from exporter import Exporter
from logger import logger
from utils import random_delay
from progress import save_progress

from config import (
    MAX_PAGES,
    RETRY_COUNT,
    CHECKPOINT_EVERY,
    DELAY_MIN,
    DELAY_MAX,
)

import time


class Crawler:

    def __init__(self, page: Page):

        self.page = page
        self.exporter = Exporter()

        self.cars = []
        self.scraped_ads = set()

        self.total_success = 0
        self.total_failed = 0

        self.start_time = time.time()

    # ---------------------------------------------
    # Open Search Page
    # ---------------------------------------------

    def open_search_page(self, url):

        self.page.goto(
            url,
            wait_until="domcontentloaded"
        )

        self.page.wait_for_timeout(1500)

    # ---------------------------------------------
    # Crawl
    # ---------------------------------------------

    def crawl(
        self,
        search_url,
        make_index,
        start_page=1
    ):

        self.cars = []
        self.scraped_ads = set()

        self.total_success = 0
        self.total_failed = 0

        self.start_time = time.time()

        logger.info(
            f"Starting Search : {search_url}"
        )

        for page_number in range(
            start_page,
            MAX_PAGES + 1
        ):

            if page_number == 1:

                page_url = search_url

            else:

                if "?" in search_url:

                    page_url = (
                        f"{search_url}&page={page_number}"
                    )

                else:

                    page_url = (
                        f"{search_url}?page={page_number}"
                    )

            print()
            print("=" * 70)
            print(f"PAGE {page_number}")
            print("=" * 70)

            logger.info(
                f"Opening Page {page_number}"
            )

            try:

                self.open_search_page(page_url)

                self.page.locator(
                    "a.car-name.ad-detail-path"
                ).first.wait_for(timeout=10000)

            except Exception:

                print(
                    f"No more pages after Page {page_number}"
                )

                logger.info(
                    f"Finished model at page {page_number}"
                )

                save_progress(
                    make_index=make_index + 1,
                    page_number=1,
                    total_cars=len(self.cars)
                )

                break

            urls = collect_listing_urls(self.page)

            print(
                f"Found {len(urls)} listings"
            )

            logger.info(
                f"Found {len(urls)} listings"
            )

            for index, url in enumerate(
                urls,
                start=1
            ):

                print(
                    f"[{index}/{len(urls)}]"
                )

                self.scrape_listing(
                    url=url,
                    make_index=make_index,
                    page_number=page_number
                )

            print()
            print(
                f"Completed Page {page_number}"
            )

            print(
                f"Cars Collected : {len(self.cars)}"
            )

        elapsed = int(
            time.time() - self.start_time
        )

        print()
        print("=" * 70)
        print("MODEL COMPLETED")
        print("=" * 70)

        print(f"Cars : {len(self.cars)}")
        print(f"Success : {self.total_success}")
        print(f"Failed : {self.total_failed}")
        print(f"Elapsed : {elapsed} sec")

        logger.info(
            "Finished Current Model"
        )

        return self.cars
    # ---------------------------------------------
    # Scrape One Listing
    # ---------------------------------------------

    def scrape_listing(
        self,
        url,
        make_index,
        page_number
    ):

        success = False

        for attempt in range(1, RETRY_COUNT + 1):

            try:

                self.page.goto(
                    url,
                    wait_until="domcontentloaded"
                )

                self.page.locator("h1").wait_for(
                    timeout=10000
                )

                car = extract_car_data(self.page)

                if not car:
                    raise Exception("Parser returned empty data")

                car["url"] = url

                ad_ref = car.get("ad_ref")

                if ad_ref:

                    if ad_ref in self.scraped_ads:

                        logger.info(
                            f"Duplicate skipped : {ad_ref}"
                        )
                        return

                    self.scraped_ads.add(ad_ref)

                self.cars.append(car)

                self.total_success += 1

                # Save progress AFTER EVERY CAR
                save_progress(
                    make_index=make_index,
                    page_number=page_number,
                    total_cars=len(self.cars)
                )

                logger.info(
                    f"Scraped : {car.get('title', 'Unknown')}"
                )

                success = True

                break

            except Exception as e:

                logger.warning(
                    f"Retry {attempt}/{RETRY_COUNT} : {url}"
                )

                print(
                    f"Retry {attempt}/{RETRY_COUNT}"
                )

                print(e)

                if attempt == RETRY_COUNT:

                    self.total_failed += 1

                    logger.error(
                        f"Failed : {url}"
                    )

        random_delay(
            self.page,
            DELAY_MIN,
            DELAY_MAX
        )

        # ---------------------------------------------
        # Checkpoint Save
        # ---------------------------------------------

        if (
            len(self.cars) > 0
            and len(self.cars) % CHECKPOINT_EVERY == 0
        ):

            self.exporter.save(
                self.cars,
                checkpoint=True
            )

            elapsed = int(
                time.time() - self.start_time
            )

            print()
            print("=" * 60)
            print("CHECKPOINT SAVED")
            print("=" * 60)
            print(f"Cars Saved : {len(self.cars)}")
            print(f"Elapsed : {elapsed} sec")

            logger.info(
                f"Checkpoint saved : {len(self.cars)} cars"
            )