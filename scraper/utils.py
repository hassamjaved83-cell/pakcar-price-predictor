import random


def random_delay(page, min_ms, max_ms):

    page.wait_for_timeout(

        random.randint(min_ms, max_ms)

    )