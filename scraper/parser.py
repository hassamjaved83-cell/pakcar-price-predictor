from playwright.sync_api import Page

# ------------------------------------------
# Extract Car Features
# ------------------------------------------
def extract_features(page: Page):

    features = []

    items = page.locator("ul.car-feature-list li")

    count = items.count()

    for i in range(count):

        text = items.nth(i).inner_text().strip()

        if text:
            features.append(text)

    return features


# ------------------------------------------
# Extract Complete Car Data
# ------------------------------------------
def extract_car_data(page: Page):

    data = {}

    # -----------------------
    # Title
    # -----------------------
    try:
        data["title"] = page.locator("h1").inner_text().strip()
    except:
        data["title"] = None

    # -----------------------
    # Price
    # -----------------------
    try:
        data["price"] = page.locator("strong.generic-green").inner_text().strip()
    except:
        data["price"] = None

    # -----------------------
    # Car Details
    # -----------------------

    KEY_MAP = {
        "Registered In": "registered_city",
        "Color": "color",
        "Assembly": "assembly",
        "Engine Capacity": "engine_cc",
        "Body Type": "body_type",
        "Last Updated:": "last_updated",
        "Ad Ref #": "ad_ref",
        "Auction Grade": "auction_grade",
        "Chassis No.": "chassis_no",
        "Mileage": "mileage",
        "Transmission": "transmission",
        "Fuel Type": "fuel_type",
    }

    try:

        details = page.locator("#scroll_car_detail li")

        count = details.count()

        for i in range(0, count, 2):

            key = details.nth(i).inner_text().strip()

            value = details.nth(i + 1).inner_text().strip()

            key = KEY_MAP.get(
                key,
                key.lower().replace(" ", "_").replace(".", "").replace(":", "")
            )

            data[key] = value

    except Exception as e:

        print("Could not extract details:", e)

    # -----------------------
    # Features
    # -----------------------

    try:

        features = extract_features(page)

        data["features"] = ", ".join(features)

    except:

        data["features"] = ""

    return data