import pandas as pd
import re
import os


# ------------------------------------------
# Clean Price
# ------------------------------------------
def clean_price(price):

    if pd.isna(price):
        return None

    price = str(price).lower()

    number = re.findall(r"[\d.]+", price)

    if not number:
        return None

    number = float(number[0])

    if "lac" in price:
        return int(number * 100000)

    if "crore" in price:
        return int(number * 10000000)

    return int(number)


# ------------------------------------------
# Clean Engine CC
# ------------------------------------------
def clean_engine(engine):

    if pd.isna(engine):
        return None

    number = re.findall(r"\d+", str(engine))

    if number:
        return int(number[0])

    return None


# ------------------------------------------
# Split Title
# ------------------------------------------
def split_title(title):

    if pd.isna(title):
        return None, None, None, None

    parts = str(title).split()

    if len(parts) < 3:
        return None, None, None, None

    brand = parts[0]
    model = parts[1]

    year = None

    if parts[-1].isdigit():

        year = int(parts[-1])

        variant = " ".join(parts[2:-1])

    else:

        variant = " ".join(parts[2:])

    return brand, model, variant, year


# ------------------------------------------
# Convert Features into ML Columns
# ------------------------------------------
def expand_features(df):

    if "features" not in df.columns:
        return df

    all_features = set()

    for value in df["features"].fillna(""):

        for feature in value.split(","):

            feature = feature.strip()

            if feature:
                all_features.add(feature)

    print(f"\nFound {len(all_features)} unique features")

    for feature in sorted(all_features):

        column = (
            feature.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace(".", "")
        )

        df[column] = df["features"].fillna("").apply(
            lambda x: 1 if feature in x else 0
        )

    return df


# ------------------------------------------
# Main Cleaner
# ------------------------------------------
def clean_dataset():

    df = pd.read_csv("../data/raw/cars.csv")

    # Rename columns
    df.rename(columns={
        "Registered In": "registered_city",
        "Engine Capacity": "engine_cc",
        "Body Type": "body_type",
        "Color": "color",
        "Assembly": "assembly",
        "price": "price_pkr",
        "title": "title"
    }, inplace=True)

    # Clean numeric values
    df["price_pkr"] = df["price_pkr"].apply(clean_price)
    df["engine_cc"] = df["engine_cc"].apply(clean_engine)

    # Split title
    df[["brand", "model", "variant", "year"]] = df["title"].apply(
        lambda x: pd.Series(split_title(x))
    )

   

    # Expand Features (NEW)
    df = expand_features(df)

    # Remove duplicates using Ad Ref
    if "ad_ref" in df.columns:
        df.drop_duplicates(subset="ad_ref", inplace=True)
    else:
        df.drop_duplicates(inplace=True)

    os.makedirs("../data/processed", exist_ok=True)

    df.to_csv("../data/processed/cleaned_cars.csv", index=False)

    df.to_excel("../data/processed/cleaned_cars.xlsx", index=False)

    print("\n===================================")
    print("Dataset Cleaned Successfully!")
    print("===================================")

    print(f"Total Cars : {len(df)}")
    print(f"Total Columns : {len(df.columns)}")

    print("\nSaved Files:")
    print("../data/processed/cleaned_cars.csv")
    print("../data/processed/cleaned_cars.xlsx")


if __name__ == "__main__":
    clean_dataset()