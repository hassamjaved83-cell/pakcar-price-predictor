import os
import pandas as pd

from config import OUTPUT_DIR, SAVE_CSV, SAVE_EXCEL


class Exporter:

    def __init__(self):

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.csv_path = os.path.join(OUTPUT_DIR, "cars.csv")
        self.excel_path = os.path.join(OUTPUT_DIR, "cars.xlsx")

    # ----------------------------------------
    # Save Dataset
    # ----------------------------------------

    def save(self, cars, checkpoint=False):

        if len(cars) == 0:
            print("Nothing to save.")
            return

        new_df = pd.DataFrame(cars)

        # ----------------------------------------
        # Merge with existing CSV
        # ----------------------------------------

        if os.path.exists(self.csv_path):

            try:

                old_df = pd.read_csv(self.csv_path)

                new_df = pd.concat(
                    [old_df, new_df],
                    ignore_index=True
                )

            except Exception as e:

                print("Couldn't read existing CSV")

                print(e)

        # ----------------------------------------
        # Remove Duplicate Ads
        # ----------------------------------------

        if "ad_ref" in new_df.columns:

            before = len(new_df)

            new_df = new_df.drop_duplicates(
                subset="ad_ref",
                keep="first"
            )

            after = len(new_df)

            print(f"Duplicates Removed : {before-after}")

        else:

            before = len(new_df)

            new_df = new_df.drop_duplicates()

            after = len(new_df)

            print(f"Duplicates Removed : {before-after}")

        # ----------------------------------------
        # Save CSV
        # ----------------------------------------

        if SAVE_CSV:

            new_df.to_csv(
                self.csv_path,
                index=False
            )

        # ----------------------------------------
        # Save Excel
        # ----------------------------------------

        if SAVE_EXCEL:

            new_df.to_excel(
                self.excel_path,
                index=False
            )

        # ----------------------------------------
        # Messages
        # ----------------------------------------

        if checkpoint:

            print()
            print("=" * 50)
            print("CHECKPOINT SAVED")
            print("=" * 50)

        else:

            print()
            print("=" * 50)
            print("FINAL DATASET SAVED")
            print("=" * 50)

        print(f"Total Cars : {len(new_df)}")
        print(f"CSV : {self.csv_path}")
        print(f"Excel : {self.excel_path}")