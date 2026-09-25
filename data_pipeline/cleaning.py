
"""Clean scraped book data and apply the project-defined fixed GBP-to-INR rate."""

import pandas as pd

GBP_TO_INR = 105.50


def clean_books(input_csv="books_raw.csv", output_csv="books_clean.csv"):
    df = pd.read_csv(input_csv)

    # Clean and convert prices
    df["price_gbp"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
        .str.strip()
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"], errors="coerce"
    )

    # Convert ratings to numeric values
    df["rating"] = pd.to_numeric(
        df["rating"], errors="coerce"
    )

    # Identify books that are in stock
    df["in_stock"] = (
        df["availability"]
        .astype(str)
        .str.contains("In stock", case=False, na=False)
    )

    # Convert GBP to INR
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    # Check for invalid values
    if df["price_gbp"].isna().any() or df["rating"].isna().any():
        raise ValueError(
            "Some price/rating values failed to parse; inspect the raw rows."
        )

    # Save cleaned data
    df.to_csv(output_csv, index=False)

    print(f"Cleaning completed successfully! {len(df)} books processed.")
    print(f"Cleaned data saved to: {output_csv}")

    return df


if __name__ == "__main__":
    clean_books()