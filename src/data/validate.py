from load import load_all


def validate_data(data):

    print("\n========== DATA VALIDATION ==========\n")

    for name, df in data.items():

        print(f"\n--- {name} ---")

        print("Shape:", df.shape)

        missing = df.isna().sum()

        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing):
            print("\nMissing values:")
            print(missing.head(10))
        else:
            print("Missing values: none")

        print("\nColumns:")
        print(list(df.columns))


if __name__ == "__main__":
    data = load_all()
    validate_data(data)