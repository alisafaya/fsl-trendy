import pandas as pd
import os

if __name__ == "__main__":
    indir = "topic_queries"
    outdir = "topic_negatives_2ndpass"

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    for filename in os.listdir(indir):
        df = pd.read_json(os.path.join(indir, filename), lines=True)
        print("=" * 30)
        print(filename, len(df))
        print("=" * 30)

        # unique by id
        df = df.drop_duplicates(subset=["id", "tweet", "user_id", "conversation_id"])
        print("Dropped duplicates", len(df))

        df = df[df.language == "en"]
        print("English only", len(df))

        df = df.sample(frac=0.1)
        if filename in os.listdir(outdir):
            continue

        new_samples = []
        for i, row in df.iterrows():
            print("=" * 30)
            print(row["tweet"])
            if input("Sample? ") == "y":
                new_samples.append(row)

            if len(new_samples) >= 16:
                break

        new_df = pd.DataFrame(new_samples)
        new_df.to_json(os.path.join(outdir, filename), orient="records", lines=True)
