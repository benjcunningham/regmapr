import argparse
import pandas as pd

def read_dataset(path):

    # Read data, set column names
    cols = ["sentence_A", "sentence_B", "relatedness_score"]
    df = pd.read_table(path, sep="\t")[cols]
    df.columns = ["s1", "s2", "target"]

    # Scale from [1, 5] to [0, 1]
    cent = df["target"] - min(df["target"])
    df["target"] = cent / max(cent)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df = read_dataset(args.path)
    df.to_csv(args.out, sep="\t", float_format="%.3f", index=False, header=True)

