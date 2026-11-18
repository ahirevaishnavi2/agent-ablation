import argparse
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw CSV in data/raw/")
    ap.add_argument("--out", default="data/processed/sample_v1.parquet")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-col", default="consumer_complaint_narrative")
    ap.add_argument("--label-col", default="product")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--scan-limit", type=int, default=None,
                    help="Optional: stop after reading this many raw rows (debug)")
    args = ap.parse_args()

    in_path = ROOT / args.input
    if not in_path.exists():
        raise SystemExit(f"Not found: {in_path}")

    header = pd.read_csv(in_path, nrows=0).columns.tolist()
    missing = {args.text_col, args.label_col} - set(header)
    if missing:
        raise SystemExit(
            f"Missing column(s): {sorted(missing)}\n"
            f"Available columns: {header}"
        )

    rng = random.Random(args.seed)
    reservoir = []          # list of dicts, max len = args.n
    seen_raw = 0            # total raw rows read
    seen_ok = 0             # eligible rows seen (for reservoir bookkeeping)

    reader = pd.read_csv(
        in_path,
        usecols=[args.text_col, args.label_col],
        dtype=str,
        chunksize=args.chunksize,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
    )

    for chunk in reader:
        seen_raw += len(chunk)

        chunk = chunk.dropna(subset=[args.text_col, args.label_col])
        chunk = chunk[chunk[args.text_col].str.strip().str.len() >= args.min_chars]
        chunk = chunk[chunk[args.text_col].str.contains(r"\w", regex=True, na=False)]

        for text, label in chunk.itertuples(index=False, name=None):
            seen_ok += 1
            if len(reservoir) < args.n:
                reservoir.append({"text": text, "label": label})
            else:
                j = rng.randrange(seen_ok)
                if j < args.n:
                    reservoir[j] = {"text": text, "label": label}

        print(f"raw={seen_raw:,}  eligible={seen_ok:,}  kept={len(reservoir):,}")

        if args.scan_limit and seen_raw >= args.scan_limit:
            print(f"Reached scan limit {args.scan_limit:,}; stopping early.")
            break
        if len(reservoir) >= args.n and seen_raw >= 2_000_000:
            # We already have a full reservoir; we still keep scanning so the
            # sample stays uniform across the whole file. Uncomment the break
            # below if you prefer speed over exact uniformity.
            # break
            pass

    if not reservoir:
        raise SystemExit("No eligible rows found. Check column names / filter.")

    df = pd.DataFrame(reservoir).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Canonical names so base.yaml never needs editing again.
    df = df.rename(columns={"text": "consumer_complaint_narrative", "label": "product"})

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"\nraw rows read: {seen_raw:,}")
    print(f"eligible rows: {seen_ok:,}  ({seen_ok / max(1, seen_raw):.2%} of raw)")
    print(f"sampled: {len(df):,}  seed={args.seed}  -> {out}")
    print("\nLabel distribution:")
    print(df["product"].value_counts())
    print("\nNarrative length (chars):")
    print(df["consumer_complaint_narrative"].str.len().describe())


if __name__ == "__main__":
    main()