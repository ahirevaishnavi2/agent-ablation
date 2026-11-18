import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def strip_headers(text):
    parts = text.split('\n\n', 1)
    return parts[1] if len(parts) > 1 else text

def strip_quotes(text):
    lines = text.split('\n')
    return '\n'.join([line for line in lines if not line.startswith('>')])

def strip_footers(text):
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        if lines[line_num].strip().startswith('--'):
            return '\n'.join(lines[:line_num])
    return text

def clean_text(text):
    text = strip_headers(text)
    text = strip_footers(text)
    text = strip_quotes(text)
    return re.sub(r'\s+', ' ', text).strip()

def main():
    print("Loading 20 Newsgroups manually from folders...")
    
    # 1. Find the actual folder that contains the categories (alt.atheism, etc.)
    # This handles the double-nested "20_newsgroups/20_newsgroups/" issue automatically.
    anchor = list(ROOT.glob("data/raw/**/alt.atheism"))
    if not anchor:
        raise FileNotFoundError("Could not find the 'alt.atheism' folder. Check your extraction.")
    
    base_dir = anchor[0].parent
    print(f"Reading categories from: {base_dir}\n")

    texts = []
    labels = []

    # 2. Loop through each category folder and read the files
    for category_dir in base_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        category_name = category_dir.name
        files_processed = 0
        
        for file_path in category_dir.iterdir():
            if not file_path.is_file():
                continue
                
            try:
                # 20 newsgroups is latin-1 encoded
                raw_text = file_path.read_text(encoding='latin-1', errors='ignore')
                cleaned = clean_text(raw_text)
                
                if len(cleaned) > 50:
                    texts.append(cleaned)
                    labels.append(category_name)
                    files_processed += 1
            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")

        print(f"Loaded {files_processed:4d} docs for category: {category_name}")

    # 3. Create DataFrame and save
    df = pd.DataFrame({'text': texts, 'label': labels})

    out = ROOT / "data/processed/sample_v1.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    
    print(f"\nSuccess! Saved {len(df)} rows to {out}")
    print("\nAverage text length (chars):")
    print(df['text'].str.len().describe())

if __name__ == "__main__":
    main()