from datasets import load_dataset
import os
import argparse
import subprocess

dataset = load_dataset("wisesight_sentiment")

# Normalize text column name to 'texts' (other code expects 'texts')
# try common column names and rename to 'texts' if needed
sample_cols = dataset['train'].column_names
if 'texts' not in sample_cols:
    if 'text' in sample_cols:
            dataset = dataset.map(lambda ex: {'texts': ex['text']})
    elif 'sentence' in sample_cols:
            dataset = dataset.map(lambda ex: {'texts': ex['sentence']})
    else:
        raise RuntimeError(f"No known text column found in dataset. Columns: {sample_cols}")

train_data = dataset['train'].filter(lambda x: x['category'] != 3)
test_data = dataset['test'].filter(lambda x: x['category'] != 3)


def download_thai_sentiment(explore=False):
    dataset = load_dataset("wisesight_sentiment")

    train_data = dataset['train'].filter(lambda x: x['category'] != 3)
    test_data = dataset['test'].filter(lambda x: x['category'] != 3)

    os.makedirs("./data", exist_ok=True)

    from datasets import DatasetDict
    dataset_clean = DatasetDict({
        'train': train_data,
        'test': test_data
    })

    dataset_clean.save_to_disk("./data/wisesight")

    print(f"Train: {len(train_data)}")
    print(f"Test: {len(test_data)}")
    print(f"Label distribution: pos/neu/neg only")

    if explore:
        # run explore.py to verify dataset structure and basic stats
        try:
            print('\nRunning explore.py to verify dataset...')
            subprocess.run(["python", "explore.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"explore.py failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Wisesight dataset and save to ./data/wisesight")
    parser.add_argument("--explore", action="store_true", help="Run explore.py after download to inspect dataset")
    args = parser.parse_args()
    download_thai_sentiment(explore=args.explore)