from datasets import load_dataset
import os
import argparse
import subprocess


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