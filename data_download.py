from datasets import load_dataset
import os

def download_thai_sentiment():
    # Thai sentiment from social media (wisesight)
    dataset = load_dataset("wisesight_sentiment")
    
    os.makedirs("./data", exist_ok=True)
    dataset.save_to_disk("./data/wisesight")
    
    print(f"Train: {len(dataset['train'])}")
    print(f"Test: {len(dataset['test'])}")
    
    # Check actual column names
    print(f"Columns: {dataset['train'].column_names}")
    print(f"First sample: {dataset['train'][0]}")
    
if __name__ == "__main__":
    download_thai_sentiment()