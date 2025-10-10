from datasets import load_dataset
import os

def download_thai_sentiment():
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
    
if __name__ == "__main__":
    download_thai_sentiment()