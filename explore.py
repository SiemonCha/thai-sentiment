from datasets import load_from_disk
from collections import Counter

dataset = load_from_disk("./data/wisesight")
train = dataset['train']

# Check structure first
print(f"Columns: {train.column_names}")
print(f"First sample:\n{train[0]}\n")

# Stats
print(f"Total samples: {len(train)}")

# Label distribution (adjust column name based on output)
label_col = 'category' if 'category' in train.column_names else 'label'
print(f"\nLabel distribution:")
print(Counter(train[label_col]))