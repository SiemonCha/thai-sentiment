import pandas as pd
from predict import predict_sentiment

# Read CSV
df = pd.read_csv('input.csv')  # Must have 'text' column

# Predict all
results = []
for text in df['text']:
    result = predict_sentiment(text)
    results.append({
        'text': text,
        'sentiment': result['sentiment'],
        'confidence': result['confidence']
    })

# Save results
pd.DataFrame(results).to_csv('output.csv', index=False)
print(">>> Saved predictions to output.csv")