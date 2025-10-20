import pytest
from predict import predict_sentiment

def test_positive_sentiment():
    result = predict_sentiment("อาหารอร่อยมาก บริการดีเยี่ยม")
    assert result['sentiment'] == 'positive'
    assert float(result['confidence'].rstrip('%')) > 80

def test_negative_sentiment():
    result = predict_sentiment("ราคาแพงไป คุณภาพไม่คุ้ม")
    assert result['sentiment'] == 'negative'

def test_neutral_sentiment():
    result = predict_sentiment("ปกติครับ")
    assert result['sentiment'] in ['neutral', 'positive']

def test_empty_text():
    result = predict_sentiment("")
    assert result['sentiment'] in ['positive', 'neutral', 'negative']
    assert 'confidence' in result

def test_confidence_range():
    result = predict_sentiment("อาหารอร่อยมาก")
    conf = float(result['confidence'].rstrip('%'))
    assert 0 <= conf <= 100

def test_all_scores_sum():
    result = predict_sentiment("ทดสอบ")
    scores = [float(s.rstrip('%')) for s in result['all_scores'].values()]
    assert 99 <= sum(scores) <= 101  # Allow rounding error