import pytest
from predict import predict_sentiment

def test_long_text():
    """Test truncation for text > 128 tokens"""
    long_text = "อร่อย " * 100  # 100 words
    result = predict_sentiment(long_text)
    assert result['sentiment'] in ['positive', 'neutral', 'negative']

def test_mixed_language():
    """Test Thai + English mix"""
    result = predict_sentiment("อาหาร delicious มาก")
    assert result['sentiment'] == 'positive'

def test_special_characters():
    """Test emojis and special chars"""
    result = predict_sentiment("อร่อย!!! 😋😋😋 ⭐⭐⭐⭐⭐")
    assert result['sentiment'] == 'positive'

def test_very_negative():
    """Test strongly negative text"""
    result = predict_sentiment("แย่มาก ไม่ดีเลย แย่สุดๆ")
    assert result['sentiment'] == 'negative'
    assert float(result['confidence'].rstrip('%')) > 70

def test_confidence_scores_format():
    """Test all_scores structure"""
    result = predict_sentiment("ทดสอบ")
    assert len(result['all_scores']) == 3
    assert 'positive' in result['all_scores']
    assert 'neutral' in result['all_scores']
    assert 'negative' in result['all_scores']

def test_batch_consistency():
    """Test same input gives same output"""
    text = "อาหารอร่อย"
    result1 = predict_sentiment(text)
    result2 = predict_sentiment(text)
    assert result1['sentiment'] == result2['sentiment']