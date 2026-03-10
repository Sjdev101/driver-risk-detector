from src.analyzer import DrowsinessAnalyzer, PhoneAnalyzer

# DrowsinessAnalyzer tests
def test_drowsy_when_ear_below_threshold():
    analyzer = DrowsinessAnalyzer(ear_threshold=0.25, frame_threshold=1)
    result = analyzer.update(0.15)
    assert result == True

def test_alert_when_ear_above_threshold():
    analyzer = DrowsinessAnalyzer(ear_threshold=0.25, frame_threshold=1)
    result = analyzer.update(0.35)
    assert result == False

def test_drowsy_only_after_frame_threshold():
    analyzer = DrowsinessAnalyzer(ear_threshold=0.25, frame_threshold=3)
    analyzer.update(0.15)
    analyzer.update(0.15)
    result = analyzer.update(0.15)
    assert result == True

def test_is_new_event_triggers_once():
    analyzer = DrowsinessAnalyzer(ear_threshold=0.25, frame_threshold=1)
    analyzer.update(0.15)
    first = analyzer.is_new_event()
    second = analyzer.is_new_event()
    assert first == True
    assert second == False

# PhoneAnalyzer tests
def test_phone_new_event_when_detected():
    analyzer = PhoneAnalyzer()
    result = analyzer.is_new_event([{"bbox": (0,0,100,100), "confidence": 0.9}])
    assert result == True

def test_phone_no_duplicate_event():
    analyzer = PhoneAnalyzer()
    analyzer.is_new_event([{"bbox": (0,0,100,100), "confidence": 0.9}])
    result = analyzer.is_new_event([{"bbox": (0,0,100,100), "confidence": 0.9}])
    assert result == False

def test_phone_resets_when_gone():
    analyzer = PhoneAnalyzer()
    analyzer.is_new_event([{"bbox": (0,0,100,100), "confidence": 0.9}])
    analyzer.is_new_event([])
    result = analyzer.is_new_event([{"bbox": (0,0,100,100), "confidence": 0.9}])
    assert result == True
