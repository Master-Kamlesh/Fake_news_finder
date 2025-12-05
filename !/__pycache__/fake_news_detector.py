"""
fake_news_detector.py

A fake-news detection system using pre-trained transformer models.
Provides both a simple rule-based detector and an ML-based classifier.

Usage:
    from fake_news_detector import FakeNewsDetector
    detector = FakeNewsDetector()
    result = detector.predict("Your headline or article text here")
    print(result)
"""

import re
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class SimpleRuleBasedDetector:
    """A lightweight rule-based detector using heuristics for quick assessment."""

    # Common sensational/clickbait patterns
    SENSATIONAL_WORDS = [
        "shocking", "unbelievable", "you won't believe", "this will shock you",
        "celebrities hate", "doctors hate", "government doesn't want you to know",
        "one weird trick", "number 7 will blow your mind", "he did what?!",
        "she secretly", "he secretly", "this is why", "the truth about",
    ]

    # Patterns suggesting low credibility
    SUSPICIOUS_PATTERNS = [
        r"(?i)all caps",  # Heavy use of caps
        r"(?i)[\!\?]{3,}",  # Multiple ! or ?
        r"(?i)you won't believe",
        r"(?i)sponsored content",
    ]

    def __init__(self):
        self.sensational_score = 0.0

    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze text using heuristics. Returns dict with 'fake_score' (0-1)."""
        text_lower = text.lower()
        scores = []

        # Check for sensational words
        sensational_count = sum(1 for word in self.SENSATIONAL_WORDS if word in text_lower)
        if sensational_count > 0:
            scores.append(min(sensational_count * 0.15, 0.5))

        # Check for excessive punctuation
        exclamation_ratio = text.count("!") / max(len(text.split()), 1)
        if exclamation_ratio > 0.05:
            scores.append(min(exclamation_ratio * 2, 0.3))

        # Check for caps lock abuse
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len([c for c in text if c.isalpha()]), 1)
        if caps_ratio > 0.3:
            scores.append(0.2)

        # Check for suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text):
                scores.append(0.15)

        # Check text length (very short or very long can be suspicious)
        word_count = len(text.split())
        if word_count < 5 or word_count > 500:
            scores.append(0.1)

        fake_score = min(sum(scores), 1.0)
        return {"fake_score": fake_score, "method": "rule-based"}


class TransformerBasedDetector:
    """ML-based detector using pre-trained transformer models from Hugging Face."""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize with a pre-trained model.
        Default: DistilBERT fine-tuned for sentiment (proxy for credibility).
        You can swap for specialized models if available.
        """
        self.model_name = model_name
        self.pipeline = None
        self._init_model()

    def _init_model(self):
        """Lazy-load the transformer pipeline."""
        try:
            from transformers import pipeline
            print(f"[INFO] Loading transformer model: {self.model_name}")
            self.pipeline = pipeline("sentiment-analysis", model=self.model_name)
            print("[INFO] Model loaded successfully.")
        except ImportError:
            print(
                "[WARNING] transformers library not found. Install with:"
                " pip install transformers torch"
            )
            self.pipeline = None
        except Exception as e:
            print(f"[WARNING] Failed to load transformer model: {e}")
            self.pipeline = None

    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze text using transformer. Returns dict with 'fake_score' (0-1)."""
        if not self.pipeline:
            return {"fake_score": 0.5, "error": "Model not loaded", "method": "transformer"}

        # Truncate text if too long (most models have token limits)
        text = text[:512]

        try:
            result = self.pipeline(text)[0]
            label = result["label"].lower()
            score = result["score"]

            # Heuristic: negative sentiment can indicate sensationalism/fake news
            # (This is a rough proxy; specialized models would be better)
            if label == "negative":
                fake_score = score
            else:
                fake_score = 1.0 - score

            return {
                "fake_score": fake_score,
                "sentiment": label,
                "confidence": score,
                "method": "transformer",
            }
        except Exception as e:
            print(f"[ERROR] Transformer analysis failed: {e}")
            return {"fake_score": 0.5, "error": str(e), "method": "transformer"}


class FakeNewsDetector:
    """Main detector combining rule-based and ML approaches."""

    def __init__(self, use_transformer: bool = False):
        """Initialize detector.

        Args:
            use_transformer: If True, use transformer model (requires transformers + torch).
                           If False, use faster rule-based detector.
        """
        self.use_transformer = use_transformer
        self.rule_detector = SimpleRuleBasedDetector()
        self.transformer_detector = None
        if use_transformer:
            self.transformer_detector = TransformerBasedDetector()

    def predict(self, text: str) -> Dict:
        """Predict if text is fake news (0 = real, 1 = fake).

        Returns dict with:
            - fake_score: float 0-1
            - confidence: float 0-1
            - label: "FAKE" or "REAL"
            - method: detection method used
            - details: additional analysis info
        """
        if not text or len(text.strip()) == 0:
            return {
                "fake_score": 0.5,
                "confidence": 0.0,
                "label": "UNKNOWN",
                "error": "Empty text",
            }

        # Start with rule-based analysis
        rule_result = self.rule_detector.analyze(text)
        fake_score = rule_result["fake_score"]
        details = {"rule_based": rule_result}

        # Optionally combine with transformer analysis
        if self.use_transformer and self.transformer_detector:
            transformer_result = self.transformer_detector.analyze(text)
            details["transformer"] = transformer_result
            if "fake_score" in transformer_result and "error" not in transformer_result:
                # Weighted average: 60% rule-based, 40% transformer
                fake_score = 0.6 * fake_score + 0.4 * transformer_result["fake_score"]

        # Determine label based on threshold
        confidence = abs(fake_score - 0.5) * 2  # 0-1 confidence
        label = "FAKE" if fake_score > 0.5 else "REAL"

        return {
            "fake_score": round(fake_score, 3),
            "confidence": round(confidence, 3),
            "label": label,
            "method": "hybrid" if self.use_transformer else "rule-based",
            "details": details,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict on multiple texts. Returns list of prediction dicts."""
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("Fake News Detector - Test")
    print("=" * 60)

    detector_simple = FakeNewsDetector(use_transformer=False)
    test_texts = [
        "Breaking: Scientists discover cure for all diseases!!!",
        "The city council approved a new transportation plan.",
        "You won't BELIEVE what celebrities are hiding from you!!! Number 7 will shock you!!!",
        "A peer-reviewed study published in Nature found new evidence for climate change.",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n[Test {i}] {text[:70]}...")
        result = detector_simple.predict(text)
        print(f"  Label: {result['label']}")
        print(f"  Fake Score: {result['fake_score']}")
        print(f"  Confidence: {result['confidence']}")
