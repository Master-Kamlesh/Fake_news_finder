"""
detector_app.py

Interactive CLI application for fake-news detection.
Users can input headlines/articles and get real-time predictions.

Usage:
    python detector_app.py
    python detector_app.py --use-transformer (for ML model)
    python detector_app.py --file articles.txt (batch process)
"""

import sys
import argparse
import json
from pathlib import Path
from fake_news_detector import FakeNewsDetector
from url_analyzer import URLArticleAnalyzer


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("  🔍 FAKE NEWS DETECTOR - AI-Powered Article Analysis")
    print("=" * 70)
    print("\nThis tool analyzes text for signs of misinformation using:")
    print("  • Rule-based heuristics (sensationalism, caps lock, punctuation)")
    print("  • Optional: Transformer-based ML models for deeper analysis")
    print("\n" + "-" * 70 + "\n")


def print_result(text: str, result: dict, index: int = None):
    """Pretty-print prediction result."""
    prefix = f"[Article {index}] " if index else ""
    print(f"\n{prefix}Input: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Label:      {result['label']}")
    print(f"  Score:      {result['fake_score']} (0=Real, 1=Fake)")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Method:     {result['method']}")

    if "details" in result and "rule_based" in result["details"]:
        details = result["details"]["rule_based"]
        print(f"  Analysis:   {details}")

    # Show color-coded emoji
    if result["label"] == "FAKE":
        print("  Status:     ⚠️  LIKELY FAKE NEWS")
    else:
        print("  Status:     ✓ LIKELY CREDIBLE")


def print_url_result(result: dict):
    """Pretty-print URL analysis result."""
    if result.get("status") != "success":
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        return

    print(f"\n{'=' * 70}")
    print(f"📄 Article: {result.get('title', 'No title')}")
    print(f"🔗 URL: {result.get('url')}")
    print(f"{'=' * 70}")

    if result.get("content_preview"):
        print(f"\nContent preview: {result['content_preview']}...")

    if "title_analysis" in result:
        print(f"\n📰 Headline Analysis:")
        print(f"  Label:  {result['title_analysis']['label']}")
        print(f"  Score:  {result['title_analysis']['fake_score']}")

    if "content_analysis" in result:
        print(f"\n📝 Content Analysis:")
        print(f"  Label:  {result['content_analysis']['label']}")
        print(f"  Score:  {result['content_analysis']['fake_score']}")

    if "overall_label" in result:
        print(f"\n{'🚨' if result['overall_label'] == 'FAKE' else '✅'} OVERALL VERDICT:")
        print(f"  Label:  {result['overall_label']}")
        print(f"  Score:  {result['overall_fake_score']} (0=Real, 1=Fake)")

    print(f"\n{'=' * 70}")


def interactive_mode(detector: FakeNewsDetector):
    """Run interactive CLI mode."""
    print_banner()
    print("Enter headlines or article text (type 'quit' to exit, 'clear' to reset):\n")

    while True:
        try:
            text = input(">> ").strip()

            if text.lower() == "quit":
                print("\nGoodbye!")
                break

            if text.lower() == "clear":
                print("\n" * 2)
                continue

            if not text:
                continue

            result = detector.predict(text)
            print_result(text, result)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_mode(detector: FakeNewsDetector, file_path: str):
    """Process a batch of articles from a file."""
    print_banner()
    print(f"Processing file: {file_path}\n")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        results = []
        for i, line in enumerate(lines, 1):
            text = line.strip()
            if not text:
                continue
            result = detector.predict(text)
            results.append((text, result))
            print_result(text, result, index=i)

        # Summary
        fake_count = sum(1 for _, r in results if r["label"] == "FAKE")
        real_count = len(results) - fake_count
        print("\n" + "=" * 70)
        print(f"Summary: {real_count} likely real, {fake_count} likely fake out of {len(results)} articles")
        print("=" * 70 + "\n")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fake News Detector - AI-powered misinformation analyzer"
    )
    parser.add_argument(
        "--use-transformer",
        action="store_true",
        help="Use transformer model (slower, more accurate; requires transformers + torch)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Batch process a file with one article/headline per line",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Analyze a single text directly",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Fetch and analyze article from URL",
    )

    args = parser.parse_args()

    # Initialize detector
    print("[INFO] Initializing detector...")
    detector = FakeNewsDetector(use_transformer=args.use_transformer)

    if args.url:
        # URL analysis mode
        print_banner()
        print(f"[INFO] Fetching article from: {args.url}")
        try:
            analyzer = URLArticleAnalyzer(use_transformer=args.use_transformer)
            result = analyzer.analyze_url(args.url)
            print_url_result(result)
        except ImportError as e:
            print(f"[ERROR] {e}")
            print("[INFO] Install required packages: pip install requests beautifulsoup4")

    elif args.text:
        # Single text mode
        print_banner()
        result = detector.predict(args.text)
        print_result(args.text, result)

    elif args.file:
        # Batch mode
        batch_mode(detector, args.file)

    else:
        # Interactive mode
        interactive_mode(detector)


if __name__ == "__main__":
    main()
