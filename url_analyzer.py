"""
url_analyzer.py

Fetch articles from URLs and analyze them for fake news.
Supports: direct HTML scraping, Open Graph meta tags, and article extraction.

Usage:
    from url_analyzer import fetch_and_analyze
    result = fetch_and_analyze('https://example.com/article', use_transformer=False)
    print(result)
"""

import re
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

from fake_news_detector import FakeNewsDetector


class ArticleExtractor:
    """Extract article title, content, and metadata from HTML."""

    @staticmethod
    def extract_text_from_html(html: str) -> Tuple[str, str]:
        """Extract title and main content from HTML.

        Returns:
            (title, content) tuple
        """
        if not BeautifulSoup:
            raise ImportError("BeautifulSoup4 required. Install: pip install beautifulsoup4")

        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = ""
        if soup.find("title"):
            title = soup.find("title").get_text(strip=True)
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        elif soup.find("meta", {"property": "og:title"}):
            title = soup.find("meta", {"property": "og:title"}).get("content", "")

        # Extract main content
        content = ""

        # Try common article containers
        article = soup.find("article")
        if article:
            content = article.get_text(separator=" ", strip=True)
        else:
            # Try main content areas
            for selector in ["main", "div.content", "div.article-body", "div.post-content"]:
                elem = soup.select_one(selector)
                if elem:
                    content = elem.get_text(separator=" ", strip=True)
                    break

        # Fallback: extract all paragraphs
        if not content or len(content) < 50:
            paragraphs = soup.find_all("p")
            content = " ".join([p.get_text(strip=True) for p in paragraphs])

        # Remove common boilerplate
        content = re.sub(r"(Subscribe|Read more|Copyright|©|All rights reserved).*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s+", " ", content).strip()

        return title, content

    @staticmethod
    def extract_meta_description(html: str) -> str:
        """Extract description from Open Graph or meta description tags."""
        if not BeautifulSoup:
            return ""

        soup = BeautifulSoup(html, "html.parser")

        # Try Open Graph
        og_desc = soup.find("meta", {"property": "og:description"})
        if og_desc:
            return og_desc.get("content", "")

        # Try standard meta description
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc:
            return meta_desc.get("content", "")

        return ""


class URLArticleAnalyzer:
    """Fetch and analyze articles from URLs."""

    def __init__(self, timeout: int = 10, use_transformer: bool = False):
        """Initialize analyzer.

        Args:
            timeout: Request timeout in seconds
            use_transformer: Use ML model for analysis
        """
        if not requests:
            raise ImportError("requests library required. Install: pip install requests")

        self.timeout = timeout
        self.detector = FakeNewsDetector(use_transformer=use_transformer)
        self.extractor = ArticleExtractor()

    def fetch_article(self, url: str) -> Dict[str, Optional[str]]:
        """Fetch article from URL.

        Returns:
            Dict with 'title', 'content', 'url', 'status', 'error' keys
        """
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme:
                url = "https://" + url
            if not parsed.netloc and not url.startswith("https://"):
                return {
                    "url": url,
                    "title": None,
                    "content": None,
                    "status": "error",
                    "error": "Invalid URL format",
                }

            # Fetch HTML
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()

            # Extract content
            title, content = self.extractor.extract_text_from_html(resp.text)
            description = self.extractor.extract_meta_description(resp.text)

            if not content:
                return {
                    "url": url,
                    "title": title,
                    "content": None,
                    "status": "error",
                    "error": "Could not extract article content",
                }

            return {
                "url": url,
                "title": title,
                "description": description,
                "content": content,
                "status": "success",
            }

        except requests.exceptions.Timeout:
            return {
                "url": url,
                "status": "error",
                "error": f"Request timeout (>{self.timeout}s)",
            }
        except requests.exceptions.ConnectionError:
            return {
                "url": url,
                "status": "error",
                "error": "Connection error. Check URL or internet connection.",
            }
        except requests.exceptions.HTTPError as e:
            return {
                "url": url,
                "status": "error",
                "error": f"HTTP error: {e.response.status_code}",
            }
        except Exception as e:
            return {
                "url": url,
                "status": "error",
                "error": f"Extraction error: {str(e)}",
            }

    def analyze_url(self, url: str, analyze_title: bool = True, analyze_content: bool = True) -> Dict:
        """Fetch and analyze article from URL.

        Args:
            url: Article URL
            analyze_title: Also analyze headline
            analyze_content: Analyze article body

        Returns:
            Dict with fetch status and analysis results
        """
        # Fetch article
        fetch_result = self.fetch_article(url)

        if fetch_result.get("status") != "success":
            return fetch_result

        results = {
            "url": url,
            "title": fetch_result.get("title"),
            "content_preview": fetch_result.get("content", "")[:200] if fetch_result.get("content") else None,
            "status": "success",
        }

        # Analyze title
        if analyze_title and fetch_result.get("title"):
            title_result = self.detector.predict(fetch_result["title"])
            results["title_analysis"] = title_result

        # Analyze content
        if analyze_content and fetch_result.get("content"):
            # Truncate if too long (transformer has token limits)
            content = fetch_result["content"][:2000]
            content_result = self.detector.predict(content)
            results["content_analysis"] = content_result

        # Combined score (average if both)
        if "title_analysis" in results and "content_analysis" in results:
            combined_score = (
                0.3 * results["title_analysis"]["fake_score"]
                + 0.7 * results["content_analysis"]["fake_score"]
            )
            results["overall_fake_score"] = round(combined_score, 3)
            results["overall_label"] = "FAKE" if combined_score > 0.5 else "REAL"
        elif "content_analysis" in results:
            results["overall_fake_score"] = results["content_analysis"]["fake_score"]
            results["overall_label"] = results["content_analysis"]["label"]
        elif "title_analysis" in results:
            results["overall_fake_score"] = results["title_analysis"]["fake_score"]
            results["overall_label"] = results["title_analysis"]["label"]

        return results

def fetch_and_analyze(url: str, use_transformer: bool = False) -> Dict:
    """Convenience function to fetch and analyze a single URL.

    Args:
        url: Article URL
        use_transformer: Use ML model for analysis

    Returns:
        Analysis result dict
    """
    analyzer = URLArticleAnalyzer(use_transformer=use_transformer)
    return analyzer.analyze_url(url)


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


if __name__ == "__main__":
    import json

    print("\n" + "=" * 70)
    print("  🔍 URL ARTICLE ANALYZER - Fake News Detection")
    print("=" * 70)
    print("\nThis tool fetches articles from URLs and analyzes them for misinformation.\n")

    # Get URL from user
    url = input("Enter article URL to analyze (or 'quit' to exit): ").strip()

    if url.lower() == 'quit':
        print("\nGoodbye!")
    elif not url:
        print("\n❌ No URL provided.")
    else:
        print(f"\n[INFO] Fetching article from: {url}")
        print("[INFO] This may take a few seconds...\n")
        
        try:
            result = fetch_and_analyze(url, use_transformer=False)
            print_url_result(result)
        except ImportError as e:
            print(f"\n❌ Error: {e}")
            print("[INFO] Install required packages with:")
            print("       pip install -r requirements.txt")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
