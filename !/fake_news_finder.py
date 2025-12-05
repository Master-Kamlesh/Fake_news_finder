"""
fake_news_finder.py

Simple CLI to check service health and call the Fake News Detector API.

Usage examples:
    python fake_news_finder.py --health --url https://example.com
    python fake_news_finder.py --text "Some headline to check"
    python fake_news_finder.py --host https://your-service.onrender.com --text "..."
"""

import argparse
import os
import sys
import json
import requests

DEFAULT_HOST = os.environ.get('FAKE_NEWS_SERVICE_URL', 'http://localhost:5000')


def check_health(host):
    url = host.rstrip('/') + '/api/health'
    try:
        r = requests.get(url, timeout=8)
        print(f'Health {r.status_code}:')
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text[:1000])
        return r
    except Exception as e:
        print('Health check failed:', e)
        return None


def analyze_text(host, text):
    url = host.rstrip('/') + '/api/analyze-text'
    payload = {'text': text}
    try:
        r = requests.post(url, json=payload, timeout=12)
        print(f'Response {r.status_code}:')
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text[:2000])
        return r
    except Exception as e:
        print('Request failed:', e)
        return None


def analyze_url(host, target_url):
    url = host.rstrip('/') + '/api/analyze-url'
    payload = {'url': target_url}
    try:
        r = requests.post(url, json=payload, timeout=20)
        print(f'Response {r.status_code}:')
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text[:2000])
        return r
    except Exception as e:
        print('Request failed:', e)
        return None


def build_parser():
    p = argparse.ArgumentParser(prog='fake_news_finder')
    p.add_argument('--host', '-H', default=DEFAULT_HOST, help='Service base URL (default from FAKE_NEWS_SERVICE_URL or localhost:5000)')
    p.add_argument('--health', action='store_true', help='Run health check')
    p.add_argument('--text', '-t', help='Analyze a text string')
    p.add_argument('--url', '-u', dest='target_url', help='Analyze a URL')
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    host = args.host

    if args.health:
        return check_health(host)

    if args.text:
        return analyze_text(host, args.text)

    if args.target_url:
        return analyze_url(host, args.target_url)

    # If nothing provided, print help
    parser.print_help()
    return None


if __name__ == '__main__':
    main()
