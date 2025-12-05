"""
app.py - Flask web application for Fake News Detector

A web interface for the fake news detection system.
Can be deployed to free platforms like Vercel, Heroku, or Google Cloud Run.

Run locally:
    pip install flask
    python app.py

Then visit: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import json
import traceback
import sys
import os
from flask_cors import CORS


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# Initialize detector with error handling
try:
    from fake_news_detector import FakeNewsDetector
    detector = FakeNewsDetector(use_transformer=False)
except Exception as e:
    print(f"ERROR: Failed to load fake_news_detector: {e}", file=sys.stderr)
    traceback.print_exc()
    detector = None

# Initialize URL analyzer
try:
    from url_analyzer import URLArticleAnalyzer
except Exception as e:
    print(f"WARNING: Failed to load url_analyzer: {e}", file=sys.stderr)
    URLArticleAnalyzer = None


@app.route('/')
def index():
    """Home page with analysis forms."""
    return render_template('index.html')


@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """API endpoint for analyzing text."""
    try:
        if not detector:
            return jsonify({'error': 'Detector not initialized. Check server logs.'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if len(text) > 5000:
            return jsonify({'error': 'Text too long (max 5000 characters)'}), 400

        result = detector.predict(text)
        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        print(f"ERROR in analyze_text: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/analyze-url', methods=['POST'])
def analyze_url():
    """API endpoint for analyzing articles from URLs."""
    try:
        if not URLArticleAnalyzer:
            return jsonify({'error': 'URL analyzer not available'}), 500
        
        data = request.get_json()
        url = data.get('url', '').strip()

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        analyzer = URLArticleAnalyzer(use_transformer=False)
        result = analyzer.analyze_url(url)

        return jsonify({
            'success': True,
            'result': result
        })

    except ImportError as e:
        print(f"ERROR: Missing dependencies for URL analysis: {e}", file=sys.stderr)
        return jsonify({'error': f'URL analysis requires: pip install requests beautifulsoup4'}), 500
    except Exception as e:
        print(f"ERROR in analyze_url: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch():
    """API endpoint for batch analysis of multiple texts."""
    try:
        data = request.get_json()
        texts = data.get('texts', [])

        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Invalid text list'}), 400

        if len(texts) > 50:
            return jsonify({'error': 'Too many items (max 50)'}), 400

        results = []
        for text in texts:
            if text.strip():
                result = detector.predict(text.strip())
                results.append({
                    'text': text[:100],
                    'result': result
                })

        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total': len(results),
                'fake': sum(1 for r in results if r['result']['label'] == 'FAKE'),
                'real': sum(1 for r in results if r['result']['label'] == 'REAL'),
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About page with documentation."""
    return render_template('about.html')


@app.route('/api/health')
def health():
    """Health check endpoint for deployment monitoring."""
    return jsonify({'status': 'healthy'})


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Run locally with debug mode
    app.run(debug=True, host='0.0.0.0', port=5000)

CORS(app)  # Enable CORS for all routes
