import requests
import time
import logging
from typing import List, Dict

KEYWORDS = ["Tesla", "Apple", "Google", "Microsoft", "pharma"]
INTERVAL = 60  # 1 minute
API_KEY_FILE = "Scribe/NewsFetcher/api_keys.txt"
OUTPUT_FILE = "Scribe/NewsFetcher/news_output.txt"

def setup_logger(log_file: str = 'Scribe/NewsFetcher/news_fetcher.log'):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s'
    )

def get_api_key(service_name, path=API_KEY_FILE):
    with open(path) as f:
        for line in f:
            if line.startswith(service_name):
                return line.split('-')[1].strip()
    raise ValueError(f"API key for {service_name} not found.")

def fetch_newsapi_news(api_key, keywords, page_size=20):
    url = 'https://newsapi.org/v2/everything'
    query = ' OR '.join(keywords)
    params = {
        'q': query,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': page_size,
        'apiKey': api_key
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 429:
            print('Rate limit reached. Pausing.')
            time.sleep(60)
            return []
        response.raise_for_status()
        data = response.json()
        return data.get('articles', [])
    except Exception as e:
        print(f'Failed to fetch news: {e}')
        return []

def print_and_save_articles(articles, seen_urls):
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for art in articles:
            if art.get('url') in seen_urls:
                continue
            #line = f"[{art.get('publishedAt', '')}] {art.get('source', {}).get('name', '')} - {art.get('title', '')}\nURL: {art.get('url', '')}\nContent: {art.get('content', '')}\n{'-'*80}\n"
            line = f"[{art.get('publishedAt', '')}] {art.get('source', {}).get('name', '')} - {art.get('title', '')}\n"
            print(line, end='')
            f.write(line)
            seen_urls.add(art.get('url'))

def main():
    setup_logger()
    api_key = get_api_key('NewsAPI')
    seen_urls = set()
    while True:
        articles = fetch_newsapi_news(api_key, KEYWORDS)
        print_and_save_articles(articles, seen_urls)
        time.sleep(INTERVAL)

if __name__ == '__main__':
    main() 