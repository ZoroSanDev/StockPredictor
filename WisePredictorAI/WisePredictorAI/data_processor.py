import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)

    def get_historical_data(self, period='1y'):
        try:
            return self.stock.history(period=period)
        except Exception as e:
            logger.error(f"Error fetching historical data for {self.symbol}: {str(e)}")
            return pd.DataFrame()

    def get_stock_info(self):
        try:
            info = self.stock.info
            return {
                'currentPrice': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'dayChange': info.get('regularMarketChangePercent', 0),
                'marketCap': info.get('marketCap', 0),
                'peRatio': info.get('forwardPE', 0),
                'yearHigh': info.get('fiftyTwoWeekHigh', 0),
                'yearLow': info.get('fiftyTwoWeekLow', 0),
                'longBusinessSummary': info.get('longBusinessSummary', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', '')
            }
        except Exception as e:
            logger.error(f"Error fetching stock info for {self.symbol}: {str(e)}")
            return {}

    def get_news(self):
        """Get news for the stock with improved error handling and retry logic"""
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # Get news from Yahoo Finance
                news_items = self.stock.news

                if not news_items:
                    logger.info(f"No news found for {self.symbol}")
                    return self._create_default_news("No recent news available")

                processed_news = []
                for item in news_items:
                    # Verify we have valid news data
                    if not isinstance(item, dict):
                        continue

                    # Extract news details with fallbacks
                    title = item.get('title', '')
                    summary = item.get('summary', '')

                    # Skip invalid news items
                    if not title or not summary:
                        continue

                    # Get additional details
                    publisher = item.get('publisher', 'Unknown Publisher')
                    link = item.get('link', f'https://finance.yahoo.com/quote/{self.symbol}')

                    # Format timestamp
                    timestamp = item.get('providerPublishTime', time.time())
                    if isinstance(timestamp, (int, float)):
                        published_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        published_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    processed_news.append({
                        'title': title,
                        'summary': summary,
                        'publisher': publisher,
                        'link': link,
                        'publishedDate': published_date
                    })

                if processed_news:
                    return processed_news

                return self._create_default_news("No recent news available")

            except Exception as e:
                logger.error(f"Error fetching news (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return self._create_default_news(f"Error fetching news: Please try again later")

    def _create_default_news(self, message):
        """Create a default news item when no news is available"""
        return [{
            'title': message,
            'summary': 'Check financial news websites for the latest updates.',
            'publisher': 'System Message',
            'link': f'https://finance.yahoo.com/quote/{self.symbol}/news',
            'publishedDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }]