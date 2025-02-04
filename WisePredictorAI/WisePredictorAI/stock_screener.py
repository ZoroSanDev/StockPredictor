import yfinance as yf
import pandas as pd
from openai import OpenAI
import os
from datetime import datetime, timedelta
import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockScreener:

    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get(
            "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"
        ))
        self.model = "gpt-3.5-turbo"
        self.last_api_call = 0
        self.min_delay = 3  # Minimum delay between API calls in seconds

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.min_delay:
            time.sleep(self.min_delay - time_since_last_call)
        self.last_api_call = time.time()

    def get_top_stocks(self, market="^GSPC", max_stocks=5):
        """Analyzes top stocks from a market index"""
        try:
            tickers = self._get_sp500_tickers()
            analysis_results = []

            for idx, ticker in enumerate(tickers[:max_stocks]):
                try:
                    logger.info(
                        f"Analyzing stock {idx + 1}/{max_stocks}: {ticker}")
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    history = stock.history(period="6mo")

                    if not info or history.empty:
                        logger.warning(f"No data available for {ticker}")
                        continue

                    # Add delay between API calls
                    if idx > 0:
                        self._wait_for_rate_limit()

                    analysis = self._analyze_stock(ticker, info, history)
                    if analysis:
                        analysis_results.append(analysis)

                except Exception as e:
                    logger.error(f"Error analyzing {ticker}: {str(e)}")
                    continue

            return sorted(analysis_results,
                          key=lambda x: x.get('confidence_score', 0),
                          reverse=True)

        except Exception as e:
            logger.error(f"Error in get_top_stocks: {str(e)}")
            return []

    def _get_sp500_tickers(self):
        """Get S&P 500 tickers using Yahoo Finance"""
        try:
            # This is a simplified list of top companies
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        except Exception as e:
            logger.error(f"Error getting S&P 500 tickers: {str(e)}")
            return []

    def _analyze_stock(self, ticker, info, history):
        """Analyze individual stock using OpenAI"""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                stock_data = {
                    "ticker": ticker,
                    "current_price": info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    "pe_ratio": info.get('forwardPE', 0),
                    "market_cap": info.get('marketCap', 0),
                    "sector": info.get('sector', ''),
                    "industry": info.get('industry', ''),
                    "price_change_6m": ((history['Close'].iloc[-1] - history['Close'].iloc[0]) / history['Close'].iloc[0] * 100) if not history.empty else 0
                }

                messages = [{
                    "role": "system",
                    "content": """You are a stock market analyst. Analyze this stock data and provide investment recommendations.
                        Format your response EXACTLY as this JSON structure:
                        {
                            "ticker": string (stock symbol),
                            "recommendation": string (one of: "BUY", "SELL", or "HOLD"),
                            "confidence_score": number (between 0 and 1),
                            "analysis_summary": string (2-3 sentences of analysis),
                            "key_factors": array of strings (3-4 key points)
                        }"""
                }, {
                    "role": "user",
                    "content": json.dumps(stock_data)
                }]

                self._wait_for_rate_limit()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )

                result = json.loads(response.choices[0].message.content)

                # Validate response format
                required_fields = ['ticker', 'recommendation', 'confidence_score', 'analysis_summary', 'key_factors']
                if not all(field in result for field in required_fields):
                    logger.error(f"Invalid response format for {ticker}: missing required fields")
                    return None

                # Ensure recommendation is uppercase and valid
                result['recommendation'] = result['recommendation'].upper()
                if result['recommendation'] not in ['BUY', 'SELL', 'HOLD']:
                    result['recommendation'] = 'HOLD'

                # Ensure confidence score is between 0 and 1
                result['confidence_score'] = max(0, min(1, float(result['confidence_score'])))

                return result

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {ticker} (attempt {attempt + 1}): {str(e)}")
            except Exception as e:
                logger.error(f"Error in _analyze_stock for {ticker} (attempt {attempt + 1}): {str(e)}")

            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

        return None

    def get_market_summary(self):
        """Get overall market analysis and recommendations"""
        try:
            self._wait_for_rate_limit(
            )  # Rate limit for market summary request

            sp500 = yf.Ticker("^GSPC")
            dow = yf.Ticker("^DJI")
            nasdaq = yf.Ticker("^IXIC")

            market_data = {
                "sp500": self._calculate_metrics(sp500.history(period="1mo")),
                "dow": self._calculate_metrics(dow.history(period="1mo")),
                "nasdaq": self._calculate_metrics(nasdaq.history(period="1mo"))
            }

            messages = [{
                "role":
                "system",
                "content":
                """You are a market analyst. Analyze the market data and provide a summary.
                    Format your response exactly as a JSON object with these fields:
                    {
                        "market_sentiment": string,
                        "confidence_score": number,
                        "key_trends": array
                    }"""
            }, {
                "role": "user",
                "content": json.dumps(market_data)
            }]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"})

            result = response.choices[0].message.content
            return json.loads(result)

        except Exception as e:
            logger.error(f"Error in get_market_summary: {str(e)}")
            return None

    def _calculate_metrics(self, data):
        """Calculate key metrics for market analysis"""
        if data.empty:
            return {}

        return {
            "price_change": ((data['Close'].iloc[-1] - data['Close'].iloc[0]) /
                             data['Close'].iloc[0] * 100),
            "volatility":
            data['Close'].std(),
            "volume_trend":
            data['Volume'].mean(),
            "last_price":
            data['Close'].iloc[-1]
        }