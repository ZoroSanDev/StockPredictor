import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from stock_predictor import StockPredictor
from data_processor import DataProcessor
from stock_screener import StockScreener
from utils import load_css, create_confidence_indicator
import logging

# Configure basic logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Initialize session state
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = 'AAPL'
if 'screener' not in st.session_state:
    st.session_state.screener = StockScreener()

# Header
st.title("🤖 AI-Powered Stock Prediction Dashboard")

# Sidebar
with st.sidebar:
    st.header("Stock Selection")
    stock_symbol = st.text_input("Enter Stock Symbol", st.session_state.selected_stock)
    st.session_state.selected_stock = stock_symbol.upper()

    st.header("Analysis Parameters")
    prediction_days = st.slider("Prediction Days", 1, 30, 7)
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7)

    # Market Analysis Section in Sidebar
    st.header("Market Analysis")
    if st.button("Update Market Analysis"):
        with st.spinner("Analyzing market conditions..."):
            market_summary = st.session_state.screener.get_market_summary()
            if market_summary:
                st.session_state.market_summary = market_summary

    if 'market_summary' in st.session_state:
        ms = st.session_state.market_summary
        st.markdown(f"**Market Sentiment:** {ms['market_sentiment']}")
        st.markdown(f"**Confidence:** {ms['confidence_score']:.1%}")
        with st.expander("Market Trends"):
            for trend in ms['key_trends']:
                st.markdown(f"• {trend}")

# Main content
try:
    # Initialize processors
    data_processor = DataProcessor(st.session_state.selected_stock)
    stock_predictor = StockPredictor()

    # Create tabs
    tab1, tab2 = st.tabs(["Stock Analysis", "AI Stock Screener"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Stock price chart
            st.subheader("Stock Price Analysis")
            fig = go.Figure()

            # Historical data
            historical_data = data_processor.get_historical_data()
            fig.add_trace(go.Candlestick(
                x=historical_data.index,
                open=historical_data['Open'],
                high=historical_data['High'],
                low=historical_data['Low'],
                close=historical_data['Close'],
                name='Market Data'
            ))

            # Prediction
            prediction_data = stock_predictor.predict(historical_data, prediction_days)
            fig.add_trace(go.Scatter(
                x=prediction_data.index,
                y=prediction_data['Predicted'],
                mode='lines',
                line=dict(color='rgba(255, 75, 75, 0.8)'),
                name='AI Prediction'
            ))

            fig.update_layout(
                template='plotly_dark',
                xaxis_rangeslider_visible=False,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Company info and metrics
            stock_info = data_processor.get_stock_info()
            st.subheader("Company Overview")

            metrics_container = st.container()
            with metrics_container:
                st.metric(
                    "Current Price",
                    f"${stock_info['currentPrice']:.2f}",
                    f"{stock_info['dayChange']:.2f}%"
                )

                # AI Recommendation
                recommendation = stock_predictor.get_recommendation(
                    historical_data,
                    confidence_threshold
                )

                st.markdown("### AI Recommendation")
                create_confidence_indicator(
                    recommendation['action'],
                    recommendation['confidence']
                )

                st.markdown("### Key Metrics")
                st.markdown(f"**Market Cap:** ${stock_info['marketCap']:,.0f}")
                st.markdown(f"**P/E Ratio:** {stock_info['peRatio']:.2f}")
                st.markdown(f"**52W High:** ${stock_info['yearHigh']:.2f}")
                st.markdown(f"**52W Low:** ${stock_info['yearLow']:.2f}")

                with st.expander("Company Information"):
                    st.markdown(f"**Sector:** {stock_info['sector']}")
                    st.markdown(f"**Industry:** {stock_info['industry']}")
                    st.markdown(stock_info['longBusinessSummary'])

    with tab2:
        st.subheader("AI Stock Screener")
        if st.button("Scan for Top Stock Picks"):
            with st.spinner("Analyzing market data..."):
                try:
                    top_stocks = st.session_state.screener.get_top_stocks()

                    if not top_stocks:
                        st.warning("No stock recommendations available at the moment. Please try again later.")
                    else:
                        for stock in top_stocks:
                            if not isinstance(stock, dict):
                                logger.error(f"Invalid stock data format: {stock}")
                                continue

                            recommendation = stock.get('recommendation', 'HOLD')
                            confidence = stock.get('confidence_score', 0.0)
                            analysis = stock.get('analysis_summary', 'No analysis available')
                            factors = stock.get('key_factors', [])
                            ticker = stock.get('ticker', 'Unknown')

                            with st.expander(f"{ticker} - {recommendation}"):
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(analysis)
                                    st.markdown("### Key Factors")
                                    for factor in factors:
                                        st.markdown(f"• {factor}")
                                with col2:
                                    st.markdown(f"**Confidence Score:** {confidence:.1%}")
                                    create_confidence_indicator(recommendation, confidence)
                except Exception as e:
                    st.error(f"Error analyzing stocks: {str(e)}")
                    logger.error(f"Stock analysis error: {str(e)}")

    # News and Developments
    st.subheader("Recent News & Developments")
    news_data = data_processor.get_news()
    for news in news_data[:5]:
        with st.expander(f"{news['title']} - Published: {news['publishedDate']}"):
            st.write(news['summary'])
            if 'link' in news and news['link']:
                st.markdown(f"[Read more]({news['link']})")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please check the stock symbol and try again.")