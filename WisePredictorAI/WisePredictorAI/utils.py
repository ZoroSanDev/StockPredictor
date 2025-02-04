import streamlit as st

def load_css():
    st.markdown("""
        <style>
            .stApp {
                max-width: 100%;
            }
            
            .stock-card {
                background-color: #262730;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            }
            
            .confidence-indicator {
                padding: 0.5rem;
                border-radius: 0.25rem;
                text-align: center;
                font-weight: bold;
                margin: 0.5rem 0;
            }
            
            .buy-indicator {
                background-color: rgba(0, 255, 0, 0.2);
                color: #00ff00;
            }
            
            .sell-indicator {
                background-color: rgba(255, 0, 0, 0.2);
                color: #ff0000;
            }
            
            .hold-indicator {
                background-color: rgba(255, 255, 0, 0.2);
                color: #ffff00;
            }
        </style>
    """, unsafe_allow_html=True)

def create_confidence_indicator(action, confidence):
    color_class = {
        "BUY": "buy-indicator",
        "SELL": "sell-indicator",
        "HOLD": "hold-indicator"
    }
    
    st.markdown(f"""
        <div class="confidence-indicator {color_class[action]}">
            {action} ({confidence:.1%} Confidence)
        </div>
    """, unsafe_allow_html=True)
