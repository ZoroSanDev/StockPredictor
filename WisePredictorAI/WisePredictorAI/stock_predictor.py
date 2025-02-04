import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ml_models import StockMLModels
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self):
        self.ml_models = StockMLModels()

    def predict(self, historical_data, prediction_days):
        """Make predictions using advanced ML models"""
        try:
            # Train models if needed
            if not self.ml_models.train(historical_data):
                logger.error("Failed to train ML models")
                return None

            # Get predictions
            predictions = self.ml_models.predict(historical_data, prediction_days)
            if predictions is None:
                return None

            # Create prediction DataFrame
            last_date = historical_data.index[-1]
            prediction_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=prediction_days
            )

            return pd.DataFrame(
                predictions,
                index=prediction_dates,
                columns=['Predicted']
            )

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return None

    def get_recommendation(self, historical_data, confidence_threshold):
        """Get enhanced trading recommendation with detailed analysis"""
        try:
            recent_data = historical_data.tail(30)
            prediction = self.predict(historical_data, 7)

            if prediction is None:
                return self._create_default_recommendation()

            current_price = recent_data['Close'].iloc[-1]
            predicted_price = prediction['Predicted'].iloc[-1]
            price_change = ((predicted_price - current_price) / current_price) * 100

            # Get feature importance analysis
            feature_importance = self.ml_models.get_feature_importance()

            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(historical_data)

            # Adjust confidence based on multiple factors
            prediction_confidence = self._calculate_confidence(
                price_change,
                trend_strength,
                feature_importance
            )

            # Determine action based on enhanced analysis
            if prediction_confidence >= confidence_threshold:
                if price_change > 0:
                    action = "BUY"
                else:
                    action = "SELL"
            else:
                action = "HOLD"

            return {
                "action": action,
                "confidence": prediction_confidence,
                "predicted_change": price_change,
                "trend_strength": trend_strength,
                "key_factors": self._get_key_factors(feature_importance)
            }

        except Exception as e:
            logger.error(f"Error in getting recommendation: {str(e)}")
            return self._create_default_recommendation()

    def _calculate_trend_strength(self, data):
        """Calculate the strength of the current trend"""
        try:
            returns = data['Close'].pct_change()
            volatility = returns.std()
            momentum = returns.mean()
            trend_direction = 1 if momentum > 0 else -1

            return abs(momentum) / volatility * trend_direction
        except:
            return 0

    def _calculate_confidence(self, price_change, trend_strength, feature_importance):
        """Calculate confidence score based on multiple factors"""
        try:
            # Base confidence from price change
            base_confidence = min(abs(price_change) / 10, 1.0)

            # Adjust based on trend strength
            trend_factor = min(abs(trend_strength), 1.0)

            # Adjust based on feature consistency
            feature_consistency = self._calculate_feature_consistency(feature_importance)

            # Weighted combination of factors
            confidence = (
                0.4 * base_confidence +
                0.3 * trend_factor +
                0.3 * feature_consistency
            )

            return max(0, min(1, confidence))
        except:
            return 0.0

    def _calculate_feature_consistency(self, feature_importance):
        """Calculate how consistent the feature importance is across models"""
        try:
            rf_features = set(sorted(
                feature_importance['RF'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])

            gb_features = set(sorted(
                feature_importance['GB'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])

            # Calculate Jaccard similarity between top features
            common_features = len(rf_features.intersection(gb_features))
            total_features = len(rf_features.union(gb_features))

            return common_features / total_features
        except:
            return 0.5

    def _get_key_factors(self, feature_importance):
        """Extract key factors influencing the prediction"""
        try:
            # Combine importance scores from both models
            combined_importance = {}
            for feature in feature_importance['RF'].keys():
                combined_importance[feature] = (
                    feature_importance['RF'][feature] +
                    feature_importance['GB'][feature]
                ) / 2

            # Get top 5 most important features
            top_features = sorted(
                combined_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            return [f"{feature}: {importance:.3f}" for feature, importance in top_features]
        except:
            return []

    def _create_default_recommendation(self):
        """Create a default recommendation when analysis fails"""
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "predicted_change": 0.0,
            "trend_strength": 0.0,
            "key_factors": []
        }