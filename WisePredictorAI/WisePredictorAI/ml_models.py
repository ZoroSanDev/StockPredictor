import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockMLModels:
    def __init__(self):
        self.scalers = {
            'features': MinMaxScaler()
        }
        # Enhanced model configurations
        self.models = {
            'RF': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'GB': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        self.feature_importance = {}

    def _add_technical_indicators(self, data):
        """Add enhanced technical indicators as features"""
        df = data.copy()

        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log1p(df['Returns'])

        # Trend indicators
        for window in [7, 14, 21]:
            # RSI with multiple timeframes
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'RSI_{window}'] = 100 - (100 / (1 + rs))

            # Moving averages and trends
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()

            # Volatility
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()

        # MACD with signal line
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Enhanced Bollinger Bands
        for window in [20, 30]:
            sma = df['Close'].rolling(window=window).mean()
            std = df['Close'].rolling(window=window).std()
            df[f'BB_upper_{window}'] = sma + (std * 2)
            df[f'BB_middle_{window}'] = sma
            df[f'BB_lower_{window}'] = sma - (std * 2)
            df[f'BB_width_{window}'] = (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']) / df[f'BB_middle_{window}']

        # Volume-based indicators
        df['Volume_ROC'] = df['Volume'].pct_change()
        df['Volume_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

        # Trend strength indicators
        df['ADX'] = self._calculate_adx(df)

        # Price patterns
        df['Higher_Highs'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Lower_Lows'] = (df['Low'] < df['Low'].shift(1)).astype(int)

        # Fill NaN values with 0
        df = df.fillna(0)
        return df

    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index (ADX)"""
        df = df.copy()

        # True Range
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )

        # Directional Movement
        df['DM_plus'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0),
            0
        )
        df['DM_minus'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0),
            0
        )

        # Smoothed TR and DM
        df['ATR'] = df['TR'].rolling(window=period).mean()
        df['DI_plus'] = 100 * (df['DM_plus'].rolling(window=period).mean() / df['ATR'])
        df['DI_minus'] = 100 * (df['DM_minus'].rolling(window=period).mean() / df['ATR'])

        # ADX
        df['DX'] = 100 * abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'])
        adx = df['DX'].rolling(window=period).mean()

        return adx

    def _prepare_ensemble_data(self, data):
        """Prepare data with enhanced feature engineering"""
        df = self._add_technical_indicators(data)

        # Select relevant features
        feature_columns = [
            'RSI_7', 'RSI_14', 'RSI_21',
            'EMA_7', 'EMA_14', 'EMA_21',
            'SMA_7', 'SMA_14', 'SMA_21',
            'Volatility_7', 'Volatility_14', 'Volatility_21',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_width_20', 'BB_width_30',
            'Volume_ROC', 'Volume_MA_Ratio',
            'ADX', 'Higher_Highs', 'Lower_Lows',
            'Returns', 'Log_Returns'
        ]

        X = df[feature_columns].values
        y = df['Close'].values

        X = self.scalers['features'].fit_transform(X)
        return X, y, feature_columns

    def train(self, data):
        """Train models with cross-validation"""
        try:
            X_ensemble, y_ensemble, feature_columns = self._prepare_ensemble_data(data)

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            for model_name, model in self.models.items():
                model_scores = []
                for train_idx, val_idx in tscv.split(X_ensemble):
                    X_train, X_val = X_ensemble[train_idx], X_ensemble[val_idx]
                    y_train, y_val = y_ensemble[train_idx], y_ensemble[val_idx]

                    # Train model
                    model.fit(X_train, y_train)
                    score = model.score(X_val, y_val)
                    model_scores.append(score)

                # Store feature importance
                self.feature_importance[model_name] = dict(zip(
                    feature_columns,
                    model.feature_importances_
                ))

                logger.info(f"{model_name} CV Scores: {np.mean(model_scores):.4f} (±{np.std(model_scores):.4f})")

            return True

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False

    def predict(self, data, prediction_days):
        """Make predictions using improved ensemble approach"""
        try:
            predictions = {}

            # Prepare features for prediction
            X_ensemble, _, _ = self._prepare_ensemble_data(data)

            # Get predictions from each model
            rf_pred = self.models['RF'].predict(X_ensemble[-prediction_days:])
            gb_pred = self.models['GB'].predict(X_ensemble[-prediction_days:])

            predictions['RF'] = rf_pred.reshape(-1, 1)
            predictions['GB'] = gb_pred.reshape(-1, 1)

            # Dynamic weights based on recent performance
            rf_score = self.models['RF'].score(X_ensemble[-30:], data['Close'].values[-30:])
            gb_score = self.models['GB'].score(X_ensemble[-30:], data['Close'].values[-30:])

            total_score = rf_score + gb_score
            weights = {
                'RF': rf_score / total_score,
                'GB': gb_score / total_score
            }

            final_predictions = np.zeros((prediction_days, 1))
            for model, weight in weights.items():
                final_predictions += predictions[model] * weight

            return final_predictions

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None

    def get_feature_importance(self):
        """Return the feature importance analysis"""
        return self.feature_importance