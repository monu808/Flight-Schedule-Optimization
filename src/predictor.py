"""
Delay Prediction Model
Uses machine learning to predict flight delays.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class DelayPredictor:
    """Machine learning model to predict flight delays."""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize the delay predictor.
        
        Args:
            model_type: 'xgboost' or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning including congestion factors."""
        df_ml = df.copy()
        
        # Time-based features
        df_ml['Hour'] = pd.to_datetime(df_ml['Scheduled_Time']).dt.hour
        df_ml['Day_of_Week'] = pd.to_datetime(df_ml['Scheduled_Time']).dt.dayofweek
        df_ml['Month'] = pd.to_datetime(df_ml['Scheduled_Time']).dt.month
        df_ml['Is_Weekend'] = df_ml['Day_of_Week'].isin([5, 6]).astype(int)
        
        # Peak hour indicators
        df_ml['Is_Morning_Peak'] = df_ml['Hour'].isin([6, 7, 8, 9]).astype(int)
        df_ml['Is_Evening_Peak'] = df_ml['Hour'].isin([18, 19, 20, 21]).astype(int)
        df_ml['Is_Night'] = df_ml['Hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        
        # Congestion-specific features for Mumbai/Delhi optimization
        if 'Congestion_Factor' in df_ml.columns:
            df_ml['Congestion_Factor'] = df_ml['Congestion_Factor']
        else:
            df_ml['Congestion_Factor'] = 1.0
            
        if 'Peak_Category' in df_ml.columns:
            # Encode peak categories
            peak_mapping = {'super_peak': 4, 'peak': 3, 'moderate': 2, 'low': 1}
            df_ml['Peak_Category_Numeric'] = df_ml['Peak_Category'].map(peak_mapping).fillna(2)
        else:
            df_ml['Peak_Category_Numeric'] = 2
            
        if 'Runway_Efficiency' in df_ml.columns:
            df_ml['Runway_Efficiency'] = df_ml['Runway_Efficiency']
        else:
            df_ml['Runway_Efficiency'] = 1.0
            
        if 'Runway_Capacity' in df_ml.columns:
            df_ml['Runway_Capacity'] = df_ml['Runway_Capacity']
        else:
            df_ml['Runway_Capacity'] = 30
            
        # Airport congestion indicators
        df_ml['Is_Congested_Airport'] = df_ml['Origin'].isin(['BOM', 'DEL']).astype(int)
        
        # Runway utilization features
        if 'Runway_Capacity' in df_ml.columns and 'Hour' in df_ml.columns:
            # Calculate estimated runway utilization
            hourly_flights = df_ml.groupby(['Origin', 'Hour']).size().reset_index(name='Hourly_Flights')
            df_ml = df_ml.merge(hourly_flights, on=['Origin', 'Hour'], how='left')
            df_ml['Runway_Utilization'] = df_ml['Hourly_Flights'] / df_ml['Runway_Capacity']
            df_ml['Runway_Utilization'] = df_ml['Runway_Utilization'].fillna(0.5)
        else:
            df_ml['Runway_Utilization'] = 0.5
        
        # Airline encoding (one-hot)
        airline_dummies = pd.get_dummies(df_ml['Airline'], prefix='Airline')
        df_ml = pd.concat([df_ml, airline_dummies], axis=1)
        
        # Aircraft type encoding
        aircraft_dummies = pd.get_dummies(df_ml['Aircraft_Type'], prefix='Aircraft')
        df_ml = pd.concat([df_ml, aircraft_dummies], axis=1)
        
        # Runway encoding
        runway_dummies = pd.get_dummies(df_ml['Runway'], prefix='Runway')
        df_ml = pd.concat([df_ml, runway_dummies], axis=1)
        
        # Destination encoding (only major ones)
        major_destinations = ['BOM', 'DEL', 'BLR', 'MAA', 'CCU', 'HYD']
        for dest in major_destinations:
            df_ml[f'Dest_{dest}'] = (df_ml['Destination'] == dest).astype(int)
        
        # High impact flight indicator
        df_ml['Is_High_Impact_Int'] = df_ml.get('Is_High_Impact', False).astype(int)
        
        # Capacity-based features
        df_ml['Is_Large_Aircraft'] = (df_ml['Capacity'] > 250).astype(int)
        df_ml['Capacity_Category'] = pd.cut(df_ml['Capacity'], 
                                          bins=[0, 150, 200, 250, 350], 
                                          labels=['Small', 'Medium', 'Large', 'XLarge'])
        capacity_dummies = pd.get_dummies(df_ml['Capacity_Category'], prefix='Cap')
        df_ml = pd.concat([df_ml, capacity_dummies], axis=1)
        
        # Airport congestion features
        if 'Hourly_Flight_Count' in df_ml.columns:
            df_ml['Is_Congested_Hour'] = (df_ml['Hourly_Flight_Count'] > df_ml['Hourly_Flight_Count'].quantile(0.75)).astype(int)
        
        if 'Runway_Hourly_Count' in df_ml.columns:
            df_ml['Is_Runway_Congested'] = (df_ml['Runway_Hourly_Count'] > df_ml['Runway_Hourly_Count'].quantile(0.75)).astype(int)
        
        return df_ml
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for modeling including congestion factors."""
        # Base features
        features = [
            'Hour', 'Day_of_Week', 'Month', 'Is_Weekend',
            'Is_Morning_Peak', 'Is_Evening_Peak', 'Is_Night',
            'Capacity', 'Is_Large_Aircraft', 'Is_High_Impact_Int'
        ]
        
        # Congestion-specific features for Mumbai/Delhi optimization
        congestion_features = [
            'Congestion_Factor', 'Peak_Category_Numeric', 'Runway_Efficiency',
            'Runway_Capacity', 'Is_Congested_Airport', 'Runway_Utilization'
        ]
        features.extend([f for f in congestion_features if f in df.columns])
        
        # Add airline features
        airline_cols = [col for col in df.columns if col.startswith('Airline_')]
        features.extend(airline_cols)
        
        # Add aircraft features
        aircraft_cols = [col for col in df.columns if col.startswith('Aircraft_')]
        features.extend(aircraft_cols)
        
        # Add runway features
        runway_cols = [col for col in df.columns if col.startswith('Runway_')]
        features.extend(runway_cols)
        
        # Add destination features
        dest_cols = [col for col in df.columns if col.startswith('Dest_')]
        features.extend(dest_cols)
        
        # Add capacity features
        cap_cols = [col for col in df.columns if col.startswith('Cap_')]
        features.extend(cap_cols)
        
        # Add traditional congestion features if available
        traditional_congestion_cols = ['Hourly_Flight_Count', 'Runway_Hourly_Count', 
                                     'Is_Congested_Hour', 'Is_Runway_Congested', 'Hourly_Flights']
        for col in traditional_congestion_cols:
            if col in df.columns:
                features.append(col)
        
        # Filter features that actually exist in the dataframe
        features = [f for f in features if f in df.columns]
        
        return features
    
    def train(self, df: pd.DataFrame, target_column: str = 'Delay_Minutes') -> Dict:
        """Train the delay prediction model."""
        # Prepare features
        df_ml = self.prepare_features(df)
        self.feature_columns = self.select_features(df_ml)
        
        # Prepare data
        X = df_ml[self.feature_columns].fillna(0)
        y = df_ml[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y > 0
        )
        
        # Train model
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        else:  # random_forest
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_importance': self.get_feature_importance()
        }
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict delays for new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        df_ml = self.prepare_features(df)
        X = df_ml[self.feature_columns].fillna(0)
        
        return self.model.predict(X)
    
    def predict_delay_probability(self, df: pd.DataFrame, threshold: float = 15.0) -> np.ndarray:
        """Predict probability of delay > threshold minutes."""
        delay_predictions = self.predict(df)
        return (delay_predictions > threshold).astype(float)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            return {}
        
        return dict(zip(self.feature_columns, importances))
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

def create_risk_heatmap(df: pd.DataFrame, predictor: DelayPredictor) -> pd.DataFrame:
    """Create risk heatmap for upcoming time slots."""
    # Predict delays
    delay_predictions = predictor.predict(df)
    delay_probabilities = predictor.predict_delay_probability(df)
    
    # Add predictions to dataframe
    df_risk = df.copy()
    df_risk['Predicted_Delay'] = delay_predictions
    df_risk['Delay_Risk'] = delay_probabilities
    
    # Create hourly risk summary
    df_risk['Hour'] = pd.to_datetime(df_risk['Scheduled_Time']).dt.hour
    df_risk['Date'] = pd.to_datetime(df_risk['Scheduled_Time']).dt.date
    
    risk_summary = df_risk.groupby(['Date', 'Hour']).agg({
        'Delay_Risk': 'mean',
        'Predicted_Delay': 'mean',
        'Flight_ID': 'count'
    }).rename(columns={'Flight_ID': 'Flight_Count'}).reset_index()
    
    # Create risk categories
    risk_summary['Risk_Level'] = pd.cut(
        risk_summary['Delay_Risk'],
        bins=[0, 0.2, 0.4, 0.6, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    return risk_summary

def plot_feature_importance(importance_dict: Dict[str, float], top_n: int = 15):
    """Plot feature importance."""
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    features, importances = zip(*sorted_features)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances for Delay Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def main():
    """Train and evaluate delay prediction model."""
    # Load data
    df = pd.read_csv('data/flight_schedule_data.csv')
    
    print("Training delay prediction model...")
    
    # Train XGBoost model
    predictor = DelayPredictor(model_type='xgboost')
    metrics = predictor.train(df)
    
    print("\n=== Model Performance ===")
    print(f"Test MAE: {metrics['test_mae']:.2f} minutes")
    print(f"Test R²: {metrics['test_r2']:.3f}")
    
    # Save model
    model_path = 'models/delay_predictor.joblib'
    predictor.save_model(model_path)
    
    # Create risk heatmap
    print("\nGenerating risk heatmap...")
    risk_df = create_risk_heatmap(df, predictor)
    risk_df.to_csv('data/delay_risk_heatmap.csv', index=False)
    
    print(f"\n=== Risk Analysis ===")
    print(risk_df['Risk_Level'].value_counts())
    
    # Feature importance
    importance = predictor.get_feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n=== Top 10 Important Features ===")
    for feature, imp in top_features:
        print(f"{feature}: {imp:.3f}")

if __name__ == "__main__":
    main()
