"""
Basic Analytics Module
Provides basic analytics without heavy ML dependencies.
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class BasicAnalyzer:
    """
    Basic analytics for flight data without heavy dependencies.
    """
    
    def __init__(self):
        pass
    
    def analyze_hourly_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze basic hourly flight patterns.
        
        Args:
            df: Flight schedule DataFrame
            
        Returns:
            DataFrame with hourly statistics
        """
        # Ensure datetime column
        df['Scheduled_Time'] = pd.to_datetime(df['Scheduled_Time'])
        df['Hour'] = df['Scheduled_Time'].dt.hour
        df['Day_of_Week'] = df['Scheduled_Time'].dt.dayofweek
        
        # Calculate hourly metrics
        hourly_stats = df.groupby(['Hour', 'Day_of_Week']).agg({
            'Flight_ID': 'count',
            'Delay_Minutes': ['mean', 'std', 'max'] if 'Delay_Minutes' in df.columns else 'count',
            'Capacity': 'sum' if 'Capacity' in df.columns else 'count',
            'Runway': 'nunique'
        }).round(2)
        
        # Flatten column names
        if isinstance(hourly_stats.columns, pd.MultiIndex):
            hourly_stats.columns = ['_'.join(str(col).strip() for col in cols) for cols in hourly_stats.columns]
        
        hourly_stats = hourly_stats.reset_index()
        
        # Basic congestion scoring
        hourly_stats['Flight_Count'] = hourly_stats.get('Flight_ID_count', hourly_stats.get('Flight_ID', 0))
        hourly_stats['Congestion_Score'] = hourly_stats['Flight_Count'] * 0.1  # Simple metric
        
        return hourly_stats
    
    def get_peak_hours(self, df: pd.DataFrame) -> List[int]:
        """
        Identify peak hours based on flight count.
        
        Args:
            df: Flight schedule DataFrame
            
        Returns:
            List of peak hours
        """
        df['Hour'] = pd.to_datetime(df['Scheduled_Time']).dt.hour
        hourly_counts = df.groupby('Hour').size()
        
        # Get hours with above-average traffic
        avg_flights = hourly_counts.mean()
        peak_hours = hourly_counts[hourly_counts > avg_flights * 1.2].index.tolist()
        
        return peak_hours
    
    def analyze_delays(self, df: pd.DataFrame) -> Dict:
        """
        Basic delay analysis.
        
        Args:
            df: Flight schedule DataFrame
            
        Returns:
            Dictionary with delay statistics
        """
        if 'Delay_Minutes' not in df.columns:
            return {'message': 'No delay data available'}
        
        delayed_flights = df[df['Delay_Minutes'] > 0]
        
        return {
            'total_flights': len(df),
            'delayed_flights': len(delayed_flights),
            'delay_rate': len(delayed_flights) / len(df) * 100 if len(df) > 0 else 0,
            'avg_delay': df['Delay_Minutes'].mean(),
            'max_delay': df['Delay_Minutes'].max(),
            'median_delay': df['Delay_Minutes'].median()
        }
    
    def analyze_runway_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic runway utilization analysis.
        
        Args:
            df: Flight schedule DataFrame
            
        Returns:
            DataFrame with runway statistics
        """
        runway_stats = df.groupby('Runway').agg({
            'Flight_ID': 'count',
            'Delay_Minutes': 'mean' if 'Delay_Minutes' in df.columns else lambda x: 0,
            'Capacity': 'sum' if 'Capacity' in df.columns else lambda x: 0
        }).reset_index()
        
        runway_stats.columns = ['Runway', 'Flight_Count', 'Avg_Delay', 'Total_Passengers']
        runway_stats['Utilization_Score'] = runway_stats['Flight_Count'] / runway_stats['Flight_Count'].max() * 100
        
        return runway_stats
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """
        Generate basic optimization recommendations.
        
        Args:
            df: Flight schedule DataFrame
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Analyze peak hours
        peak_hours = self.get_peak_hours(df)
        if peak_hours:
            recommendations.append(f"Consider redistributing flights from peak hours: {', '.join(map(str, peak_hours))}")
        
        # Analyze delays
        delay_stats = self.analyze_delays(df)
        if delay_stats.get('delay_rate', 0) > 30:
            recommendations.append("High delay rate detected - consider increasing buffer times")
        
        # Analyze runway utilization
        runway_stats = self.analyze_runway_utilization(df)
        overutilized = runway_stats[runway_stats['Utilization_Score'] > 80]
        if not overutilized.empty:
            runways = ', '.join(overutilized['Runway'].tolist())
            recommendations.append(f"High utilization detected on runways: {runways}")
        
        if not recommendations:
            recommendations.append("No major issues detected - schedule appears well optimized")
        
        return recommendations

# Create aliases for compatibility
PeakTimeAnalyzer = BasicAnalyzer
CascadeDelayPredictor = BasicAnalyzer
RunwayOptimizer = BasicAnalyzer
FlightAnomalyDetector = BasicAnalyzer
