"""
Weather-Based Runway Capacity Reduction Module
Implements dynamic runway capacity adjustments based on weather conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random

class WeatherCondition(Enum):
    """Weather condition categories affecting runway operations."""
    CLEAR = "Clear"
    LIGHT_RAIN = "Light Rain"
    MODERATE_RAIN = "Moderate Rain"
    HEAVY_RAIN = "Heavy Rain"
    FOG = "Fog"
    THUNDERSTORM = "Thunderstorm"
    STRONG_WINDS = "Strong Winds"
    SNOW = "Snow"

@dataclass
class WeatherImpact:
    """Weather impact on runway operations."""
    condition: WeatherCondition
    visibility_km: float
    wind_speed_kts: int
    precipitation_mm: float
    capacity_reduction_factor: float  # 0.0 to 1.0 (1.0 = no reduction)
    landing_restrictions: bool
    takeoff_restrictions: bool
    crosswind_runway_closure: bool

class WeatherRunwayOptimizer:
    """
    Manages runway capacity dynamically based on weather conditions.
    Implements realistic weather impact on airport operations.
    """
    
    def __init__(self):
        """Initialize weather-based runway optimizer."""
        self.weather_impacts = self._initialize_weather_impacts()
        self.runway_weather_sensitivity = self._initialize_runway_sensitivity()
        self.current_weather_conditions = {}
        
    def _initialize_weather_impacts(self) -> Dict[WeatherCondition, WeatherImpact]:
        """Define how different weather conditions affect runway capacity."""
        return {
            WeatherCondition.CLEAR: WeatherImpact(
                condition=WeatherCondition.CLEAR,
                visibility_km=10.0,
                wind_speed_kts=5,
                precipitation_mm=0.0,
                capacity_reduction_factor=1.0,  # No reduction
                landing_restrictions=False,
                takeoff_restrictions=False,
                crosswind_runway_closure=False
            ),
            WeatherCondition.LIGHT_RAIN: WeatherImpact(
                condition=WeatherCondition.LIGHT_RAIN,
                visibility_km=8.0,
                wind_speed_kts=8,
                precipitation_mm=2.5,
                capacity_reduction_factor=0.90,  # 10% reduction
                landing_restrictions=False,
                takeoff_restrictions=False,
                crosswind_runway_closure=False
            ),
            WeatherCondition.MODERATE_RAIN: WeatherImpact(
                condition=WeatherCondition.MODERATE_RAIN,
                visibility_km=5.0,
                wind_speed_kts=12,
                precipitation_mm=7.5,
                capacity_reduction_factor=0.75,  # 25% reduction
                landing_restrictions=True,
                takeoff_restrictions=False,
                crosswind_runway_closure=False
            ),
            WeatherCondition.HEAVY_RAIN: WeatherImpact(
                condition=WeatherCondition.HEAVY_RAIN,
                visibility_km=2.0,
                wind_speed_kts=15,
                precipitation_mm=15.0,
                capacity_reduction_factor=0.60,  # 40% reduction
                landing_restrictions=True,
                takeoff_restrictions=True,
                crosswind_runway_closure=True
            ),
            WeatherCondition.FOG: WeatherImpact(
                condition=WeatherCondition.FOG,
                visibility_km=0.8,
                wind_speed_kts=3,
                precipitation_mm=0.0,
                capacity_reduction_factor=0.50,  # 50% reduction
                landing_restrictions=True,
                takeoff_restrictions=True,
                crosswind_runway_closure=False
            ),
            WeatherCondition.THUNDERSTORM: WeatherImpact(
                condition=WeatherCondition.THUNDERSTORM,
                visibility_km=3.0,
                wind_speed_kts=25,
                precipitation_mm=20.0,
                capacity_reduction_factor=0.30,  # 70% reduction
                landing_restrictions=True,
                takeoff_restrictions=True,
                crosswind_runway_closure=True
            ),
            WeatherCondition.STRONG_WINDS: WeatherImpact(
                condition=WeatherCondition.STRONG_WINDS,
                visibility_km=10.0,
                wind_speed_kts=30,
                precipitation_mm=0.0,
                capacity_reduction_factor=0.70,  # 30% reduction
                landing_restrictions=True,
                takeoff_restrictions=False,
                crosswind_runway_closure=True
            ),
            WeatherCondition.SNOW: WeatherImpact(
                condition=WeatherCondition.SNOW,
                visibility_km=1.5,
                wind_speed_kts=10,
                precipitation_mm=8.0,
                capacity_reduction_factor=0.40,  # 60% reduction
                landing_restrictions=True,
                takeoff_restrictions=True,
                crosswind_runway_closure=True
            )
        }
    
    def _initialize_runway_sensitivity(self) -> Dict[str, Dict[str, float]]:
        """Define runway-specific weather sensitivity factors."""
        return {
            '09R/27L': {
                'wind_sensitivity': 0.8,      # Less sensitive to crosswinds
                'rain_sensitivity': 0.9,      # Good drainage
                'fog_sensitivity': 1.0,       # Standard fog impact
                'length_advantage': 1.1       # Longer runway advantage
            },
            '09L/27R': {
                'wind_sensitivity': 0.8,      # Less sensitive to crosswinds
                'rain_sensitivity': 0.9,      # Good drainage
                'fog_sensitivity': 1.0,       # Standard fog impact
                'length_advantage': 1.1       # Longer runway advantage
            },
            '14/32': {
                'wind_sensitivity': 1.3,      # More sensitive (crosswind runway)
                'rain_sensitivity': 1.2,      # Poorer drainage
                'fog_sensitivity': 1.1,       # Slightly more affected
                'length_advantage': 0.9       # Shorter runway disadvantage
            }
        }
    
    def generate_weather_conditions(self, start_date: str, days: int = 7) -> pd.DataFrame:
        """
        Generate realistic weather patterns for the given period.
        
        Args:
            start_date: Start date string in format 'YYYY-MM-DD'
            days: Number of days to generate weather for
            
        Returns:
            DataFrame with hourly weather conditions
        """
        weather_data = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Monsoon/weather probabilities for Indian airports
        weather_probabilities = {
            WeatherCondition.CLEAR: 0.50,
            WeatherCondition.LIGHT_RAIN: 0.20,
            WeatherCondition.MODERATE_RAIN: 0.12,
            WeatherCondition.HEAVY_RAIN: 0.06,
            WeatherCondition.FOG: 0.05,
            WeatherCondition.THUNDERSTORM: 0.04,
            WeatherCondition.STRONG_WINDS: 0.02,
            WeatherCondition.SNOW: 0.01  # Rare in most Indian airports
        }
        
        for day in range(days):
            current_date = start_dt + timedelta(days=day)
            
            # Generate daily weather pattern (weather tends to persist)
            daily_base_weather = random.choices(
                list(weather_probabilities.keys()),
                weights=list(weather_probabilities.values())
            )[0]
            
            for hour in range(24):
                current_time = current_date.replace(hour=hour)
                
                # Weather can change during the day, but with persistence
                if random.random() < 0.15:  # 15% chance of weather change each hour
                    current_weather = random.choices(
                        list(weather_probabilities.keys()),
                        weights=list(weather_probabilities.values())
                    )[0]
                else:
                    current_weather = daily_base_weather
                
                weather_impact = self.weather_impacts[current_weather]
                
                # Add some randomness to weather parameters
                visibility_variance = random.uniform(0.8, 1.2)
                wind_variance = random.uniform(0.7, 1.3)
                precipitation_variance = random.uniform(0.5, 1.5)
                
                weather_data.append({
                    'Datetime': current_time,
                    'Hour': hour,
                    'Weather_Condition': current_weather.value,
                    'Visibility_km': weather_impact.visibility_km * visibility_variance,
                    'Wind_Speed_kts': int(weather_impact.wind_speed_kts * wind_variance),
                    'Precipitation_mm': weather_impact.precipitation_mm * precipitation_variance,
                    'Capacity_Reduction_Factor': weather_impact.capacity_reduction_factor,
                    'Landing_Restrictions': weather_impact.landing_restrictions,
                    'Takeoff_Restrictions': weather_impact.takeoff_restrictions,
                    'Crosswind_Runway_Closure': weather_impact.crosswind_runway_closure
                })
        
        return pd.DataFrame(weather_data)
    
    def apply_weather_impact_to_schedule(self, flight_df: pd.DataFrame, 
                                       weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply weather-based runway capacity reductions to flight schedule.
        
        Args:
            flight_df: Flight schedule DataFrame
            weather_df: Weather conditions DataFrame
            
        Returns:
            Modified flight DataFrame with weather impacts
        """
        # Merge flight schedule with weather data
        flight_df['Scheduled_Time'] = pd.to_datetime(flight_df['Scheduled_Time'])
        weather_df['Datetime'] = pd.to_datetime(weather_df['Datetime'])
        
        # Create hour column for merging
        flight_df['Weather_Hour'] = flight_df['Scheduled_Time'].dt.floor('H')
        weather_df['Weather_Hour'] = weather_df['Datetime'].dt.floor('H')
        
        # Merge on the hour
        merged_df = flight_df.merge(
            weather_df[['Weather_Hour', 'Weather_Condition', 'Capacity_Reduction_Factor', 
                       'Landing_Restrictions', 'Takeoff_Restrictions', 'Crosswind_Runway_Closure']],
            on='Weather_Hour',
            how='left'
        )
        
        # Fill missing weather data with clear conditions
        merged_df['Weather_Condition'].fillna('Clear', inplace=True)
        merged_df['Capacity_Reduction_Factor'].fillna(1.0, inplace=True)
        merged_df['Landing_Restrictions'].fillna(False, inplace=True)
        merged_df['Takeoff_Restrictions'].fillna(False, inplace=True)
        merged_df['Crosswind_Runway_Closure'].fillna(False, inplace=True)
        
        # Apply runway-specific weather sensitivity
        merged_df['Weather_Adjusted_Runway_Capacity'] = merged_df.apply(
            self._calculate_weather_adjusted_capacity, axis=1
        )
        
        # Calculate weather-induced delays
        merged_df['Weather_Delay_Minutes'] = merged_df.apply(
            self._calculate_weather_delay, axis=1
        )
        
        # Update total delay
        merged_df['Total_Delay_Minutes'] = (
            merged_df.get('Delay_Minutes', 0) + merged_df['Weather_Delay_Minutes']
        )
        
        # Flag flights that may need runway reassignment
        merged_df['Needs_Runway_Reassignment'] = (
            (merged_df['Runway'] == '14/32') & merged_df['Crosswind_Runway_Closure']
        )
        
        return merged_df
    
    def _calculate_weather_adjusted_capacity(self, row) -> int:
        """Calculate weather-adjusted runway capacity for a specific flight."""
        base_capacity = row['Runway_Capacity']
        weather_reduction = row['Capacity_Reduction_Factor']
        runway_id = row['Runway']
        weather_condition = row['Weather_Condition']
        
        # Get runway-specific sensitivity
        sensitivity = self.runway_weather_sensitivity.get(runway_id, {
            'wind_sensitivity': 1.0,
            'rain_sensitivity': 1.0,
            'fog_sensitivity': 1.0,
            'length_advantage': 1.0
        })
        
        # Apply runway-specific adjustments
        if 'Rain' in weather_condition:
            weather_reduction *= (2.0 - sensitivity['rain_sensitivity'])
        elif 'Fog' in weather_condition:
            weather_reduction *= (2.0 - sensitivity['fog_sensitivity'])
        elif 'Wind' in weather_condition:
            weather_reduction *= (2.0 - sensitivity['wind_sensitivity'])
        
        # Apply length advantage for severe weather
        if weather_reduction < 0.7:  # Severe weather
            weather_reduction *= sensitivity['length_advantage']
        
        # Ensure minimum capacity
        final_capacity = max(int(base_capacity * weather_reduction), 5)
        
        return final_capacity
    
    def _calculate_weather_delay(self, row) -> float:
        """Calculate additional delay due to weather conditions."""
        weather_condition = row['Weather_Condition']
        base_delay = row.get('Delay_Minutes', 0)
        
        # Weather delay factors (additional minutes)
        weather_delays = {
            'Clear': 0,
            'Light Rain': 2,
            'Moderate Rain': 8,
            'Heavy Rain': 15,
            'Fog': 20,
            'Thunderstorm': 30,
            'Strong Winds': 10,
            'Snow': 25
        }
        
        additional_delay = weather_delays.get(weather_condition, 0)
        
        # Add randomness to delay
        delay_variance = random.uniform(0.5, 1.5)
        final_delay = additional_delay * delay_variance
        
        # Increase delay for restricted operations
        if row['Landing_Restrictions'] or row['Takeoff_Restrictions']:
            final_delay *= 1.5
        
        return final_delay
    
    def analyze_weather_impact(self, weather_affected_df: pd.DataFrame) -> Dict:
        """
        Analyze the overall impact of weather on flight operations.
        
        Args:
            weather_affected_df: DataFrame with weather impact applied
            
        Returns:
            Analysis results dictionary
        """
        # Calculate weather impact statistics
        total_flights = len(weather_affected_df)
        weather_affected_flights = len(weather_affected_df[weather_affected_df['Weather_Condition'] != 'Clear'])
        
        # Calculate capacity reductions
        avg_capacity_reduction = 1 - weather_affected_df['Capacity_Reduction_Factor'].mean()
        
        # Calculate additional delays
        total_weather_delay = weather_affected_df['Weather_Delay_Minutes'].sum()
        avg_weather_delay_per_flight = weather_affected_df['Weather_Delay_Minutes'].mean()
        
        # Runway closure analysis
        crosswind_closures = weather_affected_df['Crosswind_Runway_Closure'].sum()
        flights_needing_reassignment = weather_affected_df['Needs_Runway_Reassignment'].sum()
        
        # Weather condition distribution
        weather_distribution = weather_affected_df['Weather_Condition'].value_counts()
        
        # Hourly impact analysis
        hourly_impact = weather_affected_df.groupby('Hour').agg({
            'Weather_Delay_Minutes': 'mean',
            'Capacity_Reduction_Factor': 'mean',
            'Weather_Condition': lambda x: (x != 'Clear').sum()
        }).round(2)
        
        return {
            'total_flights': total_flights,
            'weather_affected_flights': weather_affected_flights,
            'weather_impact_percentage': (weather_affected_flights / total_flights) * 100,
            'avg_capacity_reduction_percentage': avg_capacity_reduction * 100,
            'total_weather_delay_minutes': total_weather_delay,
            'avg_weather_delay_per_flight': avg_weather_delay_per_flight,
            'crosswind_runway_closures': crosswind_closures,
            'flights_needing_reassignment': flights_needing_reassignment,
            'weather_distribution': weather_distribution.to_dict(),
            'hourly_impact': hourly_impact
        }
    
    def get_weather_recommendations(self, weather_affected_df: pd.DataFrame) -> List[Dict]:
        """
        Generate operational recommendations based on weather conditions.
        
        Args:
            weather_affected_df: DataFrame with weather impact applied
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Analyze severe weather periods
        severe_weather_mask = weather_affected_df['Capacity_Reduction_Factor'] < 0.7
        severe_weather_periods = weather_affected_df[severe_weather_mask]
        
        if not severe_weather_periods.empty:
            for hour in severe_weather_periods['Hour'].unique():
                hour_data = severe_weather_periods[severe_weather_periods['Hour'] == hour]
                
                recommendations.append({
                    'type': 'Weather Alert',
                    'hour': hour,
                    'condition': hour_data['Weather_Condition'].iloc[0],
                    'affected_flights': len(hour_data),
                    'recommendation': f"Consider redistributing {len(hour_data)} flights from hour {hour} due to {hour_data['Weather_Condition'].iloc[0]}",
                    'capacity_impact': f"{(1 - hour_data['Capacity_Reduction_Factor'].mean()) * 100:.1f}% capacity reduction"
                })
        
        # Runway reassignment recommendations
        reassignment_needed = weather_affected_df[weather_affected_df['Needs_Runway_Reassignment']]
        if not reassignment_needed.empty:
            recommendations.append({
                'type': 'Runway Reassignment',
                'affected_flights': len(reassignment_needed),
                'recommendation': f"Reassign {len(reassignment_needed)} flights from crosswind runway 14/32 to main runways",
                'alternative_runways': ['09R/27L', '09L/27R']
            })
        
        # Peak weather impact hours
        hourly_delays = weather_affected_df.groupby('Hour')['Weather_Delay_Minutes'].sum()
        high_delay_hours = hourly_delays[hourly_delays > hourly_delays.quantile(0.8)].index.tolist()
        
        if high_delay_hours:
            recommendations.append({
                'type': 'Schedule Optimization',
                'high_delay_hours': high_delay_hours,
                'recommendation': f"Consider avoiding scheduling high-priority flights during hours: {high_delay_hours}",
                'alternative_suggestion': "Redistribute flights to hours with better weather conditions"
            })
        
        return recommendations
