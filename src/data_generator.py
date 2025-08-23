"""
Flight Schedule Data Generator
Generates realistic synthetic flight data for the optimization system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict
import os

class FlightDataGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the flight data generator with a random seed."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Congested Indian airports focus (Mumbai/Delhi patterns)
        self.airlines = ['AI', '6E', 'UK', 'SG', 'G8', 'QP', 'I5', '9W', 'VT', 'IX']
        
        # Major Indian destinations with congestion levels
        self.destinations = {
            # Tier 1 - Highly congested airports
            'BOM': {'congestion_factor': 1.8, 'priority': 'high'},  # Mumbai
            'DEL': {'congestion_factor': 1.9, 'priority': 'high'},  # Delhi
            'BLR': {'congestion_factor': 1.6, 'priority': 'high'},  # Bangalore
            
            # Tier 2 - Moderately congested
            'MAA': {'congestion_factor': 1.4, 'priority': 'medium'},  # Chennai
            'CCU': {'congestion_factor': 1.3, 'priority': 'medium'},  # Kolkata
            'HYD': {'congestion_factor': 1.5, 'priority': 'medium'},  # Hyderabad
            
            # Tier 3 - Regular traffic
            'AMD': {'congestion_factor': 1.1, 'priority': 'normal'},  # Ahmedabad
            'COK': {'congestion_factor': 1.2, 'priority': 'normal'},  # Kochi
            'GOI': {'congestion_factor': 0.9, 'priority': 'normal'},  # Goa
            'PNQ': {'congestion_factor': 1.3, 'priority': 'medium'},  # Pune
            'JAI': {'congestion_factor': 1.0, 'priority': 'normal'},  # Jaipur
            'LKO': {'congestion_factor': 1.1, 'priority': 'normal'},  # Lucknow
        }
        
        # Mumbai/Delhi runway configuration (high capacity but congested)
        self.runways = {
            '09R/27L': {'capacity': 35, 'efficiency': 0.85},  # Main runway
            '09L/27R': {'capacity': 32, 'efficiency': 0.90},  # Secondary runway  
            '14/32': {'capacity': 28, 'efficiency': 0.80},    # Cross runway (weather dependent)
        }
        
        # Aircraft types with Mumbai/Delhi operational patterns
        self.aircraft_types = {
            # Domestic workhorses (high frequency)
            'A320': {'capacity': 180, 'delay_prob': 0.18, 'frequency': 0.35, 'turnaround': 45},
            'B737': {'capacity': 160, 'delay_prob': 0.15, 'frequency': 0.25, 'turnaround': 40},
            'A321': {'capacity': 220, 'delay_prob': 0.22, 'frequency': 0.15, 'turnaround': 50},
            
            # International/premium (medium frequency, higher impact)
            'B777': {'capacity': 300, 'delay_prob': 0.25, 'frequency': 0.10, 'turnaround': 90},
            'A350': {'capacity': 280, 'delay_prob': 0.20, 'frequency': 0.08, 'turnaround': 85},
            'B787': {'capacity': 250, 'delay_prob': 0.18, 'frequency': 0.07, 'turnaround': 80},
        }
        
        # Peak traffic patterns for congested airports
        self.peak_patterns = {
            'super_peak': [7, 8, 9, 19, 20, 21],      # Extreme congestion
            'peak': [6, 10, 18, 22],                   # High congestion  
            'moderate': [5, 11, 12, 13, 14, 17, 23],   # Moderate traffic
            'low': [0, 1, 2, 3, 4, 15, 16],           # Low traffic
        }
    
    def generate_base_schedule(self, start_date: str, days: int = 7) -> pd.DataFrame:
        """Generate base flight schedule without delays."""
        flights = []
        flight_id = 1000
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        
        for day in range(days):
            current_date = start_dt + timedelta(days=day)
            
            # Generate flights throughout the day with realistic patterns
            # Peak hours: 6-9 AM, 12-2 PM, 6-9 PM
            hourly_flights = self._get_hourly_flight_distribution()
            
            for hour, num_flights in hourly_flights.items():
                for flight_num in range(num_flights):
                    # Random minute within the hour
                    minute = random.randint(0, 59)
                    scheduled_time = current_date.replace(hour=hour, minute=minute)
                    
                    # Generate flight details with congestion factors
                    airline = random.choice(self.airlines)
                    
                    # Select destination with weighted probability (favor congested routes)
                    destinations = list(self.destinations.keys())
                    weights = [self.destinations[dest]['congestion_factor'] for dest in destinations]
                    destination = random.choices(destinations, weights=weights)[0]
                    
                    # Select aircraft type based on frequency patterns
                    aircraft_types = list(self.aircraft_types.keys())
                    frequencies = [self.aircraft_types[ac]['frequency'] for ac in aircraft_types]
                    aircraft_type = random.choices(aircraft_types, weights=frequencies)[0]
                    
                    # Select runway (weighted by capacity and efficiency)
                    runways = list(self.runways.keys())
                    runway_weights = [self.runways[rw]['capacity'] * self.runways[rw]['efficiency'] for rw in runways]
                    runway = random.choices(runways, weights=runway_weights)[0]
                    
                    aircraft_id = f"{aircraft_type}-{random.randint(100, 999)}"
                    
                    flights.append({
                        'Flight_ID': f"{airline}{flight_id}",
                        'Airline': airline,
                        'Scheduled_Time': scheduled_time,
                        'Destination': destination,
                        'Destination_Priority': self.destinations[destination]['priority'],
                        'Congestion_Factor': self.destinations[destination]['congestion_factor'],
                        'Aircraft_Type': aircraft_type,
                        'Aircraft_ID': aircraft_id,
                        'Runway': runway,
                        'Runway_Capacity': self.runways[runway]['capacity'],
                        'Runway_Efficiency': self.runways[runway]['efficiency'],
                        'Capacity': self.aircraft_types[aircraft_type]['capacity'],
                        'Base_Turnaround': self.aircraft_types[aircraft_type]['turnaround'],
                        'Hour': hour,
                        'Peak_Category': self._get_peak_category(hour)
                    })
                    
                    flight_id += 1
        
        return pd.DataFrame(flights)
    
    def _get_hourly_flight_distribution(self) -> Dict[int, int]:
        """Get realistic hourly flight distribution for congested airports like Mumbai/Delhi."""
        # Base flights with congested airport patterns
        base_flights = {hour: 3 for hour in range(24)}  # Minimum 3 flights per hour
        
        # Mumbai/Delhi specific peak patterns
        for hour in self.peak_patterns['super_peak']:
            # Extreme congestion during these hours
            base_flights[hour] += random.randint(15, 25)  # 18-28 flights/hour
        
        for hour in self.peak_patterns['peak']:
            # High congestion
            base_flights[hour] += random.randint(10, 15)  # 13-18 flights/hour
            
        for hour in self.peak_patterns['moderate']:
            # Moderate traffic
            base_flights[hour] += random.randint(5, 10)   # 8-13 flights/hour
        
        for hour in self.peak_patterns['low']:
            # Low traffic (night hours)
            base_flights[hour] = random.randint(2, 5)     # 2-5 flights/hour
        
        return base_flights
    
    def _get_peak_category(self, hour: int) -> str:
        """Categorize hour by traffic intensity."""
        if hour in self.peak_patterns['super_peak']:
            return 'super_peak'
        elif hour in self.peak_patterns['peak']:
            return 'peak'
        elif hour in self.peak_patterns['moderate']:
            return 'moderate'
        else:
            return 'low'
    
    def add_realistic_delays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic delays for congested airports with cascading effects."""
        df = df.copy()
        delays = []
        
        for _, flight in df.iterrows():
            # Base delay probability from aircraft type
            base_delay_prob = self.aircraft_types[flight['Aircraft_Type']]['delay_prob']
            
            # Congestion factor multiplier
            congestion_multiplier = flight['Congestion_Factor']
            
            # Peak hour multiplier based on traffic intensity
            peak_multipliers = {
                'super_peak': 2.5,    # Extreme delays during super peak
                'peak': 1.8,          # High delays during peak
                'moderate': 1.2,      # Moderate delays
                'low': 0.6            # Fewer delays during low traffic
            }
            peak_multiplier = peak_multipliers[flight['Peak_Category']]
            
            # Runway efficiency factor (less efficient = more delays)
            runway_factor = 1.0 / flight['Runway_Efficiency']
            
            # Calculate final delay probability
            final_delay_prob = min(0.85, base_delay_prob * congestion_multiplier * peak_multiplier * runway_factor)
            
            # Generate delay
            if random.random() < final_delay_prob:
                # More sophisticated delay distribution for congested airports
                if flight['Peak_Category'] == 'super_peak':
                    # Longer delays during super peak hours
                    delay_minutes = random.choices(
                        [15, 30, 45, 60, 90, 120, 180, 240],
                        weights=[0.15, 0.20, 0.20, 0.20, 0.15, 0.05, 0.03, 0.02]
                    )[0]
                elif flight['Peak_Category'] == 'peak':
                    delay_minutes = random.choices(
                        [15, 30, 45, 60, 90, 120, 180],
                        weights=[0.25, 0.25, 0.20, 0.15, 0.10, 0.03, 0.02]
                    )[0]
                else:
                    # Standard delay distribution for off-peak
                    delay_minutes = random.choices(
                        [15, 30, 45, 60, 90, 120],
                        weights=[0.35, 0.25, 0.20, 0.12, 0.05, 0.03]
                    )[0]
            else:
                delay_minutes = 0
            
            delays.append(delay_minutes)
        
        df['Delay_Minutes'] = delays
        
        # Add cascading delay effects for congested airports
        df = self.add_congestion_cascading_delays(df)
        
        df['Actual_Time'] = df['Scheduled_Time'] + pd.to_timedelta(df['Delay_Minutes'], unit='minutes')
        
        return df
    
    def add_congestion_cascading_delays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cascading delay effects specific to congested airports."""
        df = df.copy()
        
        # For Mumbai/Delhi focused dataset, assume all flights are from congested airports
        # Group by runway and day to handle cascading effects
        for (runway, date), group in df.groupby(['Runway', df['Scheduled_Time'].dt.date]):
            group_sorted = group.sort_values('Scheduled_Time')
            accumulated_delay = 0
            
            for idx, (flight_idx, flight) in enumerate(group_sorted.iterrows()):
                # Calculate runway congestion based on flights in the last hour
                recent_flights = group_sorted[
                    (group_sorted['Scheduled_Time'] >= flight['Scheduled_Time'] - pd.Timedelta(hours=1)) &
                    (group_sorted['Scheduled_Time'] <= flight['Scheduled_Time'])
                ]
                
                # Runway utilization factor
                runway_utilization = len(recent_flights) / flight['Runway_Capacity']
                
                # Add cascading delay if runway is overutilized
                if runway_utilization > 0.8 and accumulated_delay > 0:
                    cascading_delay = min(60, accumulated_delay * 0.3 * runway_utilization)
                    df.loc[flight_idx, 'Delay_Minutes'] += int(cascading_delay)
                    accumulated_delay += cascading_delay
                else:
                    accumulated_delay = max(0, accumulated_delay * 0.8)  # Decay over time
                
                # Update accumulated delay
                if df.loc[flight_idx, 'Delay_Minutes'] > 0:
                    accumulated_delay += df.loc[flight_idx, 'Delay_Minutes'] * 0.1
        
        return df
    
    def add_cascading_delays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cascading delays based on aircraft turnaround."""
        df = df.copy()
        df = df.sort_values('Scheduled_Time').reset_index(drop=True)
        
        # Group by aircraft to simulate turnaround delays
        for aircraft_id in df['Aircraft_ID'].unique():
            aircraft_flights = df[df['Aircraft_ID'] == aircraft_id].sort_values('Scheduled_Time')
            
            if len(aircraft_flights) > 1:
                for i in range(1, len(aircraft_flights)):
                    prev_flight_idx = aircraft_flights.iloc[i-1].name
                    current_flight_idx = aircraft_flights.iloc[i].name
                    
                    prev_actual_time = df.loc[prev_flight_idx, 'Actual_Time']
                    current_scheduled = df.loc[current_flight_idx, 'Scheduled_Time']
                    
                    # Minimum turnaround time: 45 minutes for domestic
                    min_turnaround = timedelta(minutes=45)
                    earliest_departure = prev_actual_time + min_turnaround
                    
                    if earliest_departure > current_scheduled:
                        # Add cascading delay
                        additional_delay = (earliest_departure - current_scheduled).total_seconds() / 60
                        df.loc[current_flight_idx, 'Delay_Minutes'] += additional_delay
                        df.loc[current_flight_idx, 'Actual_Time'] = earliest_departure
        
        return df
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional features for analysis."""
        df = df.copy()
        
        # Time-based features
        df['Hour'] = df['Scheduled_Time'].dt.hour
        df['Day_of_Week'] = df['Scheduled_Time'].dt.dayofweek
        df['Date'] = df['Scheduled_Time'].dt.date
        
        # Peak hour indicator
        df['Is_Peak_Hour'] = df['Hour'].isin([6, 7, 8, 9, 18, 19, 20, 21])
        
        # Calculate slot congestion (flights in same hour)
        hourly_counts = df.groupby(['Date', 'Hour']).size().reset_index(name='Hourly_Flight_Count')
        df = df.merge(hourly_counts, on=['Date', 'Hour'], how='left')
        
        # Runway congestion
        runway_hourly = df.groupby(['Date', 'Hour', 'Runway']).size().reset_index(name='Runway_Hourly_Count')
        df = df.merge(runway_hourly, on=['Date', 'Hour', 'Runway'], how='left')
        
        # High impact flight indicator (large aircraft or major destinations)
        major_destinations = ['BOM', 'DEL', 'BLR', 'MAA', 'CCU', 'HYD']
        large_aircraft = ['B777', 'A350', 'B787']
        
        df['Is_High_Impact'] = (
            (df['Destination'].isin(major_destinations)) | 
            (df['Aircraft_Type'].isin(large_aircraft)) |
            (df['Capacity'] > 250)
        )
        
        return df
    
    def generate_complete_dataset(self, start_date: str = '2025-08-15', days: int = 7) -> pd.DataFrame:
        """Generate complete flight dataset with all features and delays."""
        print("Generating base schedule...")
        df = self.generate_base_schedule(start_date, days)
        
        print("Adding realistic delays...")
        df = self.add_realistic_delays(df)
        
        print("Adding cascading delays...")
        df = self.add_cascading_delays(df)
        
        print("Adding features...")
        df = self.add_features(df)
        
        print(f"Generated {len(df)} flights over {days} days")
        return df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")

def main():
    """Generate and save flight dataset."""
    generator = FlightDataGenerator()
    
    # Generate 7 days of flight data
    df = generator.generate_complete_dataset()
    
    # Save to data directory
    filepath = os.path.join('data', 'flight_schedule_data.csv')
    generator.save_dataset(df, filepath)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total flights: {len(df)}")
    print(f"Delayed flights: {len(df[df['Delay_Minutes'] > 0])} ({len(df[df['Delay_Minutes'] > 0])/len(df)*100:.1f}%)")
    print(f"Average delay: {df['Delay_Minutes'].mean():.1f} minutes")
    print(f"Max delay: {df['Delay_Minutes'].max():.0f} minutes")
    
    print("\n=== Peak Hours Analysis ===")
    peak_delays = df[df['Is_Peak_Hour']]['Delay_Minutes'].mean()
    off_peak_delays = df[~df['Is_Peak_Hour']]['Delay_Minutes'].mean()
    print(f"Peak hour avg delay: {peak_delays:.1f} minutes")
    print(f"Off-peak avg delay: {off_peak_delays:.1f} minutes")

if __name__ == "__main__":
    main()
