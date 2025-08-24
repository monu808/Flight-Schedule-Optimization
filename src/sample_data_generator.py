"""
Sample Data Generator for Flight Schedule Optimization
====================================================

Generates realistic flight data with various scenarios for testing and demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import random
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class FlightDataGenerator:
    """Generate realistic flight schedule data for testing"""
    
    def __init__(self):
        # Airline configurations
        self.airlines = {
            'AI': {'name': 'Air India', 'fleet': ['A320', 'A321', 'B777', 'B787'], 'on_time_rate': 0.72},
            '6E': {'name': 'IndiGo', 'fleet': ['A320', 'A321'], 'on_time_rate': 0.85},
            'SG': {'name': 'SpiceJet', 'fleet': ['B737', 'B738'], 'on_time_rate': 0.78},
            'UK': {'name': 'Vistara', 'fleet': ['A320', 'A321', 'B787'], 'on_time_rate': 0.88},
            'G8': {'name': 'GoAir', 'fleet': ['A320'], 'on_time_rate': 0.74},
            'IX': {'name': 'Air India Express', 'fleet': ['B737'], 'on_time_rate': 0.69}
        }
        
        # Destination configurations
        self.destinations = {
            'Delhi (DEL)': {'distance': 1150, 'flight_time': 135, 'popularity': 0.25, 'weather_delay_prob': 0.15},
            'Bengaluru (BLR)': {'distance': 840, 'flight_time': 95, 'popularity': 0.20, 'weather_delay_prob': 0.08},
            'Chennai (MAA)': {'distance': 1040, 'flight_time': 125, 'popularity': 0.15, 'weather_delay_prob': 0.12},
            'Hyderabad (HYD)': {'distance': 625, 'flight_time': 85, 'popularity': 0.12, 'weather_delay_prob': 0.06},
            'Kolkata (CCU)': {'distance': 1650, 'flight_time': 165, 'popularity': 0.10, 'weather_delay_prob': 0.18},
            'Ahmedabad (AMD)': {'distance': 525, 'flight_time': 75, 'popularity': 0.08, 'weather_delay_prob': 0.05},
            'Pune (PNQ)': {'distance': 150, 'flight_time': 45, 'popularity': 0.05, 'weather_delay_prob': 0.03},
            'Goa (GOI)': {'distance': 440, 'flight_time': 65, 'popularity': 0.05, 'weather_delay_prob': 0.07}
        }
        
        # Time slot configurations
        self.time_slots = {
            'Early Morning (5-7 AM)': {'congestion': 0.3, 'delay_factor': 0.8, 'slots': 20},
            'Morning Peak (7-9 AM)': {'congestion': 0.9, 'delay_factor': 1.4, 'slots': 45},
            'Mid Morning (9-11 AM)': {'congestion': 0.7, 'delay_factor': 1.2, 'slots': 35},
            'Late Morning (11 AM-1 PM)': {'congestion': 0.6, 'delay_factor': 1.0, 'slots': 30}
        }
    
    def generate_flight_schedule(self, num_flights: int = 500, scenario: str = 'normal') -> pd.DataFrame:
        """
        Generate realistic flight schedule data
        
        Args:
            num_flights (int): Number of flights to generate
            scenario (str): Scenario type ('normal', 'high_delay', 'weather_event', 'peak_congestion')
            
        Returns:
            pd.DataFrame: Generated flight data
        """
        
        flights = []
        base_date = datetime.now().date()
        
        # Scenario-specific adjustments
        scenario_config = self._get_scenario_config(scenario)
        
        for i in range(num_flights):
            flight_data = self._generate_single_flight(i + 1, base_date, scenario_config)
            flights.append(flight_data)
        
        df = pd.DataFrame(flights)
        
        # Add derived columns
        df = self._add_derived_columns(df)
        
        logger.info(f"Generated {len(df)} flights for scenario: {scenario}")
        return df
    
    def _generate_single_flight(self, flight_num: int, base_date: datetime, scenario_config: Dict) -> Dict:
        """Generate a single flight record"""
        
        # Select airline
        airline_code = random.choices(
            list(self.airlines.keys()),
            weights=[0.25, 0.30, 0.15, 0.12, 0.10, 0.08]  # Market share weights
        )[0]
        airline_info = self.airlines[airline_code]
        
        # Select destination
        destination = random.choices(
            list(self.destinations.keys()),
            weights=[dest['popularity'] for dest in self.destinations.values()]
        )[0]
        dest_info = self.destinations[destination]
        
        # Generate scheduled departure time
        std_time = self._generate_departure_time()
        
        # Calculate scheduled arrival
        flight_duration = dest_info['flight_time'] + random.randint(-10, 20)  # Some variance
        sta_time = (datetime.combine(base_date, std_time) + timedelta(minutes=flight_duration)).time()
        
        # Generate delays based on scenario and airline performance
        departure_delay = self._calculate_delay(std_time, airline_info, dest_info, scenario_config)
        arrival_delay = departure_delay + random.randint(-5, 15)  # Arrival can be better/worse
        
        # Calculate actual times
        atd_time = (datetime.combine(base_date, std_time) + timedelta(minutes=departure_delay)).time()
        ata_time = (datetime.combine(base_date, sta_time) + timedelta(minutes=arrival_delay)).time()
        
        return {
            'S.No': flight_num,
            'Flight Number': f"{airline_code}{random.randint(100, 9999)}",
            'From': 'Mumbai (BOM)',
            'To': destination.split('(')[0].strip(),
            'Aircraft': random.choice(airline_info['fleet']),
            'Flight time': f"{flight_duration} min",
            'STD': std_time,
            'ATD': atd_time,
            'STA': sta_time,
            'ATA': ata_time,
            'Date': base_date,
            'departure_delay_minutes': max(0, departure_delay),
            'arrival_delay_minutes': max(0, arrival_delay),
            'airline': airline_info['name'],
            'destination_code': destination.split('(')[1].replace(')', ''),
            'distance_km': dest_info['distance'],
            'scenario': scenario_config['name']
        }
    
    def _generate_departure_time(self) -> time:
        """Generate a departure time based on slot distribution"""
        
        # Define time ranges and their probabilities
        time_ranges = [
            (time(5, 0), time(7, 0), 0.15),   # Early morning
            (time(7, 0), time(9, 0), 0.40),   # Morning peak
            (time(9, 0), time(11, 0), 0.30),  # Mid morning
            (time(11, 0), time(13, 0), 0.15)  # Late morning
        ]
        
        # Select time range
        range_choice = random.choices(time_ranges, weights=[r[2] for r in time_ranges])[0]
        start_time, end_time, _ = range_choice
        
        # Generate random time within range
        start_minutes = start_time.hour * 60 + start_time.minute
        end_minutes = end_time.hour * 60 + end_time.minute
        
        random_minutes = random.randint(start_minutes, end_minutes - 1)
        return time(random_minutes // 60, random_minutes % 60)
    
    def _calculate_delay(self, std_time: time, airline_info: Dict, dest_info: Dict, scenario_config: Dict) -> int:
        """Calculate delay based on various factors"""
        
        base_delay = 0
        
        # Time-based delay (peak hours have more delays)
        hour = std_time.hour
        if 7 <= hour < 9:  # Peak hours
            base_delay += random.randint(5, 25)
        elif 5 <= hour < 7 or 11 <= hour < 13:  # Off-peak
            base_delay += random.randint(0, 10)
        else:  # Mid-peak
            base_delay += random.randint(2, 15)
        
        # Airline performance factor
        if random.random() > airline_info['on_time_rate']:
            base_delay += random.randint(10, 45)
        
        # Weather delay probability
        if random.random() < dest_info['weather_delay_prob']:
            base_delay += random.randint(15, 90)
        
        # Scenario-specific delays
        base_delay = int(base_delay * scenario_config['delay_multiplier'])
        
        # Add some randomness
        base_delay += random.randint(-5, 15)
        
        return max(0, base_delay)
    
    def _get_scenario_config(self, scenario: str) -> Dict:
        """Get configuration for different scenarios"""
        
        scenarios = {
            'normal': {
                'name': 'Normal Operations',
                'delay_multiplier': 1.0,
                'weather_factor': 1.0,
                'congestion_factor': 1.0
            },
            'high_delay': {
                'name': 'High Delay Day',
                'delay_multiplier': 1.8,
                'weather_factor': 1.5,
                'congestion_factor': 1.4
            },
            'weather_event': {
                'name': 'Weather Event',
                'delay_multiplier': 2.2,
                'weather_factor': 3.0,
                'congestion_factor': 1.6
            },
            'peak_congestion': {
                'name': 'Peak Congestion',
                'delay_multiplier': 1.5,
                'weather_factor': 1.0,
                'congestion_factor': 2.0
            },
            'ideal_conditions': {
                'name': 'Ideal Conditions',
                'delay_multiplier': 0.3,
                'weather_factor': 0.2,
                'congestion_factor': 0.5
            }
        }
        
        return scenarios.get(scenario, scenarios['normal'])
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns similar to real data structure"""
        
        # Create route column
        df['Route'] = df['From'] + ' → ' + df['To']
        
        # Time slots
        df['time_slot'] = df['STD'].apply(self._get_time_slot)
        
        # Peak classification
        df['peak_classification'] = df['STD'].apply(self._classify_peak_time)
        
        # Hour slots
        df['STD_hour_slot'] = df['STD'].apply(lambda x: f"{x.hour}AM-{x.hour+1}AM" if x.hour < 12 else f"{x.hour}PM-{x.hour+1}PM")
        df['ATD_hour_slot'] = df['ATD'].apply(lambda x: f"{x.hour}AM-{x.hour+1}AM" if x.hour < 12 else f"{x.hour}PM-{x.hour+1}PM")
        
        # Sheet source for compatibility
        df['sheet_source'] = df['time_slot'].map({
            '6AM - 9AM': '6AM - 9AM',
            '9AM - 12PM': '9AM - 12PM'
        })
        
        return df
    
    def _get_time_slot(self, std_time: time) -> str:
        """Determine time slot for a given time"""
        if std_time.hour < 9:
            return '6AM - 9AM'
        else:
            return '9AM - 12PM'
    
    def _classify_peak_time(self, std_time: time) -> str:
        """Classify peak time for a given time"""
        hour = std_time.hour
        if 7 <= hour < 8:
            return 'Super Peak'
        elif 6 <= hour < 9:
            return 'Peak'
        elif 9 <= hour < 11:
            return 'Moderate'
        else:
            return 'Low'
    
    def generate_multiple_scenarios(self, flights_per_scenario: int = 200) -> Dict[str, pd.DataFrame]:
        """Generate data for multiple scenarios"""
        
        scenarios = ['normal', 'high_delay', 'weather_event', 'peak_congestion', 'ideal_conditions']
        scenario_data = {}
        
        for scenario in scenarios:
            scenario_data[scenario] = self.generate_flight_schedule(flights_per_scenario, scenario)
        
        return scenario_data


def generate_sample_data(scenario: str = 'normal', num_flights: int = 500) -> pd.DataFrame:
    """
    Convenience function to generate sample data
    
    Args:
        scenario (str): Scenario type
        num_flights (int): Number of flights
        
    Returns:
        pd.DataFrame: Generated flight data
    """
    generator = FlightDataGenerator()
    return generator.generate_flight_schedule(num_flights, scenario)


if __name__ == "__main__":
    # Test the generator
    generator = FlightDataGenerator()
    
    # Generate normal scenario
    data = generator.generate_flight_schedule(100, 'normal')
    print(f"Generated {len(data)} flights")
    print(data.head())
    print(f"Average delay: {data['departure_delay_minutes'].mean():.1f} minutes")
