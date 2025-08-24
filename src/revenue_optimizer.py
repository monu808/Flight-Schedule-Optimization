"""
Revenue-Optimized Flight Scheduling Module
Implements peak-hour revenue maximization with demand-based pricing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@dataclass
class RevenueSlot:
    """Revenue characteristics for different time slots."""
    hour: int
    demand_multiplier: float  # 0.5 to 2.0
    revenue_multiplier: float  # 0.7 to 1.8
    passenger_willingness_to_pay: float  # 0.6 to 1.5
    business_traveler_ratio: float  # 0.1 to 0.8

class RevenueOptimizer:
    """
    Optimizes flight schedules for maximum revenue generation around peak hours.
    Considers passenger demand patterns, pricing elasticity, and slot value.
    """
    
    def __init__(self):
        """Initialize revenue optimizer with demand patterns."""
        self.revenue_slots = self._initialize_revenue_patterns()
        self.base_ticket_prices = self._initialize_base_prices()
        self.peak_hour_premiums = self._initialize_peak_premiums()
        
    def _initialize_revenue_patterns(self) -> Dict[int, RevenueSlot]:
        """Initialize hourly revenue patterns based on business travel demand."""
        return {
            # Early morning - Business departures (high revenue)
            5: RevenueSlot(5, 0.8, 1.3, 1.2, 0.6),
            6: RevenueSlot(6, 1.4, 1.6, 1.4, 0.7),
            7: RevenueSlot(7, 1.8, 1.8, 1.5, 0.8),
            8: RevenueSlot(8, 1.9, 1.7, 1.4, 0.8),
            9: RevenueSlot(9, 1.6, 1.5, 1.3, 0.7),
            
            # Mid-day - Mixed traffic
            10: RevenueSlot(10, 1.2, 1.2, 1.1, 0.5),
            11: RevenueSlot(11, 1.1, 1.1, 1.0, 0.4),
            12: RevenueSlot(12, 1.3, 1.3, 1.2, 0.5),
            13: RevenueSlot(13, 1.4, 1.3, 1.1, 0.5),
            14: RevenueSlot(14, 1.2, 1.2, 1.0, 0.4),
            15: RevenueSlot(15, 1.0, 1.0, 0.9, 0.3),
            16: RevenueSlot(16, 1.1, 1.1, 1.0, 0.4),
            
            # Evening - Return traffic (premium hours)
            17: RevenueSlot(17, 1.5, 1.4, 1.3, 0.6),
            18: RevenueSlot(18, 1.8, 1.7, 1.5, 0.7),
            19: RevenueSlot(19, 2.0, 1.8, 1.5, 0.8),
            20: RevenueSlot(20, 1.9, 1.7, 1.4, 0.7),
            21: RevenueSlot(21, 1.6, 1.5, 1.3, 0.6),
            22: RevenueSlot(22, 1.3, 1.3, 1.2, 0.5),
            
            # Late night/early morning - Low demand
            23: RevenueSlot(23, 0.7, 0.9, 0.8, 0.2),
            0: RevenueSlot(0, 0.5, 0.7, 0.6, 0.1),
            1: RevenueSlot(1, 0.5, 0.7, 0.6, 0.1),
            2: RevenueSlot(2, 0.5, 0.7, 0.6, 0.1),
            3: RevenueSlot(3, 0.6, 0.8, 0.7, 0.2),
            4: RevenueSlot(4, 0.7, 0.9, 0.8, 0.3),
        }
    
    def _initialize_base_prices(self) -> Dict[str, float]:
        """Initialize base ticket prices by destination category."""
        return {
            # Tier 1 destinations (high business demand)
            'high': 12000,   # Mumbai, Delhi, Bangalore
            'medium': 8500,  # Chennai, Hyderabad, Kolkata
            'normal': 6200   # Other cities
        }
    
    def _initialize_peak_premiums(self) -> Dict[str, float]:
        """Peak hour premium percentages by flight category."""
        return {
            'super_peak': 0.35,  # 35% premium during super peak hours
            'peak': 0.20,        # 20% premium during peak hours
            'moderate': 0.05,    # 5% premium during moderate hours
            'low': -0.15         # 15% discount during low demand hours
        }
    
    def calculate_flight_revenue(self, flight_data: Dict) -> float:
        """
        Calculate expected revenue for a flight in a specific time slot.
        
        Args:
            flight_data: Dictionary containing flight information
            
        Returns:
            Expected revenue for the flight
        """
        hour = flight_data['Hour']
        capacity = flight_data['Capacity']
        destination_priority = flight_data['Destination_Priority']
        peak_category = flight_data['Peak_Category']
        
        # Get revenue characteristics for this hour
        revenue_slot = self.revenue_slots.get(hour, self.revenue_slots[12])  # Default to noon
        
        # Base ticket price
        base_price = self.base_ticket_prices[destination_priority]
        
        # Apply hourly revenue multiplier
        hourly_price = base_price * revenue_slot.revenue_multiplier
        
        # Apply peak hour premium/discount
        peak_premium = self.peak_hour_premiums[peak_category]
        final_price = hourly_price * (1 + peak_premium)
        
        # Calculate load factor based on demand
        base_load_factor = 0.75  # Base 75% load factor
        demand_adjusted_load_factor = min(0.95, base_load_factor * revenue_slot.demand_multiplier)
        
        # Business class premium (higher during peak hours)
        business_ratio = revenue_slot.business_traveler_ratio
        business_premium = 2.8  # Business class costs 2.8x economy
        
        # Calculate weighted average ticket price
        economy_passengers = capacity * demand_adjusted_load_factor * (1 - business_ratio * 0.15)
        business_passengers = capacity * demand_adjusted_load_factor * (business_ratio * 0.15)
        
        economy_revenue = economy_passengers * final_price
        business_revenue = business_passengers * final_price * business_premium
        
        total_revenue = economy_revenue + business_revenue
        
        return total_revenue
    
    def optimize_for_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize flight schedule to maximize revenue by prioritizing peak hours.
        
        Args:
            df: Flight schedule DataFrame
            
        Returns:
            Revenue-optimized DataFrame with recommendations
        """
        # Calculate current revenue for each flight
        df['Current_Revenue'] = df.apply(
            lambda row: self.calculate_flight_revenue(row.to_dict()), axis=1
        )
        
        # Identify revenue optimization opportunities
        optimization_recommendations = []
        
        for _, flight in df.iterrows():
            current_hour = flight['Hour']
            current_revenue = flight['Current_Revenue']
            
            # Test moving to different peak hours
            best_alternative_hour = None
            best_alternative_revenue = current_revenue
            
            # Check super peak hours (7-9 AM, 7-9 PM)
            super_peak_hours = [7, 8, 19, 20]
            for alt_hour in super_peak_hours:
                if alt_hour != current_hour:
                    alt_flight_data = flight.to_dict()
                    alt_flight_data['Hour'] = alt_hour
                    alt_flight_data['Peak_Category'] = 'super_peak'
                    
                    alt_revenue = self.calculate_flight_revenue(alt_flight_data)
                    
                    if alt_revenue > best_alternative_revenue * 1.05:  # 5% improvement threshold
                        best_alternative_hour = alt_hour
                        best_alternative_revenue = alt_revenue
            
            if best_alternative_hour:
                revenue_increase = best_alternative_revenue - current_revenue
                percentage_increase = (revenue_increase / current_revenue) * 100
                
                optimization_recommendations.append({
                    'Flight_ID': flight['Flight_ID'],
                    'Current_Hour': current_hour,
                    'Recommended_Hour': best_alternative_hour,
                    'Current_Revenue': current_revenue,
                    'Projected_Revenue': best_alternative_revenue,
                    'Revenue_Increase': revenue_increase,
                    'Percentage_Increase': percentage_increase
                })
        
        # Add revenue metrics to DataFrame
        df['Revenue_Per_Hour_Multiplier'] = df['Hour'].map(
            lambda h: self.revenue_slots.get(h, self.revenue_slots[12]).revenue_multiplier
        )
        df['Demand_Multiplier'] = df['Hour'].map(
            lambda h: self.revenue_slots.get(h, self.revenue_slots[12]).demand_multiplier
        )
        
        # Create optimization summary
        optimization_df = pd.DataFrame(optimization_recommendations)
        
        return df, optimization_df
    
    def analyze_peak_hour_consolidation(self, df: pd.DataFrame) -> Dict:
        """
        Analyze current flight distribution and recommend peak hour consolidation.
        
        Args:
            df: Flight schedule DataFrame
            
        Returns:
            Analysis results with consolidation recommendations
        """
        # Calculate hourly revenue and flight distribution
        hourly_analysis = df.groupby('Hour').agg({
            'Flight_ID': 'count',
            'Current_Revenue': ['sum', 'mean'],
            'Capacity': 'sum'
        }).round(2)
        
        hourly_analysis.columns = ['Flight_Count', 'Total_Revenue', 'Avg_Revenue_Per_Flight', 'Total_Capacity']
        hourly_analysis = hourly_analysis.reset_index()
        
        # Add revenue efficiency metrics
        hourly_analysis['Revenue_Per_Flight'] = hourly_analysis['Total_Revenue'] / hourly_analysis['Flight_Count']
        hourly_analysis['Revenue_Per_Passenger'] = hourly_analysis['Total_Revenue'] / hourly_analysis['Total_Capacity']
        
        # Identify optimal consolidation hours
        super_peak_hours = [7, 8, 19, 20]
        peak_hours = [6, 9, 18, 21]
        
        super_peak_analysis = hourly_analysis[hourly_analysis['Hour'].isin(super_peak_hours)]
        current_super_peak_utilization = super_peak_analysis['Flight_Count'].sum()
        
        # Calculate potential revenue from full peak hour utilization
        total_flights = df['Flight_ID'].count()
        avg_flight_revenue = df['Current_Revenue'].mean()
        
        # Simulate moving 60% of off-peak flights to peak hours
        off_peak_flights = df[~df['Hour'].isin(super_peak_hours + peak_hours)]
        movable_flights = len(off_peak_flights) * 0.6
        
        # Calculate revenue impact
        current_total_revenue = df['Current_Revenue'].sum()
        
        # Estimate revenue from peak hour consolidation
        peak_revenue_multiplier = 1.6  # Average peak hour multiplier
        potential_additional_revenue = movable_flights * avg_flight_revenue * (peak_revenue_multiplier - 1)
        
        return {
            'hourly_analysis': hourly_analysis,
            'current_super_peak_flights': current_super_peak_utilization,
            'total_flights': total_flights,
            'peak_utilization_percentage': (current_super_peak_utilization / total_flights) * 100,
            'current_total_revenue': current_total_revenue,
            'potential_additional_revenue': potential_additional_revenue,
            'revenue_increase_percentage': (potential_additional_revenue / current_total_revenue) * 100,
            'recommended_consolidation_hours': super_peak_hours
        }
    
    def create_revenue_visualization(self, df: pd.DataFrame) -> go.Figure:
        """Create revenue analysis visualization."""
        # Calculate hourly revenue metrics
        hourly_revenue = df.groupby('Hour').agg({
            'Current_Revenue': 'sum',
            'Flight_ID': 'count'
        }).reset_index()
        
        hourly_revenue['Avg_Revenue_Per_Flight'] = hourly_revenue['Current_Revenue'] / hourly_revenue['Flight_ID']
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hourly Total Revenue', 'Revenue Per Flight', 'Flight Distribution', 'Revenue Efficiency'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Hourly total revenue
        fig.add_trace(
            go.Bar(x=hourly_revenue['Hour'], y=hourly_revenue['Current_Revenue'], 
                   name='Total Revenue', marker_color='green'),
            row=1, col=1
        )
        
        # Revenue per flight
        fig.add_trace(
            go.Scatter(x=hourly_revenue['Hour'], y=hourly_revenue['Avg_Revenue_Per_Flight'],
                      mode='lines+markers', name='Revenue per Flight', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Flight distribution
        fig.add_trace(
            go.Bar(x=hourly_revenue['Hour'], y=hourly_revenue['Flight_ID'],
                   name='Flight Count', marker_color='orange'),
            row=2, col=1
        )
        
        # Revenue efficiency (revenue per flight)
        peak_hours = [7, 8, 19, 20]
        colors = ['red' if hour in peak_hours else 'lightblue' for hour in hourly_revenue['Hour']]
        
        fig.add_trace(
            go.Bar(x=hourly_revenue['Hour'], y=hourly_revenue['Avg_Revenue_Per_Flight'],
                   name='Revenue Efficiency', marker_color=colors),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Revenue Analysis Dashboard",
            showlegend=True,
            height=600
        )
        
        return fig
