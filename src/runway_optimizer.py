"""
Runway Optimization Module
Dynamic slot allocation based on aircraft type and priority-based scheduling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FlightPriority(Enum):
    """Flight priority levels."""
    EMERGENCY = 1
    INTERNATIONAL = 2
    DOMESTIC_MAJOR = 3
    DOMESTIC_REGIONAL = 4
    CARGO = 5

class AircraftCategory(Enum):
    """Aircraft categories based on size and runway requirements."""
    HEAVY = "Heavy"      # Wide-body, long-haul
    MEDIUM = "Medium"    # Narrow-body, medium-haul
    LIGHT = "Light"      # Regional, short-haul

@dataclass
class RunwaySlot:
    """Represents a time slot on a runway."""
    runway_id: str
    start_time: datetime
    end_time: datetime
    is_occupied: bool = False
    assigned_flight: Optional[str] = None
    buffer_time: int = 5  # minutes between flights
    
@dataclass
class AircraftType:
    """Aircraft type specifications."""
    model: str
    category: AircraftCategory
    min_runway_length: int  # meters
    wake_turbulence_category: str  # Heavy, Medium, Light
    preferred_runways: List[str] = field(default_factory=list)
    turnaround_time: int = 45  # minutes
    
@dataclass
class Runway:
    """Runway specifications and capabilities."""
    id: str
    length: int  # meters
    width: int   # meters
    surface_type: str
    max_operations_per_hour: int = 30
    preferred_aircraft_types: List[str] = field(default_factory=list)
    maintenance_windows: List[Tuple[datetime, datetime]] = field(default_factory=list)
    weather_restrictions: Dict[str, bool] = field(default_factory=dict)
    
class RunwayOptimizer:
    """
    Optimizes runway slot allocation with dynamic scheduling based on aircraft type and priorities.
    """
    
    def __init__(self):
        self.runways = self._initialize_runways()
        self.aircraft_types = self._initialize_aircraft_types()
        self.slot_duration_minutes = 15
        self.wake_turbulence_separations = {
            ('Heavy', 'Heavy'): 4,
            ('Heavy', 'Medium'): 5,
            ('Heavy', 'Light'): 6,
            ('Medium', 'Heavy'): 3,
            ('Medium', 'Medium'): 3,
            ('Medium', 'Light'): 5,
            ('Light', 'Heavy'): 3,
            ('Light', 'Medium'): 3,
            ('Light', 'Light'): 3,
        }
        
    def _initialize_runways(self) -> Dict[str, Runway]:
        """Initialize runway specifications."""
        return {
            '09R/27L': Runway(
                id='09R/27L',
                length=4430,
                width=60,
                surface_type='Asphalt',
                max_operations_per_hour=35,
                preferred_aircraft_types=['Heavy', 'Medium']
            ),
            '09L/27R': Runway(
                id='09L/27R', 
                length=4430,
                width=60,
                surface_type='Asphalt',
                max_operations_per_hour=35,
                preferred_aircraft_types=['Heavy', 'Medium']
            ),
            '14/32': Runway(
                id='14/32',
                length=2925,
                width=45,
                surface_type='Asphalt',
                max_operations_per_hour=25,
                preferred_aircraft_types=['Medium', 'Light']
            ),
            '27/09': Runway(
                id='27/09',
                length=3445,
                width=45,
                surface_type='Asphalt',
                max_operations_per_hour=28,
                preferred_aircraft_types=['Medium', 'Light']
            )
        }
    
    def _initialize_aircraft_types(self) -> Dict[str, AircraftType]:
        """Initialize aircraft type specifications."""
        return {
            'A320': AircraftType('A320', AircraftCategory.MEDIUM, 2400, 'Medium', ['09R/27L', '09L/27R', '14/32']),
            'A321': AircraftType('A321', AircraftCategory.MEDIUM, 2500, 'Medium', ['09R/27L', '09L/27R', '14/32']),
            'A330': AircraftType('A330', AircraftCategory.HEAVY, 2500, 'Heavy', ['09R/27L', '09L/27R']),
            'A350': AircraftType('A350', AircraftCategory.HEAVY, 2600, 'Heavy', ['09R/27L', '09L/27R']),
            'B737': AircraftType('B737', AircraftCategory.MEDIUM, 2100, 'Medium', ['09R/27L', '09L/27R', '14/32', '27/09']),
            'B777': AircraftType('B777', AircraftCategory.HEAVY, 2600, 'Heavy', ['09R/27L', '09L/27R']),
            'B787': AircraftType('B787', AircraftCategory.HEAVY, 2600, 'Heavy', ['09R/27L', '09L/27R']),
            'ATR72': AircraftType('ATR72', AircraftCategory.LIGHT, 1200, 'Light', ['14/32', '27/09']),
            'Q400': AircraftType('Q400', AircraftCategory.LIGHT, 1425, 'Light', ['14/32', '27/09']),
        }
    
    def determine_flight_priority(self, flight_data: Dict) -> FlightPriority:
        """
        Determine flight priority based on various factors.
        
        Args:
            flight_data: Dictionary containing flight information
            
        Returns:
            FlightPriority enum value
        """
        # Check if emergency
        if flight_data.get('is_emergency', False):
            return FlightPriority.EMERGENCY
        
        # Check if international
        if flight_data.get('is_international', False):
            return FlightPriority.INTERNATIONAL
        
        # Check aircraft capacity for domestic classification
        capacity = flight_data.get('capacity', 0)
        if capacity >= 150:
            return FlightPriority.DOMESTIC_MAJOR
        elif capacity >= 50:
            return FlightPriority.DOMESTIC_REGIONAL
        else:
            return FlightPriority.CARGO
    
    def get_aircraft_category(self, aircraft_model: str) -> AircraftCategory:
        """Get aircraft category from model."""
        aircraft_type = self.aircraft_types.get(aircraft_model)
        if aircraft_type:
            return aircraft_type.category
        
        # Default categorization based on model name
        if any(heavy in aircraft_model.upper() for heavy in ['777', '787', '330', '350', '747', '380']):
            return AircraftCategory.HEAVY
        elif any(light in aircraft_model.upper() for light in ['ATR', 'Q400', 'CRJ', 'ERJ']):
            return AircraftCategory.LIGHT
        else:
            return AircraftCategory.MEDIUM
    
    def calculate_runway_suitability(self, flight_data: Dict, runway_id: str) -> float:
        """
        Calculate how suitable a runway is for a specific flight.
        
        Args:
            flight_data: Flight information dictionary
            runway_id: Runway identifier
            
        Returns:
            Suitability score (0-1)
        """
        runway = self.runways[runway_id]
        aircraft_model = flight_data.get('aircraft_type', 'B737')
        aircraft_type = self.aircraft_types.get(aircraft_model)
        
        suitability_score = 0.5  # Base score
        
        # Check runway length requirement
        if aircraft_type and runway.length >= aircraft_type.min_runway_length:
            suitability_score += 0.2
        
        # Preferred runway bonus
        if aircraft_type and runway_id in aircraft_type.preferred_runways:
            suitability_score += 0.2
        
        # Aircraft category preference
        aircraft_category = self.get_aircraft_category(aircraft_model)
        if aircraft_category.value in runway.preferred_aircraft_types:
            suitability_score += 0.1
        
        return min(1.0, suitability_score)
    
    def create_time_slots(self, start_time: datetime, end_time: datetime, 
                         runway_id: str) -> List[RunwaySlot]:
        """
        Create time slots for a runway within a time range.
        
        Args:
            start_time: Start of scheduling period
            end_time: End of scheduling period
            runway_id: Runway identifier
            
        Returns:
            List of RunwaySlot objects
        """
        slots = []
        current_time = start_time
        
        while current_time < end_time:
            slot_end = current_time + timedelta(minutes=self.slot_duration_minutes)
            slot = RunwaySlot(
                runway_id=runway_id,
                start_time=current_time,
                end_time=slot_end
            )
            slots.append(slot)
            current_time = slot_end
        
        return slots
    
    def optimize_runway_allocation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize runway allocation for all flights using priority-based scheduling.
        
        Args:
            df: DataFrame with flight data
            
        Returns:
            DataFrame with optimized runway assignments
        """
        # Prepare flight data
        df_optimized = df.copy()
        df_optimized['Scheduled_Time'] = pd.to_datetime(df_optimized['Scheduled_Time'])
        df_optimized['Original_Runway'] = df_optimized['Runway']
        
        # Add priority and aircraft category
        flight_priorities = []
        aircraft_categories = []
        runway_suitabilities = {runway_id: [] for runway_id in self.runways.keys()}
        
        for _, row in df_optimized.iterrows():
            flight_data = row.to_dict()
            
            # Determine priority
            priority = self.determine_flight_priority(flight_data)
            flight_priorities.append(priority.value)
            
            # Get aircraft category
            aircraft_category = self.get_aircraft_category(row.get('Aircraft_Type', 'B737'))
            aircraft_categories.append(aircraft_category.value)
            
            # Calculate runway suitabilities
            for runway_id in self.runways.keys():
                suitability = self.calculate_runway_suitability(flight_data, runway_id)
                runway_suitabilities[runway_id].append(suitability)
        
        df_optimized['Priority'] = flight_priorities
        df_optimized['Aircraft_Category'] = aircraft_categories
        
        # Add suitability scores
        for runway_id in self.runways.keys():
            df_optimized[f'Suitability_{runway_id}'] = runway_suitabilities[runway_id]
        
        # Sort by priority and time
        df_optimized = df_optimized.sort_values(['Priority', 'Scheduled_Time'])
        
        # Create time slots for all runways
        min_time = df_optimized['Scheduled_Time'].min().replace(minute=0, second=0, microsecond=0)
        max_time = df_optimized['Scheduled_Time'].max() + timedelta(hours=2)
        
        runway_slots = {}
        for runway_id in self.runways.keys():
            runway_slots[runway_id] = self.create_time_slots(min_time, max_time, runway_id)
        
        # Allocate flights to optimal runways
        optimized_assignments = []
        
        for idx, row in df_optimized.iterrows():
            flight_id = row['Flight_ID']
            scheduled_time = row['Scheduled_Time']
            aircraft_model = row.get('Aircraft_Type', 'B737')
            
            # Find best available runway slot
            best_runway = None
            best_slot = None
            best_score = -1
            
            for runway_id in self.runways.keys():
                suitability = row[f'Suitability_{runway_id}']
                
                # Find available slot closest to scheduled time
                available_slots = [
                    slot for slot in runway_slots[runway_id]
                    if not slot.is_occupied and 
                    abs((slot.start_time - scheduled_time).total_seconds()) <= 3600  # Within 1 hour
                ]
                
                if available_slots:
                    # Choose slot closest to scheduled time
                    closest_slot = min(available_slots, 
                                     key=lambda s: abs((s.start_time - scheduled_time).total_seconds()))
                    
                    # Calculate time penalty
                    time_diff = abs((closest_slot.start_time - scheduled_time).total_seconds() / 60)
                    time_penalty = min(0.5, time_diff / 60)  # Max 50% penalty for 1 hour diff
                    
                    # Calculate final score
                    score = suitability * (1 - time_penalty)
                    
                    if score > best_score:
                        best_score = score
                        best_runway = runway_id
                        best_slot = closest_slot
            
            # Assign flight to best runway slot
            if best_runway and best_slot:
                best_slot.is_occupied = True
                best_slot.assigned_flight = flight_id
                
                # Calculate delay from original schedule
                time_diff = (best_slot.start_time - scheduled_time).total_seconds() / 60
                
                optimized_assignments.append({
                    'Flight_ID': flight_id,
                    'Optimized_Runway': best_runway,
                    'Optimized_Time': best_slot.start_time,
                    'Schedule_Change_Minutes': time_diff,
                    'Suitability_Score': best_score
                })
            else:
                # Keep original assignment if no better option found
                optimized_assignments.append({
                    'Flight_ID': flight_id,
                    'Optimized_Runway': row['Runway'],
                    'Optimized_Time': scheduled_time,
                    'Schedule_Change_Minutes': 0,
                    'Suitability_Score': 0.5
                })
        
        # Merge optimized assignments back
        assignments_df = pd.DataFrame(optimized_assignments)
        df_result = df_optimized.merge(assignments_df, on='Flight_ID', how='left')
        
        return df_result
    
    def calculate_runway_efficiency_metrics(self, df_optimized: pd.DataFrame) -> Dict:
        """
        Calculate efficiency metrics for the optimized runway allocation.
        
        Args:
            df_optimized: DataFrame with optimized assignments
            
        Returns:
            Dictionary with efficiency metrics
        """
        # Group by runway and hour
        df_optimized['Hour'] = pd.to_datetime(df_optimized['Optimized_Time']).dt.hour
        runway_hourly = df_optimized.groupby(['Optimized_Runway', 'Hour']).agg({
            'Flight_ID': 'count',
            'Capacity': 'sum',
            'Schedule_Change_Minutes': 'mean'
        }).reset_index()
        
        runway_hourly.rename(columns={'Flight_ID': 'Flights_Count'}, inplace=True)
        
        # Calculate utilization rates
        utilization_rates = {}
        for runway_id, runway in self.runways.items():
            runway_data = runway_hourly[runway_hourly['Optimized_Runway'] == runway_id]
            if not runway_data.empty:
                avg_flights_per_hour = runway_data['Flights_Count'].mean()
                utilization_rate = (avg_flights_per_hour / runway.max_operations_per_hour) * 100
                utilization_rates[runway_id] = utilization_rate
            else:
                utilization_rates[runway_id] = 0
        
        # Calculate throughput improvement
        original_delays = df_optimized['Delay_Minutes'].mean() if 'Delay_Minutes' in df_optimized.columns else 0
        schedule_changes = df_optimized['Schedule_Change_Minutes'].mean()
        
        # Priority compliance
        priority_compliance = {}
        for priority in [1, 2, 3, 4, 5]:  # Emergency to Cargo
            priority_flights = df_optimized[df_optimized['Priority'] == priority]
            if not priority_flights.empty:
                avg_delay = priority_flights['Schedule_Change_Minutes'].mean()
                priority_compliance[priority] = max(0, 100 - abs(avg_delay))
            else:
                priority_compliance[priority] = 100
        
        # Aircraft type efficiency
        aircraft_efficiency = df_optimized.groupby('Aircraft_Category').agg({
            'Suitability_Score': 'mean',
            'Schedule_Change_Minutes': 'mean'
        }).to_dict()
        
        return {
            'runway_utilization_rates': utilization_rates,
            'average_utilization': np.mean(list(utilization_rates.values())),
            'average_schedule_change': schedule_changes,
            'priority_compliance': priority_compliance,
            'aircraft_efficiency': aircraft_efficiency,
            'total_flights_optimized': len(df_optimized),
            'throughput_improvement_estimate': max(0, 15 - abs(schedule_changes)) / 15 * 100
        }
    
    def create_runway_optimization_dashboard(self, df_optimized: pd.DataFrame) -> Dict:
        """
        Create visualization dashboard for runway optimization results.
        
        Args:
            df_optimized: DataFrame with optimization results
            
        Returns:
            Dictionary containing plotly figures
        """
        figures = {}
        
        # 1. Runway Utilization Over Time
        df_optimized['Hour'] = pd.to_datetime(df_optimized['Optimized_Time']).dt.hour
        hourly_usage = df_optimized.groupby(['Hour', 'Optimized_Runway']).size().reset_index(name='Flight_Count')
        
        fig_utilization = px.bar(
            hourly_usage,
            x='Hour',
            y='Flight_Count',
            color='Optimized_Runway',
            title='Runway Utilization by Hour',
            labels={'Flight_Count': 'Number of Flights', 'Hour': 'Hour of Day'}
        )
        
        # Add capacity lines
        for runway_id, runway in self.runways.items():
            fig_utilization.add_hline(
                y=runway.max_operations_per_hour,
                line_dash="dash",
                annotation_text=f"{runway_id} Max Capacity"
            )
        
        figures['utilization'] = fig_utilization
        
        # 2. Schedule Changes Distribution
        fig_changes = px.histogram(
            df_optimized,
            x='Schedule_Change_Minutes',
            nbins=20,
            title='Distribution of Schedule Changes',
            labels={'Schedule_Change_Minutes': 'Schedule Change (minutes)'}
        )
        figures['schedule_changes'] = fig_changes
        
        # 3. Priority vs. Runway Assignment
        priority_labels = {1: 'Emergency', 2: 'International', 3: 'Domestic Major', 
                          4: 'Domestic Regional', 5: 'Cargo'}
        df_optimized['Priority_Label'] = df_optimized['Priority'].map(priority_labels)
        
        priority_runway = df_optimized.groupby(['Priority_Label', 'Optimized_Runway']).size().reset_index(name='Count')
        
        fig_priority = px.bar(
            priority_runway,
            x='Priority_Label',
            y='Count',
            color='Optimized_Runway',
            title='Flight Priority vs. Runway Assignment',
            labels={'Count': 'Number of Flights', 'Priority_Label': 'Flight Priority'}
        )
        figures['priority_runway'] = fig_priority
        
        # 4. Aircraft Type Efficiency
        aircraft_efficiency = df_optimized.groupby('Aircraft_Category').agg({
            'Suitability_Score': 'mean',
            'Schedule_Change_Minutes': 'mean'
        }).reset_index()
        
        fig_efficiency = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average Suitability Score', 'Average Schedule Change'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig_efficiency.add_trace(
            go.Bar(x=aircraft_efficiency['Aircraft_Category'], 
                  y=aircraft_efficiency['Suitability_Score'],
                  name='Suitability Score'),
            row=1, col=1
        )
        
        fig_efficiency.add_trace(
            go.Bar(x=aircraft_efficiency['Aircraft_Category'], 
                  y=aircraft_efficiency['Schedule_Change_Minutes'],
                  name='Schedule Change (min)'),
            row=1, col=2
        )
        
        fig_efficiency.update_layout(title='Aircraft Type Efficiency Metrics')
        figures['aircraft_efficiency'] = fig_efficiency
        
        # 5. Runway Suitability Heatmap
        suitability_cols = [col for col in df_optimized.columns if col.startswith('Suitability_')]
        if suitability_cols:
            suitability_data = df_optimized[['Aircraft_Category'] + suitability_cols]
            suitability_avg = suitability_data.groupby('Aircraft_Category').mean()
            
            # Rename columns for better display
            suitability_avg.columns = [col.replace('Suitability_', '') for col in suitability_avg.columns]
            
            fig_heatmap = px.imshow(
                suitability_avg.values,
                x=suitability_avg.columns,
                y=suitability_avg.index,
                title='Aircraft Category vs. Runway Suitability',
                color_continuous_scale='RdYlGn',
                aspect='auto'
            )
            figures['suitability_heatmap'] = fig_heatmap
        
        return figures
