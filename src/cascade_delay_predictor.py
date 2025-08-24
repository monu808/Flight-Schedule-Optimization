"""
Cascade Delay Prediction Module
Uses graph networks to model delay propagation across the flight network.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FlightNode:
    """Represents a flight in the delay propagation network."""
    flight_id: str
    airline: str
    aircraft_id: str
    scheduled_time: datetime
    origin: str
    destination: str
    delay: float
    capacity: int
    runway: str
    is_international: bool = False
    connecting_flights: List[str] = None
    
    def __post_init__(self):
        if self.connecting_flights is None:
            self.connecting_flights = []

@dataclass
class DelayPropagation:
    """Represents delay propagation between flights."""
    source_flight: str
    target_flight: str
    propagation_factor: float  # 0-1, how much delay propagates
    connection_type: str  # 'aircraft', 'crew', 'passenger', 'runway'
    time_buffer: float  # minutes between flights
    
class CascadeDelayPredictor:
    """
    Predicts how delays cascade through the flight network using graph analysis.
    """
    
    def __init__(self):
        self.flight_graph = nx.DiGraph()
        self.flights_dict = {}
        self.propagation_model = None
        self.cascade_predictions = {}
        
    def build_flight_network(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build a directed graph representing the flight network.
        
        Args:
            df: Flight schedule DataFrame
            
        Returns:
            NetworkX directed graph
        """
        # Convert DataFrame to FlightNode objects
        flights = []
        for _, row in df.iterrows():
            flight = FlightNode(
                flight_id=row['Flight_ID'],
                airline=row['Airline'],
                aircraft_id=row['Aircraft_ID'],
                scheduled_time=pd.to_datetime(row['Scheduled_Time']),
                origin=row.get('Origin', 'Unknown'),
                destination=row['Destination'],
                delay=row.get('Delay_Minutes', 0),
                capacity=row['Capacity'],
                runway=row['Runway'],
                is_international=row.get('Is_International', False)
            )
            flights.append(flight)
            self.flights_dict[flight.flight_id] = flight
        
        # Add flight nodes to graph
        for flight in flights:
            self.flight_graph.add_node(
                flight.flight_id,
                **{
                    'airline': flight.airline,
                    'aircraft_id': flight.aircraft_id,
                    'scheduled_time': flight.scheduled_time,
                    'origin': flight.origin,
                    'destination': flight.destination,
                    'delay': flight.delay,
                    'capacity': flight.capacity,
                    'runway': flight.runway,
                    'is_international': flight.is_international
                }
            )
        
        # Add edges based on different connection types
        self._add_aircraft_connections(flights)
        self._add_crew_connections(flights)
        self._add_runway_connections(flights)
        self._add_passenger_connections(flights)
        
        return self.flight_graph
    
    def _add_aircraft_connections(self, flights: List[FlightNode]):
        """Add edges for aircraft turnaround connections."""
        # Sort flights by aircraft and time
        aircraft_flights = {}
        for flight in flights:
            if flight.aircraft_id not in aircraft_flights:
                aircraft_flights[flight.aircraft_id] = []
            aircraft_flights[flight.aircraft_id].append(flight)
        
        for aircraft_id, flight_list in aircraft_flights.items():
            flight_list.sort(key=lambda x: x.scheduled_time)
            
            for i in range(len(flight_list) - 1):
                current_flight = flight_list[i]
                next_flight = flight_list[i + 1]
                
                # Check if turnaround is at same airport
                if current_flight.destination == next_flight.origin:
                    time_buffer = (next_flight.scheduled_time - 
                                 current_flight.scheduled_time).total_seconds() / 60
                    
                    # Calculate propagation factor based on turnaround time
                    if time_buffer < 60:  # Less than 1 hour
                        propagation_factor = 0.9
                    elif time_buffer < 120:  # 1-2 hours
                        propagation_factor = 0.7
                    elif time_buffer < 180:  # 2-3 hours
                        propagation_factor = 0.4
                    else:  # More than 3 hours
                        propagation_factor = 0.1
                    
                    self.flight_graph.add_edge(
                        current_flight.flight_id,
                        next_flight.flight_id,
                        connection_type='aircraft',
                        propagation_factor=propagation_factor,
                        time_buffer=time_buffer
                    )
    
    def _add_crew_connections(self, flights: List[FlightNode]):
        """Add edges for crew connections (simplified - same airline)."""
        airline_flights = {}
        for flight in flights:
            if flight.airline not in airline_flights:
                airline_flights[flight.airline] = []
            airline_flights[flight.airline].append(flight)
        
        for airline, flight_list in airline_flights.items():
            flight_list.sort(key=lambda x: x.scheduled_time)
            
            # Add connections between consecutive flights of same airline
            # This is a simplified model - actual crew scheduling is more complex
            for i in range(len(flight_list) - 1):
                current_flight = flight_list[i]
                next_flight = flight_list[i + 1]
                
                time_diff = (next_flight.scheduled_time - 
                           current_flight.scheduled_time).total_seconds() / 60
                
                # Only connect if reasonable crew transfer time
                if 90 <= time_diff <= 480:  # 1.5 to 8 hours
                    propagation_factor = 0.3  # Lower than aircraft connections
                    
                    self.flight_graph.add_edge(
                        current_flight.flight_id,
                        next_flight.flight_id,
                        connection_type='crew',
                        propagation_factor=propagation_factor,
                        time_buffer=time_diff
                    )
    
    def _add_runway_connections(self, flights: List[FlightNode]):
        """Add edges for runway capacity constraints."""
        # Group flights by runway and time slots
        runway_slots = {}
        
        for flight in flights:
            runway = flight.runway
            # Create 30-minute time slots
            slot_time = flight.scheduled_time.replace(minute=0, second=0, microsecond=0)
            if flight.scheduled_time.minute >= 30:
                slot_time = slot_time.replace(minute=30)
            
            slot_key = (runway, slot_time)
            if slot_key not in runway_slots:
                runway_slots[slot_key] = []
            runway_slots[slot_key].append(flight)
        
        # Add connections between flights in same runway slot
        for slot_key, slot_flights in runway_slots.items():
            if len(slot_flights) > 1:
                slot_flights.sort(key=lambda x: x.scheduled_time)
                
                for i in range(len(slot_flights) - 1):
                    for j in range(i + 1, len(slot_flights)):
                        current_flight = slot_flights[i]
                        next_flight = slot_flights[j]
                        
                        # Add edge if flights are close in time
                        time_diff = (next_flight.scheduled_time - 
                                   current_flight.scheduled_time).total_seconds() / 60
                        
                        if time_diff <= 30:  # Within 30 minutes
                            propagation_factor = 0.6
                            
                            self.flight_graph.add_edge(
                                current_flight.flight_id,
                                next_flight.flight_id,
                                connection_type='runway',
                                propagation_factor=propagation_factor,
                                time_buffer=time_diff
                            )
    
    def _add_passenger_connections(self, flights: List[FlightNode]):
        """Add edges for passenger connections (simplified - hub connections)."""
        # Identify potential hub airports
        destination_counts = {}
        origin_counts = {}
        
        for flight in flights:
            destination_counts[flight.destination] = destination_counts.get(flight.destination, 0) + 1
            origin_counts[flight.origin] = origin_counts.get(flight.origin, 0) + 1
        
        # Airports with high traffic are potential hubs
        hubs = set()
        for airport, count in destination_counts.items():
            if count >= 10:  # Threshold for hub status
                hubs.add(airport)
        for airport, count in origin_counts.items():
            if count >= 10:
                hubs.add(airport)
        
        # Add passenger connections through hubs
        for hub in hubs:
            arriving_flights = [f for f in flights if f.destination == hub]
            departing_flights = [f for f in flights if f.origin == hub]
            
            for arr_flight in arriving_flights:
                for dep_flight in departing_flights:
                    time_diff = (dep_flight.scheduled_time - 
                               arr_flight.scheduled_time).total_seconds() / 60
                    
                    # Reasonable connection time for passengers
                    if 45 <= time_diff <= 240:  # 45 minutes to 4 hours
                        propagation_factor = 0.2  # Lower impact
                        
                        self.flight_graph.add_edge(
                            arr_flight.flight_id,
                            dep_flight.flight_id,
                            connection_type='passenger',
                            propagation_factor=propagation_factor,
                            time_buffer=time_diff
                        )
    
    def simulate_delay_cascade(self, initial_delays: Dict[str, float], 
                              max_iterations: int = 5) -> Dict[str, float]:
        """
        Simulate how delays cascade through the network.
        
        Args:
            initial_delays: Dictionary of {flight_id: delay_minutes}
            max_iterations: Maximum number of propagation iterations
            
        Returns:
            Dictionary with final delays for all flights
        """
        current_delays = {node: 0.0 for node in self.flight_graph.nodes()}
        current_delays.update(initial_delays)
        
        for iteration in range(max_iterations):
            new_delays = current_delays.copy()
            propagation_occurred = False
            
            for flight_id in self.flight_graph.nodes():
                if current_delays[flight_id] > 0:  # Flight has delay
                    # Propagate to connected flights
                    for successor in self.flight_graph.successors(flight_id):
                        edge_data = self.flight_graph[flight_id][successor]
                        propagation_factor = edge_data['propagation_factor']
                        time_buffer = edge_data['time_buffer']
                        
                        # Calculate propagated delay
                        propagated_delay = current_delays[flight_id] * propagation_factor
                        
                        # Reduce propagation if there's sufficient buffer time
                        if time_buffer > propagated_delay:
                            propagated_delay = max(0, propagated_delay - time_buffer * 0.1)
                        
                        # Update successor's delay
                        if propagated_delay > 0:
                            new_delays[successor] = max(new_delays[successor], 
                                                      current_delays[successor] + propagated_delay)
                            propagation_occurred = True
            
            current_delays = new_delays
            
            # Stop if no significant propagation occurred
            if not propagation_occurred:
                break
        
        self.cascade_predictions = current_delays
        return current_delays
    
    def analyze_network_vulnerability(self) -> Dict:
        """
        Analyze which flights/airports are most critical for delay propagation.
        
        Returns:
            Dictionary with vulnerability metrics
        """
        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(self.flight_graph)
        in_degree = dict(self.flight_graph.in_degree())
        out_degree = dict(self.flight_graph.out_degree())
        pagerank = nx.pagerank(self.flight_graph)
        
        # Identify critical flights
        critical_flights = []
        for flight_id in self.flight_graph.nodes():
            criticality_score = (
                betweenness.get(flight_id, 0) * 0.4 +
                (out_degree.get(flight_id, 0) / max(out_degree.values())) * 0.3 +
                pagerank.get(flight_id, 0) * 0.3
            )
            
            flight_data = self.flight_graph.nodes[flight_id]
            critical_flights.append({
                'flight_id': flight_id,
                'criticality_score': criticality_score,
                'betweenness': betweenness.get(flight_id, 0),
                'out_degree': out_degree.get(flight_id, 0),
                'pagerank': pagerank.get(flight_id, 0),
                'airline': flight_data['airline'],
                'origin': flight_data['origin'],
                'destination': flight_data['destination']
            })
        
        # Sort by criticality
        critical_flights.sort(key=lambda x: x['criticality_score'], reverse=True)
        
        # Analyze airport vulnerability
        airport_impact = {}
        for flight_id in self.flight_graph.nodes():
            flight_data = self.flight_graph.nodes[flight_id]
            airport = flight_data['origin']
            
            if airport not in airport_impact:
                airport_impact[airport] = {
                    'flights_count': 0,
                    'total_criticality': 0,
                    'avg_criticality': 0
                }
            
            flight_criticality = next(f['criticality_score'] for f in critical_flights 
                                    if f['flight_id'] == flight_id)
            airport_impact[airport]['flights_count'] += 1
            airport_impact[airport]['total_criticality'] += flight_criticality
        
        # Calculate average criticality per airport
        for airport in airport_impact:
            airport_impact[airport]['avg_criticality'] = (
                airport_impact[airport]['total_criticality'] / 
                airport_impact[airport]['flights_count']
            )
        
        return {
            'critical_flights': critical_flights[:20],  # Top 20
            'airport_vulnerability': airport_impact,
            'network_metrics': {
                'total_nodes': self.flight_graph.number_of_nodes(),
                'total_edges': self.flight_graph.number_of_edges(),
                'density': nx.density(self.flight_graph),
                'avg_clustering': nx.average_clustering(self.flight_graph.to_undirected())
            }
        }
    
    def analyze_cascade_impact(self, df: pd.DataFrame) -> Dict:
        """
        Analyze flights with highest cascading delay impact potential.
        
        Args:
            df: Flight schedule DataFrame
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Check if required columns exist
            required_cols = ['Flight_ID', 'Aircraft_ID', 'Scheduled_Time', 'Destination']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Build network from dataframe
            self.build_flight_network(df)
            
            # Calculate betweenness centrality to find critical nodes
            centrality = nx.betweenness_centrality(self.flight_graph)
            
            # Calculate degree metrics
            in_degree = dict(self.flight_graph.in_degree())
            out_degree = dict(self.flight_graph.out_degree())
            
            # Calculate impact score based on network metrics
            impact_scores = {}
            for node in self.flight_graph.nodes():
                # Impact score based on centrality and connectivity
                impact_scores[node] = (
                    centrality.get(node, 0) * 10 +  # Scale up centrality
                    in_degree.get(node, 0) * 0.5 +   # Incoming connections
                    out_degree.get(node, 0) * 2      # Outgoing connections matter more
                )
            
            # Rank flights by impact score
            critical_flights = []
            for flight_id, score in sorted(impact_scores.items(), key=lambda x: x[1], reverse=True):
                if flight_id in self.flights_dict:
                    flight = self.flights_dict[flight_id]
                    critical_flights.append({
                        'flight_id': flight_id,
                        'airline': flight.airline,
                        'aircraft_id': flight.aircraft_id,
                        'scheduled_time': flight.scheduled_time,
                        'origin': flight.origin,
                        'destination': flight.destination,
                        'impact_score': score
                    })
            
            return {
                'critical_flights': critical_flights[:20],  # Return top 20
                'total_flights': len(self.flight_graph.nodes()),
                'total_connections': len(self.flight_graph.edges())
            }
        except Exception as e:
            # Return informative error
            return {
                'error': str(e),
                'critical_flights': [],
                'total_flights': 0,
                'total_connections': 0
            }
    
    def predict_cascade_impact(self, delay_scenario: Dict[str, float]) -> Dict:
        """
        Predict the impact of a delay scenario on the entire network.
        
        Args:
            delay_scenario: Dictionary of {flight_id: delay_minutes}
            
        Returns:
            Impact analysis results
        """
        # Simulate cascade
        final_delays = self.simulate_delay_cascade(delay_scenario)
        
        # Calculate impact metrics
        total_initial_delay = sum(delay_scenario.values())
        total_final_delay = sum(final_delays.values())
        amplification_factor = total_final_delay / total_initial_delay if total_initial_delay > 0 else 0
        
        affected_flights = len([d for d in final_delays.values() if d > 0])
        
        # Categorize impact by delay level
        impact_levels = {
            'minor': len([d for d in final_delays.values() if 0 < d <= 15]),
            'moderate': len([d for d in final_delays.values() if 15 < d <= 30]),
            'severe': len([d for d in final_delays.values() if d > 30])
        }
        
        # Calculate passenger impact (simplified)
        passenger_impact = 0
        for flight_id, delay in final_delays.items():
            if delay > 0:
                capacity = self.flight_graph.nodes[flight_id]['capacity']
                passenger_impact += capacity * delay
        
        return {
            'initial_delay_minutes': total_initial_delay,
            'total_propagated_delay': total_final_delay,
            'amplification_factor': amplification_factor,
            'affected_flights_count': affected_flights,
            'impact_levels': impact_levels,
            'estimated_passenger_delay_hours': passenger_impact / 60,
            'final_delays': final_delays
        }
    
    def create_network_visualization(self) -> go.Figure:
        """
        Create interactive network visualization.
        
        Returns:
            Plotly graph object
        """
        # Calculate layout
        pos = nx.spring_layout(self.flight_graph, k=1, iterations=50)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.flight_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_data = self.flight_graph[edge[0]][edge[1]]
            edge_info.append(f"Connection: {edge_data['connection_type']}<br>"
                           f"Propagation: {edge_data['propagation_factor']:.2f}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in self.flight_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = self.flight_graph.nodes[node]
            node_text.append(f"Flight: {node}<br>"
                           f"Airline: {node_data['airline']}<br>"
                           f"Route: {node_data['origin']} → {node_data['destination']}<br>"
                           f"Delay: {node_data['delay']:.1f} min")
            
            # Color by delay level
            delay = node_data['delay']
            if delay == 0:
                node_color.append('green')
            elif delay <= 15:
                node_color.append('yellow')
            elif delay <= 30:
                node_color.append('orange')
            else:
                node_color.append('red')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='RdYlGn_r',
                reversescale=True,
                color=node_color,
                size=10,
                colorbar=dict(
                    thickness=15,
                    title=dict(
                        text="Delay Level",
                        side="right"
                    ),
                    xanchor='left'
                ),
                line=dict(width=2)
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Flight Network Delay Propagation',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text="Node color indicates delay level: Green=No delay, Yellow=Minor, Orange=Moderate, Red=Severe",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
