"""
Peak Time Analysis Module
Identifies busiest slots using clustering and recommends schedule adjustments.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PeakTimeAnalyzer:
    """
    Analyzes flight schedule patterns to identify peak times and recommend optimizations.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.peak_clusters = None
        self.peak_recommendations = []
        
    def analyze_hourly_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze hourly flight patterns and congestion levels.
        
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
            'Delay_Minutes': ['mean', 'std', 'max'],
            'Capacity': 'sum',
            'Runway': 'nunique'
        }).round(2)
        
        # Flatten column names
        hourly_stats.columns = ['_'.join(col).strip() for col in hourly_stats.columns]
        hourly_stats = hourly_stats.reset_index()
        
        # Rename columns for clarity
        hourly_stats.rename(columns={
            'Flight_ID_count': 'Flight_Count',
            'Delay_Minutes_mean': 'Avg_Delay',
            'Delay_Minutes_std': 'Delay_Std',
            'Delay_Minutes_max': 'Max_Delay',
            'Capacity_sum': 'Total_Passengers',
            'Runway_nunique': 'Active_Runways'
        }, inplace=True)
        
        # Calculate congestion metrics
        hourly_stats['Flights_per_Runway'] = hourly_stats['Flight_Count'] / hourly_stats['Active_Runways']
        hourly_stats['Congestion_Score'] = (
            hourly_stats['Flight_Count'] * 0.4 + 
            hourly_stats['Avg_Delay'] * 0.3 + 
            hourly_stats['Flights_per_Runway'] * 0.3
        )
        
        return hourly_stats
    
    def perform_peak_clustering(self, hourly_stats: pd.DataFrame, n_clusters: int = 4) -> Dict:
        """
        Cluster hours based on congestion patterns.
        
        Args:
            hourly_stats: Hourly statistics DataFrame
            n_clusters: Number of clusters for K-means
            
        Returns:
            Dictionary with clustering results
        """
        # Prepare features for clustering
        features = ['Flight_Count', 'Avg_Delay', 'Flights_per_Runway', 'Total_Passengers']
        X = hourly_stats[features].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        hourly_stats['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, hourly_stats['Cluster'])
        
        # DBSCAN for anomaly detection
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        hourly_stats['Anomaly_Cluster'] = dbscan.fit_predict(X_scaled)
        
        # Label clusters based on congestion level
        cluster_stats = hourly_stats.groupby('Cluster').agg({
            'Congestion_Score': 'mean',
            'Flight_Count': 'mean',
            'Avg_Delay': 'mean'
        }).round(2)
        
        # Sort clusters by congestion score
        cluster_stats = cluster_stats.sort_values('Congestion_Score', ascending=False)
        cluster_labels = ['Super Peak', 'Peak', 'Moderate', 'Low Traffic'][:len(cluster_stats)]
        
        cluster_mapping = dict(zip(cluster_stats.index, cluster_labels))
        hourly_stats['Peak_Category'] = hourly_stats['Cluster'].map(cluster_mapping)
        
        self.peak_clusters = {
            'hourly_stats': hourly_stats,
            'cluster_stats': cluster_stats,
            'cluster_mapping': cluster_mapping,
            'silhouette_score': silhouette_avg,
            'kmeans_model': kmeans
        }
        
        return self.peak_clusters
    
    def generate_peak_recommendations(self, hourly_stats: pd.DataFrame) -> List[Dict]:
        """
        Generate recommendations for schedule adjustments based on peak analysis.
        
        Args:
            hourly_stats: DataFrame with clustering results
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Identify super peak hours
        super_peak_hours = hourly_stats[
            hourly_stats['Peak_Category'] == 'Super Peak'
        ]['Hour'].unique()
        
        for hour in super_peak_hours:
            hour_data = hourly_stats[hourly_stats['Hour'] == hour]
            avg_flights = hour_data['Flight_Count'].mean()
            avg_delay = hour_data['Avg_Delay'].mean()
            
            # Recommendation 1: Redistribute flights
            if avg_flights > 25:  # High flight volume
                recommendations.append({
                    'type': 'redistribution',
                    'hour': hour,
                    'severity': 'high',
                    'title': f'Redistribute flights from {hour:02d}:00',
                    'description': f'Move {int(avg_flights * 0.2)} flights to adjacent hours',
                    'impact': f'Expected delay reduction: {avg_delay * 0.15:.1f} minutes',
                    'priority': 1
                })
            
            # Recommendation 2: Add runway capacity
            if avg_delay > 15:  # High delays
                recommendations.append({
                    'type': 'capacity',
                    'hour': hour,
                    'severity': 'medium',
                    'title': f'Increase runway efficiency at {hour:02d}:00',
                    'description': 'Deploy additional ground crew and optimize runway assignments',
                    'impact': f'Expected delay reduction: {avg_delay * 0.25:.1f} minutes',
                    'priority': 2
                })
        
        # Identify underutilized hours
        low_traffic_hours = hourly_stats[
            hourly_stats['Peak_Category'] == 'Low Traffic'
        ]['Hour'].unique()
        
        for hour in low_traffic_hours:
            hour_data = hourly_stats[hourly_stats['Hour'] == hour]
            avg_flights = hour_data['Flight_Count'].mean()
            
            if avg_flights < 10:  # Very low utilization
                recommendations.append({
                    'type': 'utilization',
                    'hour': hour,
                    'severity': 'low',
                    'title': f'Opportunity: Utilize {hour:02d}:00 slot',
                    'description': f'Can accommodate {20 - avg_flights:.0f} additional flights',
                    'impact': 'Reduce congestion in peak hours',
                    'priority': 3
                })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        self.peak_recommendations = recommendations
        
        return recommendations
    
    def create_peak_analysis_dashboard(self, hourly_stats: pd.DataFrame) -> Dict:
        """
        Create comprehensive visualization dashboard for peak analysis.
        
        Args:
            hourly_stats: DataFrame with clustering results
            
        Returns:
            Dictionary containing plotly figures
        """
        figures = {}
        
        # 1. Hourly Congestion Heatmap
        pivot_data = hourly_stats.pivot(index='Day_of_Week', 
                                       columns='Hour', 
                                       values='Congestion_Score')
        
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=list(range(24)),
            y=day_labels,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Congestion Score"),
            hoveremplate='<b>%{y}</b><br>Hour: %{x}:00<br>Congestion: %{z:.2f}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title='Weekly Congestion Heatmap',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )
        figures['heatmap'] = fig_heatmap
        
        # 2. Peak Category Distribution
        fig_cluster = px.scatter(
            hourly_stats, 
            x='Flight_Count', 
            y='Avg_Delay',
            color='Peak_Category',
            size='Total_Passengers',
            hover_data=['Hour', 'Day_of_Week'],
            title='Flight Patterns by Peak Category'
        )
        figures['clusters'] = fig_cluster
        
        # 3. Hourly Flight Volume and Delays
        hourly_avg = hourly_stats.groupby('Hour').agg({
            'Flight_Count': 'mean',
            'Avg_Delay': 'mean',
            'Peak_Category': lambda x: x.mode()[0] if not x.empty else 'Unknown'
        }).reset_index()
        
        fig_hourly = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Average Flight Count by Hour', 'Average Delay by Hour'),
            shared_xaxes=True
        )
        
        # Flight count
        fig_hourly.add_trace(
            go.Bar(x=hourly_avg['Hour'], 
                  y=hourly_avg['Flight_Count'],
                  name='Flight Count',
                  marker_color='lightblue'),
            row=1, col=1
        )
        
        # Delays
        fig_hourly.add_trace(
            go.Scatter(x=hourly_avg['Hour'], 
                      y=hourly_avg['Avg_Delay'],
                      mode='lines+markers',
                      name='Avg Delay',
                      line=dict(color='red', width=3)),
            row=2, col=1
        )
        
        fig_hourly.update_layout(
            title='Hourly Traffic Patterns',
            height=600,
            showlegend=False
        )
        
        fig_hourly.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig_hourly.update_yaxes(title_text="Flight Count", row=1, col=1)
        fig_hourly.update_yaxes(title_text="Delay (min)", row=2, col=1)
        
        figures['hourly_patterns'] = fig_hourly
        
        # 4. Peak Recommendations Summary
        if self.peak_recommendations:
            rec_df = pd.DataFrame(self.peak_recommendations)
            fig_recommendations = px.bar(
                rec_df.groupby('type').size().reset_index(name='count'),
                x='type',
                y='count',
                title='Optimization Recommendations by Type',
                color='type'
            )
            figures['recommendations'] = fig_recommendations
        
        return figures
    
    def get_optimization_opportunities(self, hourly_stats: pd.DataFrame) -> Dict:
        """
        Identify specific optimization opportunities.
        
        Args:
            hourly_stats: DataFrame with clustering results
            
        Returns:
            Dictionary with optimization metrics
        """
        # Calculate potential improvements
        super_peak = hourly_stats[hourly_stats['Peak_Category'] == 'Super Peak']
        low_traffic = hourly_stats[hourly_stats['Peak_Category'] == 'Low Traffic']
        
        # Redistribution potential
        excess_flights = super_peak['Flight_Count'].sum() * 0.2  # 20% redistribution
        available_capacity = (20 - low_traffic['Flight_Count']).sum()
        redistribution_feasible = min(excess_flights, available_capacity)
        
        # Delay reduction potential
        current_avg_delay = hourly_stats['Avg_Delay'].mean()
        peak_delay_reduction = super_peak['Avg_Delay'].mean() * 0.25  # 25% reduction in peaks
        
        # Throughput improvement
        current_throughput = hourly_stats['Flight_Count'].sum()
        additional_throughput = redistribution_feasible
        throughput_improvement = (additional_throughput / current_throughput) * 100
        
        return {
            'current_avg_delay': current_avg_delay,
            'potential_delay_reduction': peak_delay_reduction,
            'delay_improvement_pct': (peak_delay_reduction / current_avg_delay) * 100,
            'redistribution_feasible': redistribution_feasible,
            'throughput_improvement_pct': throughput_improvement,
            'peak_hours_count': len(super_peak['Hour'].unique()),
            'optimization_score': min(95, 
                (peak_delay_reduction / current_avg_delay + 
                 throughput_improvement / 100) * 50)
        }
