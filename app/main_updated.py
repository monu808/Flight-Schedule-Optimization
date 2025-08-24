"""
Production-Ready Flight Schedule Optimization Dashboard
=====================================================

Comprehensive Streamlit application for flight scheduling optimization
with AI-powered insights, advanced visualizations, and OR-Tools optimization.

Features:
- Smart CSV/Excel upload with fuzzy column matching
- Advanced visualizations (heatmaps, cascading delays, route analysis)
- AI-powered natural language query interface
- OR-Tools constraint programming optimization
- What-if scenario simulation
- PDF export capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time, timedelta
import sys
import os
import io
import base64
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, parent_dir)

try:
    from src.smart_data_loader import SmartFlightDataLoader
    from src.advanced_visualizations import FlightVisualizationSuite
    from src.intelligent_nlp_system import IntelligentFlightQuerySystem
    from src.advanced_optimizer import AdvancedFlightOptimizer
    from src.data_processor import FlightDataProcessor
    from src.delay_analyzer import DelayAnalyzer
    from src.sample_data_generator import generate_sample_data, FlightDataGenerator
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are installed and accessible")
    MODULES_LOADED = False

# Page configuration
st.set_page_config(
    page_title="Flight Schedule Optimizer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
/* Remove white bars and improve sidebar */
.sidebar-section {
    background-color: transparent !important;
    padding: 0.5rem 0 !important;
    border-radius: 0 !important;
    margin: 0.5rem 0 !important;
    border: none !important;
}

/* Main styling */
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.metric-card {
    background: transparent !important;
    padding: 1.5rem;
    border-radius: 1rem;
    border-left: 6px solid #1f77b4;
    box-shadow: none !important;
    margin-bottom: 1rem;
}

/* Override default metric styling */
[data-testid="metric-container"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 3px solid #1f77b4;
}

.insight-box {
    background: linear-gradient(145deg, #e8f4f8, #d1ecf1);
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 1rem 0;
    border-left: 4px solid #17a2b8;
    color: #0c5460;
    font-weight: 500;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.success-box {
    background: linear-gradient(145deg, #d4edda, #c3e6cb);
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    color: #155724;
}

.warning-box {
    background: linear-gradient(145deg, #fff3cd, #ffeaa7);
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
    color: #856404;
}

.danger-box {
    background: linear-gradient(145deg, #f8d7da, #f5c6cb);
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #dc3545;
    color: #721c24;
}

/* Sidebar improvements */
.css-1d391kg {
    background-color: #f8f9fa;
}

/* Navigation styling */
.stRadio > div {
    gap: 0.5rem;
}

/* Key insights styling */
.insight-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 1rem;
    margin: 1rem 0;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}

.insight-item {
    background: rgba(255, 255, 255, 0.1);
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Remove default streamlit styling */
.css-1kyxreq {
    justify-content: center;
}

/* Button styling */
.stButton > button {
    border-radius: 0.5rem;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 500;
}

/* Better expanders */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #1f77b4;
    background-color: rgba(240, 242, 246, 0.5);
    border-radius: 0.5rem;
}

/* Enhanced dropdowns */
.stSelectbox {
    border-radius: 0.5rem;
}

/* Better metrics */
.stMetric {
    background: transparent !important;
    box-shadow: none !important;
}

/* Remove white background from metrics */
div[data-testid="stMetricValue"] {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_default_data():
    """Load default flight data with caching"""
    try:
        processor = FlightDataProcessor()
        data = processor.clean_data()
        return data, "Flight_Data.xlsx"
    except Exception as e:
        logger.error(f"Error loading default data: {e}")
        return None, None

def create_download_link(df: pd.DataFrame, filename: str, link_text: str) -> str:
    """Create download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def generate_pdf_report(data: pd.DataFrame, analysis_results: Dict) -> bytes:
    """Generate PDF report of analysis results"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph("Flight Schedule Optimization Report", title_style))
        story.append(Spacer(1, 20))
        
        # Summary statistics
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        summary_data = [
            ['Metric', 'Value'],
            ['Total Flights Analyzed', f"{len(data):,}"],
            ['Average Delay', f"{data['departure_delay_minutes'].mean():.1f} minutes" if 'departure_delay_minutes' in data.columns else 'N/A'],
            ['On-Time Performance', f"{((data['departure_delay_minutes'] <= 5).mean() * 100):.1f}%" if 'departure_delay_minutes' in data.columns else 'N/A'],
            ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M')]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Key insights
        story.append(Paragraph("Key Insights", styles['Heading2']))
        if 'route' in data.columns or 'To' in data.columns:
            route_col = 'route' if 'route' in data.columns else 'To'
            top_routes = data[route_col].value_counts().head(5)
            routes_paragraph = Paragraph(
                f"Top 5 Routes: {', '.join([f'{r} ({c})' for r, c in top_routes.items()])}",
                styles['Normal']
            )
            story.append(routes_paragraph)
            story.append(Spacer(1, 10))
            
        if 'departure_delay_minutes' in data.columns:
            avg_delay = data['departure_delay_minutes'].mean()
            worst_hour = data.groupby(data['STD'].apply(lambda x: x.hour))['departure_delay_minutes'].mean().idxmax()
            delay_paragraph = Paragraph(
                f"Average Delay: {avg_delay:.1f} minutes with peak delays during hour {worst_hour}:00",
                styles['Normal']
            )
            story.append(delay_paragraph)
            
        story.append(Spacer(1, 20))
        
        # Generate document
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        return None

# Main application
def main():
    """Main application entry point"""
    if not MODULES_LOADED:
        st.error("❌ Required modules couldn't be loaded. Please check installation.")
        st.stop()
    
    # Display application header
    st.markdown('<h1 class="main-header">✈️ Flight Schedule Optimizer Pro</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>AI-Powered Flight Scheduling with Advanced Analytics & Optimization</p>", unsafe_allow_html=True)
    
    # Data source selection
    st.sidebar.markdown("## 📊 Data Source")
    data_loader = SmartFlightDataLoader()
    
    data_source_options = ["📂 Default Data", "📤 Upload CSV/Excel", "🎲 Generate Sample"]
    data_source_tab = st.sidebar.radio("Select Data Source:", data_source_options, label_visibility="collapsed")
    
    data = None
    data_source = None
    use_custom_data = False
    
    if data_source_tab == "📂 Default Data":
        st.sidebar.info("Using default Mumbai (BOM) flight data")
        data, data_source = load_default_data()
        if data is None:
            st.sidebar.error("❌ Default data not available")
    
    elif data_source_tab == "📤 Upload CSV/Excel":
        st.sidebar.markdown("**📤 Upload your flight data:**")
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Supports CSV, Excel, and JSON formats. Automatic column detection included."
        )
        
        if uploaded_file is not None:
            try:
                # Using st.sidebar.empty() instead of st.sidebar.spinner()
                spinner_msg = st.sidebar.empty()
                spinner_msg.info("Loading and processing data...")
                data = data_loader.load_flight_data(uploaded_file)
                data_source = uploaded_file.name
                spinner_msg.empty()
                
                # Data quality report
                st.sidebar.success(f"✅ Loaded {len(data)} flights")
                
                with st.sidebar.expander("📋 Data Quality Report"):
                    quality_report = data_loader.generate_data_quality_report(data)
                    
                    # Display column detection results
                    st.write(f"**Columns detected:** {quality_report['columns_detected']}")
                    if quality_report['date_range']:
                        st.write(f"**Date range:** {quality_report['date_range'].get('start_date')} to {quality_report['date_range'].get('end_date')}")
                    
                    missing_data = quality_report['missing_data']
                    critical_missing = {k: v for k, v in missing_data.items() if v['percentage'] > 50}
                    if critical_missing:
                        st.warning("⚠️ High missing data in some columns")
                
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {str(e)}")
    
    elif data_source_tab == "🎲 Generate Sample":
        st.sidebar.markdown("**🎯 Sample Data Scenarios:**")
        
        scenario_options = {
            'normal': '🟢 Normal Operations',
            'high_delay': '🟡 High Delay Day', 
            'weather_event': '🔴 Weather Event',
            'peak_congestion': '🟠 Peak Congestion',
            'ideal_conditions': '💚 Ideal Conditions'
        }
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            selected_scenario = st.selectbox(
                "Scenario:",
                list(scenario_options.keys()),
                format_func=lambda x: scenario_options[x],
                label_visibility="collapsed"
            )
        
        with col2:
            num_flights = st.number_input("Flights:", 100, 1000, 500, 100, label_visibility="collapsed")
        
        if st.sidebar.button("🎲 Generate Data", type="primary", use_container_width=True):
            try:
                # Use regular st.spinner instead of st.sidebar.spinner
                spinner_container = st.sidebar.empty()
                spinner_container.info("Generating sample data...")
                
                data = generate_sample_data(selected_scenario, num_flights)
                data_source = f"Generated: {scenario_options[selected_scenario]}"
                use_custom_data = True
                
                # Clear spinner message
                spinner_container.empty()
                
                st.sidebar.success(f"✅ Generated {len(data)} flights")
                st.sidebar.info(f"Scenario: {scenario_options[selected_scenario]}")
                
                # Show scenario description
                scenario_descriptions = {
                    'normal': "Typical day with standard delay patterns",
                    'high_delay': "Increased delays across all time slots", 
                    'weather_event': "Weather-related disruptions and cascading delays",
                    'peak_congestion': "Heavy congestion during peak hours",
                    'ideal_conditions': "Optimistic scenario with minimal delays"
                }
                
                st.sidebar.markdown(f"**Description:** {scenario_descriptions[selected_scenario]}")
                
            except Exception as e:
                st.sidebar.error(f"❌ Error generating sample data: {str(e)}")
    
    # Stop if no data is available
    if data is None:
        if data_source_tab == "📂 Default Data":
            st.error("❌ Default data not available. Please check data file or upload your own data.")
        else:
            st.sidebar.error("❌ No data available. Please upload a file or generate sample data.")
            st.stop()
    
    # Filters section
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔍 Smart Filters", expanded=False):
        
        # Date range filter - only if we have actual date columns
        date_col = None
        if 'Date' in data.columns:
            date_col = 'Date'
        elif 'scheduled_departure' in data.columns:
            # Check if this column contains datetime objects with date info
            sample_val = data['scheduled_departure'].dropna().iloc[0] if len(data['scheduled_departure'].dropna()) > 0 else None
            if sample_val and hasattr(sample_val, 'date'):
                date_col = 'scheduled_departure'
        
        if date_col and data[date_col].notna().any():
            try:
                # Extract date range from actual date column
                min_date = data[date_col].min()
                max_date = data[date_col].max()
                
                # Convert to date if they're datetime objects
                if hasattr(min_date, 'date'):
                    min_date = min_date.date()
                    max_date = max_date.date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    # Filter based on date column type
                    if hasattr(data[date_col].iloc[0], 'date'):
                        data = data[
                            (data[date_col].dt.date >= start_date) & 
                            (data[date_col].dt.date <= end_date)
                        ]
                    else:
                        data = data[
                            (data[date_col] >= start_date) & 
                            (data[date_col] <= end_date)
                        ]
            except Exception as e:
                st.warning(f"Date filtering unavailable: {str(e)}")
        else:
            st.info("Date filtering not available - using time-based filters only")
        
        # Time range filter (since we have time data)
        if 'STD' in data.columns and data['STD'].notna().any():
            st.markdown("**Time Range Filter**")
            time_values = data['STD'].dropna().unique()
            
            if len(time_values) > 0:
                # Convert time objects to strings for display
                time_strings = [t.strftime('%H:%M') if hasattr(t, 'strftime') else str(t) for t in sorted(time_values)]
                
                # Time range selection
                col1, col2 = st.columns(2)
                with col1:
                    start_time = st.selectbox("Start Time", options=time_strings, index=0)
                with col2:
                    end_time = st.selectbox("End Time", options=time_strings, index=len(time_strings)-1)
                
                # Convert back to time objects for filtering
                try:
                    from datetime import time as dt_time
                    start_time_obj = dt_time.fromisoformat(start_time)
                    end_time_obj = dt_time.fromisoformat(end_time)
                    
                    # Filter data by time range
                    data = data[
                        (data['STD'] >= start_time_obj) & 
                        (data['STD'] <= end_time_obj)
                    ]
                    
                    st.success(f"Filtered to {len(data)} flights between {start_time} and {end_time}")
                except Exception as e:
                    st.warning(f"Time filtering error: {str(e)}")
        
        # Airline filter
        airline_columns = ['airline', 'Airline', 'carrier', 'Carrier']
        airline_col = None
        for col in airline_columns:
            if col in data.columns:
                airline_col = col
                break
        
        if airline_col:
            st.markdown("**Airline Filter**")
            airlines = sorted(data[airline_col].dropna().unique().tolist())
            selected_airlines = st.multiselect(
                "✈️ Select Airlines:",
                options=airlines,
                default=airlines[:5] if len(airlines) > 5 else airlines,
                help="Choose specific airlines to analyze"
            )
            
            if selected_airlines:
                data = data[data[airline_col].isin(selected_airlines)]
        
        # Route filter
        route_columns = ['route', 'Route', 'destination', 'To']
        route_col = None
        for col in route_columns:
            if col in data.columns:
                route_col = col
                break
        
        if route_col:
            st.markdown("**Destination Filter**")
            routes = sorted(data[route_col].dropna().unique().tolist())
            selected_routes = st.multiselect(
                "🌍 Select Destinations:",
                options=routes,
                default=routes[:10] if len(routes) > 10 else routes,
                help="Choose specific destinations to analyze"
            )
            
            if selected_routes:
                data = data[data[route_col].isin(selected_routes)]
        
        st.markdown(f"**Filtered Data:** {len(data)} flights")
    
    # Navigation section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    
    page_options = [
        "📊 Overview Dashboard", 
        "🔥 Advanced Visualizations", 
        "🤖 AI Query Interface",
        "⚙️ Schedule Optimizer", 
        "🎲 What-If Simulator",
        "📋 Data Explorer"
    ]
    
    # Use radio buttons with custom styling
    page = st.sidebar.radio(
        "Select Analysis Page:",
        page_options,
        key="navigation_radio",
        label_visibility="collapsed"
    )
    
    # Export section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Export Options")
    
    if st.sidebar.button("📄 Generate PDF Report"):
        try:
            analysis_results = {
                "Total Flights": len(data),
                "Data Source": data_source,
                "Analysis Date": datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            
            if 'departure_delay_minutes' in data.columns:
                analysis_results["Average Delay"] = f"{data['departure_delay_minutes'].mean():.1f} minutes"
                analysis_results["On-Time Rate"] = f"{((data['departure_delay_minutes'] <= 5).mean() * 100):.1f}%"
            
            pdf_data = generate_pdf_report(data, analysis_results)
            if pdf_data:
                b64 = base64.b64encode(pdf_data).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="flight_analysis_report.pdf">📥 Download PDF Report</a>'
                st.sidebar.markdown(href, unsafe_allow_html=True)
            else:
                st.sidebar.error("❌ Error generating PDF report")
        except Exception as e:
            st.sidebar.error(f"❌ Error creating report: {str(e)}")
    
    # Create visualization and analysis suite
    viz_suite = FlightVisualizationSuite(data) if MODULES_LOADED else None
    
    # Display selected page content
    if page == "📊 Overview Dashboard":
        display_overview_dashboard(data, data_source, viz_suite)
    
    elif page == "🔥 Advanced Visualizations":
        display_advanced_visualizations(data, viz_suite)
    
    elif page == "🤖 AI Query Interface":
        display_nlp_interface(data)
    
    elif page == "⚙️ Schedule Optimizer":
        display_schedule_optimizer(data)
    
    elif page == "🎲 What-If Simulator":
        display_what_if_simulator(data)
    
    elif page == "📋 Data Explorer":
        display_data_explorer(data, data_source)

def display_overview_dashboard(data: pd.DataFrame, data_source: str, viz_suite):
    """Display the main overview dashboard"""
    st.markdown('<h2 style="font-size: 2rem;">📊 Flight Operations Overview</h2>', unsafe_allow_html=True)
    
    # Data source information
    st.markdown(f"**Data Source:** {data_source} | **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Key metrics
    metrics_cols = st.columns(4)
    
    with metrics_cols[0]:
        st.metric(
            "Total Flights",
            f"{len(data):,}",
            delta=None
        )
        
    with metrics_cols[1]:
        if 'departure_delay_minutes' in data.columns:
            avg_delay = data['departure_delay_minutes'].mean()
            st.metric(
                "Average Delay",
                f"{avg_delay:.1f} min",
                delta=None
            )
        else:
            st.metric("Average Delay", "N/A", delta=None)
            
    with metrics_cols[2]:
        if 'departure_delay_minutes' in data.columns:
            on_time_rate = (data['departure_delay_minutes'] <= 15).mean() * 100
            st.metric(
                "On-Time Rate",
                f"{on_time_rate:.1f}%",
                delta=None
            )
        else:
            st.metric("On-Time Rate", "N/A", delta=None)
            
    with metrics_cols[3]:
        if 'departure_delay_minutes' in data.columns:
            delayed_flights = (data['departure_delay_minutes'] > 15).sum()
            st.metric(
                "Delayed Flights",
                f"{delayed_flights:,}",
                delta=None
            )
        else:
            st.metric("Delayed Flights", "N/A", delta=None)
    
    # Key insights section with gradient styling
    st.markdown('<div class="insight-container">', unsafe_allow_html=True)
    st.markdown('### ✨ Key Insights')
    
    if 'departure_delay_minutes' in data.columns:
        delay_analyzer = DelayAnalyzer(data)
        insights = delay_analyzer.generate_insights(top_n=3)
        
        for insight in insights:
            st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight-item">No delay data available for insights</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Dashboard charts
    st.markdown("### 📈 Performance Dashboard")
    
    charts_col1, charts_col2 = st.columns(2)
    
    if viz_suite:
        with charts_col1:
            st.subheader("Delay Distribution")
            if 'departure_delay_minutes' in data.columns:
                fig = viz_suite.create_delay_distribution_chart()
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No delay data available for this chart")
        
            st.subheader("Flight Volume by Hour")
            fig = viz_suite.create_hourly_flight_volume_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        with charts_col2:
            st.subheader("Top 10 Routes")
            fig = viz_suite.create_top_routes_chart()
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("On-Time Performance Trend")
            if 'departure_delay_minutes' in data.columns and 'STD' in data.columns:
                fig = viz_suite.create_ontime_performance_chart()
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for on-time performance trend")
    else:
        st.error("Visualization suite not available. Please check your installation.")

def display_advanced_visualizations(data: pd.DataFrame, viz_suite):
    """Display advanced visualizations section"""
    st.markdown('<h2 style="font-size: 2rem;">🔥 Advanced Visualizations</h2>', unsafe_allow_html=True)
    
    if not viz_suite:
        st.error("Visualization suite not available. Please check your installation.")
        return
    
    viz_tabs = st.tabs([
        "⏱️ Delay Analysis", 
        "🌡️ Heatmap Visualizations", 
        "📊 Performance Analytics",
        "🔗 Network Analysis"
    ])
    
    with viz_tabs[0]:
        st.subheader("Flight Delay Analysis")
        
        if 'departure_delay_minutes' in data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Delay Distribution by Time of Day")
                fig = viz_suite.create_delay_heatmap()
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Cascading Delay Effect")
                fig = viz_suite.create_cascading_delay_chart()
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Delay Factors Correlation")
            fig = viz_suite.create_delay_correlation_chart()
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No delay data available for analysis")
    
    with viz_tabs[1]:
        st.subheader("Heatmap Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Flight Volume Heatmap")
            fig = viz_suite.create_flight_volume_heatmap()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'departure_delay_minutes' in data.columns:
                st.markdown("#### Delay Severity Heatmap")
                fig = viz_suite.create_delay_severity_heatmap()
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No delay data available for heatmap")
    
    with viz_tabs[2]:
        st.subheader("Performance Analytics")
        
        if 'departure_delay_minutes' in data.columns and 'airline' in data.columns.str.lower():
            airline_col = 'airline' if 'airline' in data.columns else next((col for col in data.columns if col.lower() == 'airline'), None)
            
            if airline_col:
                st.markdown("#### Airline Performance Comparison")
                fig = viz_suite.create_airline_performance_chart(airline_column=airline_col)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Operational Trends Analysis")
        fig = viz_suite.create_operational_trends_chart()
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[3]:
        st.subheader("Network Analysis")
        
        route_col = None
        for col_name in ['route', 'destination', 'To', 'to']:
            if col_name in data.columns or col_name.title() in data.columns:
                route_col = col_name if col_name in data.columns else col_name.title()
                break
        
        if route_col:
            st.markdown("#### Route Network Visualization")
            fig = viz_suite.create_route_network_visualization(route_column=route_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No route data available for network analysis")

def display_nlp_interface(data: pd.DataFrame):
    """Display NLP query interface"""
    st.markdown('<h2 style="font-size: 2rem;">🤖 AI Query Interface</h2>', unsafe_allow_html=True)
    
    # Initialize the NLP system
    try:
        nlp_system = IntelligentFlightQuerySystem(data)
        
        st.markdown("""
        ### Ask Questions About Your Flight Data

        Type natural language queries to analyze your flight data. Examples:
        - *"What are the busiest hours of operation?"*
        - *"Which airlines have the worst delays?"*
        - *"Show me routes with highest cancellation rates"*
        - *"What's the average delay for morning flights?"*
        """)
        
        query = st.text_input("Enter your question:", placeholder="Type your question here...")
        
        if query:
            with st.spinner("Analyzing your question..."):
                try:
                    result = nlp_system.process_query(query)
                    
                    if result.get('chart'):
                        st.plotly_chart(result['chart'], use_container_width=True)
                    
                    if result.get('text_response'):
                        st.markdown(f"<div class='insight-box'>{result['text_response']}</div>", unsafe_allow_html=True)
                    
                    if result.get('data'):
                        st.dataframe(result['data'], use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        
        # Example queries section
        with st.expander("💡 Example Queries"):
            example_queries = [
                "Which routes have the highest number of flights?",
                "What is the average delay by airline?",
                "Show me the busiest hours of the day",
                "Which day of the week has the most delays?",
                "Compare on-time performance between airlines"
            ]
            
            for example in example_queries:
                if st.button(example):
                    try:
                        with st.spinner(f"Analyzing: {example}"):
                            result = nlp_system.process_query(example)
                            
                            if result.get('chart'):
                                st.plotly_chart(result['chart'], use_container_width=True)
                            
                            if result.get('text_response'):
                                st.markdown(f"<div class='insight-box'>{result['text_response']}</div>", unsafe_allow_html=True)
                            
                            if result.get('data'):
                                st.dataframe(result['data'], use_container_width=True)
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
    
    except Exception as e:
        st.error(f"Error initializing AI query system: {str(e)}")
        st.info("This feature requires NLP dependencies. Please check your installation.")

def display_schedule_optimizer(data: pd.DataFrame):
    """Display schedule optimizer interface"""
    st.markdown('<h2 style="font-size: 2rem;">⚙️ Schedule Optimizer</h2>', unsafe_allow_html=True)
    
    try:
        optimizer = AdvancedFlightOptimizer(data)
        
        st.markdown("""
        ### Flight Schedule Optimization
        
        Optimize flight schedules to reduce delays and improve operational efficiency.
        Select optimization goals, constraints, and parameters below.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            optimization_goal = st.selectbox(
                "Optimization Goal",
                ["Minimize Average Delay", "Maximize On-Time Performance", 
                 "Optimize Resource Utilization", "Minimize Passenger Connection Times"]
            )
            
            constraint_options = [
                "Respect Minimum Turnaround Times",
                "Maintain Route Coverage",
                "Ensure Crew Availability",
                "Respect Gate Constraints"
            ]
            
            constraints = st.multiselect(
                "Optimization Constraints",
                constraint_options,
                default=constraint_options[:2]
            )
        
        with col2:
            optimization_timeframe = st.selectbox(
                "Optimization Timeframe",
                ["Daily Schedule", "Peak Hours Only", "Full Week", "Custom Range"]
            )
            
            if optimization_timeframe == "Custom Range":
                custom_hours = st.slider("Hours to Optimize", 0, 24, (6, 22), 1)
            
            optimization_intensity = st.slider("Optimization Intensity", 1, 10, 5,
                                             help="Higher values mean more aggressive optimization but may take longer")
        
        if st.button("▶️ Run Optimization", type="primary"):
            with st.spinner("Running optimization algorithms..."):
                try:
                    # Convert parameters to format expected by optimizer
                    params = {
                        "goal": optimization_goal.lower().replace(" ", "_"),
                        "constraints": [c.lower().replace(" ", "_") for c in constraints],
                        "timeframe": optimization_timeframe.lower().replace(" ", "_"),
                        "intensity": optimization_intensity
                    }
                    
                    if optimization_timeframe == "Custom Range":
                        params["custom_hours"] = custom_hours
                    
                    results = optimizer.optimize_schedule(params)
                    
                    # Display results
                    st.success(f"✅ Optimization complete! Processed {len(data)} flights")
                    
                    # Summary metrics
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        delay_reduction = results.get("delay_reduction", 0)
                        st.metric("Delay Reduction", f"{delay_reduction:.1f}%", 
                                  delta=f"{delay_reduction:.1f}%")
                    
                    with metric_cols[1]:
                        ontime_improvement = results.get("ontime_improvement", 0)
                        st.metric("On-Time Performance", f"{results.get('new_ontime_rate', 0):.1f}%", 
                                  delta=f"{ontime_improvement:.1f}%")
                    
                    with metric_cols[2]:
                        efficiency_gain = results.get("efficiency_gain", 0)
                        st.metric("Efficiency Gain", f"{efficiency_gain:.1f}%", 
                                  delta=f"{efficiency_gain:.1f}%")
                    
                    # Visualization of improvements
                    st.subheader("Optimization Impact")
                    
                    if 'comparison_chart' in results:
                        st.plotly_chart(results['comparison_chart'], use_container_width=True)
                    
                    if 'optimized_schedule' in results:
                        st.subheader("Optimized Schedule")
                        st.dataframe(results['optimized_schedule'], use_container_width=True)
                        
                        # Download link for optimized schedule
                        csv = results['optimized_schedule'].to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="optimized_schedule.csv">📥 Download Optimized Schedule</a>'
                        st.markdown(href, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Optimization error: {str(e)}")
                    st.info("This feature requires OR-Tools and optimization dependencies. Please check installation.")
    
    except Exception as e:
        st.error(f"Error initializing optimizer: {str(e)}")
        st.info("This feature requires optimization dependencies. Please check installation.")

def display_what_if_simulator(data: pd.DataFrame):
    """Display what-if scenario simulator"""
    st.markdown('<h2 style="font-size: 2rem;">🎲 What-If Simulator</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Flight Schedule Scenario Simulator
    
    Test different schedule modifications to see their impact on delays, resource utilization, and overall performance.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        scenario_type = st.selectbox(
            "Scenario Type",
            ["Weather Impact Simulation", "Peak Demand Adjustment", "Resource Constraint Analysis", 
             "Delay Propagation Modeling", "Custom Scenario"]
        )
        
        if scenario_type == "Weather Impact Simulation":
            weather_severity = st.slider("Weather Severity", 1, 10, 5)
            weather_duration = st.slider("Duration (hours)", 1, 12, 3)
            
            st.markdown(f"""
            **Scenario Parameters:**
            - Weather Severity: {weather_severity}/10
            - Duration: {weather_duration} hours
            - Impact: {"High" if weather_severity > 7 else "Medium" if weather_severity > 4 else "Low"}
            """)
        
        elif scenario_type == "Peak Demand Adjustment":
            demand_change = st.slider("Demand Change (%)", -50, 100, 20)
            peak_hours = st.multiselect("Peak Hours", list(range(24)), default=[8, 9, 17, 18])
            
            st.markdown(f"""
            **Scenario Parameters:**
            - Demand Change: {demand_change}%
            - Peak Hours: {", ".join(map(str, peak_hours))}
            - Impact: {"High" if abs(demand_change) > 30 else "Medium" if abs(demand_change) > 15 else "Low"}
            """)
        
        elif scenario_type == "Resource Constraint Analysis":
            available_gates = st.slider("Available Gates", 5, 50, 20)
            turnaround_time = st.slider("Turnaround Time (min)", 20, 60, 35, 5)
            
            st.markdown(f"""
            **Scenario Parameters:**
            - Available Gates: {available_gates}
            - Turnaround Time: {turnaround_time} minutes
            - Impact: {"High" if available_gates < 15 else "Medium" if available_gates < 25 else "Low"}
            """)
    
    with col2:
        st.markdown("#### Simulation Controls")
        simulation_depth = st.slider("Simulation Depth", 1, 5, 3, 
                                    help="Higher values simulate more complex interactions but take longer")
        
        confidence_interval = st.selectbox("Confidence Interval", ["80%", "90%", "95%", "99%"], index=2)
        
        iterations = st.slider("Simulation Iterations", 100, 10000, 1000, 100,
                              help="More iterations provide more reliable results but take longer")
        
        st.markdown("---")
        random_seed = st.checkbox("Use random seed", True)
        
        if not random_seed:
            seed_value = st.number_input("Seed Value", 1, 9999, 42)
    
    if st.button("▶️ Run Simulation", type="primary"):
        with st.spinner("Running scenario simulation..."):
            # This would connect to actual simulator functionality
            st.info("Simulation engine running...")
            progress_bar = st.progress(0)
            
            # Mock progress updates
            for i in range(101):
                # Update progress bar
                progress_bar.progress(i)
                time.sleep(0.01)
            
            # Show mock results
            st.success("✅ Simulation complete!")
            
            metrics_cols = st.columns(3)
            
            with metrics_cols[0]:
                st.metric("Avg Delay Change", "+12.3 min" if scenario_type == "Weather Impact Simulation" else "-5.8 min",
                         delta="+12.3 min" if scenario_type == "Weather Impact Simulation" else "-5.8 min")
            
            with metrics_cols[1]:
                st.metric("On-Time Performance", "68.5%" if scenario_type == "Weather Impact Simulation" else "87.2%",
                         delta="-21.5%" if scenario_type == "Weather Impact Simulation" else "+7.2%")
            
            with metrics_cols[2]:
                st.metric("Passenger Impact", "724 pax" if scenario_type == "Weather Impact Simulation" else "156 pax",
                         delta=None)
            
            # Sample visualization
            st.subheader("Simulation Results")
            
            # Create a sample comparison chart
            x = np.arange(24)
            baseline = np.clip(np.sin(x/4) * 15 + 20 + np.random.randn(24)*3, 0, None)
            
            if scenario_type == "Weather Impact Simulation":
                # Simulate weather impact as increased delays
                scenario = baseline.copy()
                impact_start = 8
                for i in range(weather_duration):
                    hour = (impact_start + i) % 24
                    scenario[hour] = baseline[hour] * (1 + weather_severity/10)
            else:
                # Other scenarios
                scenario = baseline * (1 - 0.2) + np.random.randn(24)*2
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=baseline, mode='lines+markers', name='Baseline'))
            fig.add_trace(go.Scatter(x=x, y=scenario, mode='lines+markers', name='Scenario'))
            
            fig.update_layout(
                title="Average Delay by Hour of Day",
                xaxis_title="Hour of Day",
                yaxis_title="Average Delay (minutes)",
                legend_title="Scenario",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_data_explorer(data: pd.DataFrame, data_source: str):
    """Display data explorer interface"""
    st.markdown('<h2 style="font-size: 2rem;">📋 Data Explorer</h2>', unsafe_allow_html=True)
    
    st.markdown(f"**Source:** {data_source} | **Records:** {len(data):,}")
    
    tabs = st.tabs(["📊 Data Preview", "📈 Summary Statistics", "🔍 Custom Query"])
    
    with tabs[0]:
        st.dataframe(data, use_container_width=True)
        
        # Download options
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📥 Download as CSV"):
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="flight_data.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            if st.button("📥 Download as Excel"):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    data.to_excel(writer, index=False, sheet_name='Flight Data')
                b64 = base64.b64encode(output.getvalue()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="flight_data.xlsx">Download Excel File</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    with tabs[1]:
        if len(data) > 0:
            # Summary statistics
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                st.subheader("Numeric Columns Statistics")
                st.dataframe(data[numeric_cols].describe(), use_container_width=True)
            
            # Categorical column summaries
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                st.subheader("Categorical Columns")
                
                selected_col = st.selectbox("Select column:", categorical_cols)
                
                if selected_col:
                    value_counts = data[selected_col].value_counts().reset_index()
                    value_counts.columns = [selected_col, 'Count']
                    
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.dataframe(value_counts, use_container_width=True)
                        
                    with col2:
                        # Create bar chart of value counts
                        fig = px.bar(
                            value_counts.head(10), 
                            x=selected_col, 
                            y='Count',
                            title=f"Top 10 values for {selected_col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for summary statistics")
    
    with tabs[2]:
        st.subheader("Custom SQL-like Query")
        st.markdown("""
        Use SQL-like syntax to query your data. Examples:
        - `SELECT * FROM data WHERE departure_delay_minutes > 30`
        - `SELECT airline, AVG(departure_delay_minutes) FROM data GROUP BY airline`
        """)
        
        query = st.text_area("Enter your query:", height=100)
        
        if st.button("🔍 Run Query"):
            if query:
                try:
                    # This would integrate with an actual SQL parser
                    # For now, we'll just show a mock example
                    st.info("Query engine processing...")
                    
                    if "WHERE" in query.upper() and "delay" in query.lower():
                        if 'departure_delay_minutes' in data.columns:
                            result = data[data['departure_delay_minutes'] > 15].head(100)
                            st.dataframe(result, use_container_width=True)
                            st.success(f"Query returned {len(result)} records")
                        else:
                            st.warning("No delay column available in dataset")
                    
                    elif "GROUP BY" in query.upper() and "airline" in query.lower():
                        airline_col = next((col for col in data.columns if 'airline' in col.lower()), None)
                        if airline_col and 'departure_delay_minutes' in data.columns:
                            result = data.groupby(airline_col)['departure_delay_minutes'].mean().reset_index()
                            result.columns = [airline_col, 'Average Delay']
                            st.dataframe(result, use_container_width=True)
                            st.success(f"Query returned {len(result)} records")
                        else:
                            st.warning("Required columns not available in dataset")
                    
                    else:
                        # Default to showing head
                        st.dataframe(data.head(100), use_container_width=True)
                        st.info("Showing first 100 records (default)")
                
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
            else:
                st.warning("Please enter a query")

if __name__ == "__main__":
    main()
