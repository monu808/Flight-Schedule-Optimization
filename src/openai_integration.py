"""
OpenAI Integration for Flight Schedule Optimization
Provides AI-powered insights, predictions, and natural language processing
"""

import os
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install openai>=1.0.0")

class FlightAIAssistant:
    """AI-powered assistant for flight operations using OpenAI."""
    
    def __init__(self, api_key: str = None):
        """Initialize the AI assistant with OpenAI API."""
        self.openai_available = False
        
        if not OPENAI_AVAILABLE:
            print("OpenAI library not available")
            return
            
        try:
            # Set API key from parameter or environment variable
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                print("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
                return
                
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-3.5-turbo"
            self.max_tokens = 1000
            self.temperature = 0.7
            self.openai_available = True
            
        except Exception as e:
            print(f"Warning: OpenAI initialization failed: {e}")
            self.openai_available = False
    
    def analyze_flight_data_with_ai(self, df: pd.DataFrame) -> Dict:
        """Generate AI-powered insights from flight data."""
        if not self.openai_available:
            return {"error": "OpenAI not available"}
        
        try:
            # Prepare data summary for AI analysis
            data_summary = self._prepare_data_summary(df)
            
            prompt = f"""
            As an expert aviation operations analyst, analyze this flight data and provide insights:
            
            Flight Data Summary:
            {data_summary}
            
            Please provide:
            1. Key operational insights
            2. Delay pattern analysis
            3. Efficiency recommendations
            4. Risk factors identified
            5. Actionable next steps
            
            Format your response as a structured analysis with specific recommendations.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            analysis = response.choices[0].message.content
            return {
                "success": True,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}
    
    def generate_optimization_strategy(self, df: pd.DataFrame) -> Dict:
        """Generate AI-powered optimization strategies."""
        if not self.openai_available:
            return {"error": "OpenAI not available"}
        
        try:
            # Calculate key metrics
            metrics = self._calculate_optimization_metrics(df)
            
            prompt = f"""
            As a flight operations optimization expert, create a comprehensive optimization strategy:
            
            Current Performance Metrics:
            {metrics}
            
            Please provide:
            1. Priority optimization areas
            2. Resource allocation recommendations
            3. Schedule adjustment strategies
            4. Technology solutions to implement
            5. Expected performance improvements (with percentages)
            6. Implementation timeline and phases
            
            Focus on practical, data-driven recommendations that can achieve 15-20% delay reduction.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.6
            )
            
            strategy = response.choices[0].message.content
            return {
                "success": True,
                "strategy": strategy,
                "metrics_used": metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Strategy generation failed: {str(e)}"}
    
    def smart_query_processor(self, query: str, df: pd.DataFrame) -> Dict:
        """Process natural language queries about flight data using AI."""
        if not self.openai_available:
            return {"error": "OpenAI not available"}
        
        try:
            # Get data context
            data_context = self._get_data_context(df)
            
            prompt = f"""
            You are a flight data analyst. Answer this query about flight operations data:
            
            Query: "{query}"
            
            Available Data Context:
            {data_context}
            
            Provide a clear, data-driven answer with specific numbers and insights where possible.
            If the query requires calculations, perform them based on the data context provided.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Query processing failed: {str(e)}"}
    
    def predict_future_delays(self, df: pd.DataFrame, hours_ahead: int = 24) -> Dict:
        """Use AI to predict future delay patterns."""
        if not self.openai_available:
            return {"error": "OpenAI not available"}
        
        try:
            # Analyze historical patterns
            patterns = self._analyze_delay_patterns(df)
            
            prompt = f"""
            As a predictive analytics expert for aviation, forecast delay patterns for the next {hours_ahead} hours:
            
            Historical Delay Patterns:
            {patterns}
            
            Please provide:
            1. Predicted peak delay periods
            2. Expected delay severity levels
            3. Factors likely to contribute to delays
            4. Recommended proactive measures
            5. Confidence levels for predictions
            
            Format as a structured forecast with specific time periods and delay estimates.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.4
            )
            
            forecast = response.choices[0].message.content
            return {
                "success": True,
                "forecast_period": f"{hours_ahead} hours",
                "forecast": forecast,
                "patterns_analyzed": patterns,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def generate_operational_alerts(self, df: pd.DataFrame) -> Dict:
        """Generate intelligent operational alerts using AI."""
        if not self.openai_available:
            return {"error": "OpenAI not available"}
        
        try:
            # Identify potential issues
            issues = self._identify_operational_issues(df)
            
            prompt = f"""
            As an aviation operations monitor, generate intelligent alerts based on these issues:
            
            Detected Issues:
            {issues}
            
            For each issue, provide:
            1. Severity level (Critical/High/Medium/Low)
            2. Immediate actions required
            3. Potential impact on operations
            4. Recommended response timeline
            5. Preventive measures for future
            
            Focus on actionable alerts that operations teams can act upon immediately.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700,
                temperature=0.5
            )
            
            alerts = response.choices[0].message.content
            return {
                "success": True,
                "alerts": alerts,
                "issues_detected": len(issues),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Alert generation failed: {str(e)}"}
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> str:
        """Prepare a concise summary of flight data for AI analysis."""
        try:
            total_flights = len(df)
            delayed_flights = len(df[df['Delay_Minutes'] > 0])
            avg_delay = df['Delay_Minutes'].mean()
            max_delay = df['Delay_Minutes'].max()
            
            # Airline distribution
            airline_counts = df['Airline'].value_counts().head(5)
            
            # Route analysis
            route_delays = df.groupby(['Origin', 'Destination'])['Delay_Minutes'].mean().sort_values(ascending=False).head(5)
            
            summary = f"""
            Total Flights: {total_flights}
            Delayed Flights: {delayed_flights} ({delayed_flights/total_flights*100:.1f}%)
            Average Delay: {avg_delay:.1f} minutes
            Maximum Delay: {max_delay:.0f} minutes
            
            Top Airlines by Volume:
            {airline_counts.to_string()}
            
            Most Delayed Routes (avg delay):
            {route_delays.to_string()}
            """
            
            return summary
            
        except Exception as e:
            return f"Data summary error: {str(e)}"
    
    def _calculate_optimization_metrics(self, df: pd.DataFrame) -> str:
        """Calculate key metrics for optimization strategy."""
        try:
            # Performance metrics
            on_time_rate = len(df[df['Delay_Minutes'] <= 15]) / len(df) * 100
            severe_delay_rate = len(df[df['Delay_Minutes'] > 60]) / len(df) * 100
            
            # Efficiency metrics
            avg_turnaround = df['Delay_Minutes'].mean()
            peak_hour_delays = df.groupby(df['Scheduled_Departure'].str[:2])['Delay_Minutes'].mean().max()
            
            # Resource utilization
            runway_utilization = df.groupby('Runway')['Aircraft_Type'].count().describe()
            
            metrics = f"""
            On-Time Performance: {on_time_rate:.1f}%
            Severe Delay Rate: {severe_delay_rate:.1f}%
            Average Turnaround Delay: {avg_turnaround:.1f} minutes
            Peak Hour Max Delay: {peak_hour_delays:.1f} minutes
            
            Runway Utilization Stats:
            Mean flights per runway: {runway_utilization['mean']:.1f}
            Max flights per runway: {runway_utilization['max']:.0f}
            """
            
            return metrics
            
        except Exception as e:
            return f"Metrics calculation error: {str(e)}"
    
    def _get_data_context(self, df: pd.DataFrame) -> str:
        """Get relevant data context for query processing."""
        try:
            context = f"""
            Dataset contains {len(df)} flights
            Columns: {', '.join(df.columns[:10])}
            Date range: {df['Flight_Date'].min()} to {df['Flight_Date'].max()}
            Airlines: {', '.join(df['Airline'].unique()[:5])}
            Average delay: {df['Delay_Minutes'].mean():.1f} minutes
            """
            return context
        except Exception as e:
            return f"Context error: {str(e)}"
    
    def _analyze_delay_patterns(self, df: pd.DataFrame) -> str:
        """Analyze historical delay patterns."""
        try:
            # Hourly patterns
            hourly_delays = df.groupby(df['Scheduled_Departure'].str[:2])['Delay_Minutes'].mean()
            
            # Daily patterns
            daily_delays = df.groupby('Flight_Date')['Delay_Minutes'].mean()
            
            # Airline patterns
            airline_delays = df.groupby('Airline')['Delay_Minutes'].mean().sort_values(ascending=False)
            
            patterns = f"""
            Peak delay hours: {hourly_delays.idxmax()} ({hourly_delays.max():.1f} min avg)
            Best performance hours: {hourly_delays.idxmin()} ({hourly_delays.min():.1f} min avg)
            
            Worst performing airline: {airline_delays.index[0]} ({airline_delays.iloc[0]:.1f} min avg)
            Best performing airline: {airline_delays.index[-1]} ({airline_delays.iloc[-1]:.1f} min avg)
            
            Recent trend: {daily_delays.tail(3).mean():.1f} min avg (last 3 days)
            """
            
            return patterns
            
        except Exception as e:
            return f"Pattern analysis error: {str(e)}"
    
    def _identify_operational_issues(self, df: pd.DataFrame) -> List[str]:
        """Identify potential operational issues."""
        issues = []
        
        try:
            # High delay rate
            delay_rate = len(df[df['Delay_Minutes'] > 15]) / len(df)
            if delay_rate > 0.3:
                issues.append(f"High delay rate: {delay_rate*100:.1f}% of flights delayed >15 min")
            
            # Runway congestion
            runway_counts = df['Runway'].value_counts()
            if runway_counts.max() > runway_counts.mean() * 2:
                issues.append(f"Runway imbalance: {runway_counts.idxmax()} handling {runway_counts.max()} flights")
            
            # Severe delays
            severe_delays = len(df[df['Delay_Minutes'] > 120])
            if severe_delays > 0:
                issues.append(f"Severe delays detected: {severe_delays} flights delayed >2 hours")
            
            return issues
            
        except Exception as e:
            return [f"Issue detection error: {str(e)}"]
