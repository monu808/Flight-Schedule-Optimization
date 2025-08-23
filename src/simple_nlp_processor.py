"""
Simple Natural Language Processing Query Interface
Enables basic natural language queries without heavy dependencies.
"""

import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class QueryIntent:
    """Represents a parsed query intent."""
    action: str  # 'show', 'optimize', 'predict', 'analyze'
    entity: str  # 'flights', 'delays', 'runways', 'schedule'
    filters: Dict[str, Any]  # time, airport, airline, etc.
    modifiers: List[str]  # 'most', 'least', 'tomorrow', etc.
    confidence: float

class SimpleNLPQueryProcessor:
    """
    Simple natural language query processor without heavy ML dependencies.
    """
    
    def __init__(self):
        self.intent_patterns = self._load_intent_patterns()
        self.entity_mappings = self._load_entity_mappings()
        self.time_patterns = self._load_time_patterns()
        
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent patterns for query classification."""
        return {
            'show': [
                'show', 'display', 'list', 'get', 'find', 'what', 'which', 'tell me'
            ],
            'optimize': [
                'optimize', 'improve', 'adjust', 'schedule', 'reschedule', 'rearrange'
            ],
            'predict': [
                'predict', 'forecast', 'estimate', 'expect', 'anticipate', 'will'
            ],
            'analyze': [
                'analyze', 'examine', 'study', 'investigate', 'review', 'assess'
            ]
        }
    
    def _load_entity_mappings(self) -> Dict[str, List[str]]:
        """Load entity mappings for query parsing."""
        return {
            'flights': [
                'flight', 'flights', 'aircraft', 'plane', 'airplane'
            ],
            'delays': [
                'delay', 'delays', 'delayed', 'late', 'lateness', 'behind schedule'
            ],
            'runways': [
                'runway', 'runways', 'strip', 'landing strip', 'takeoff strip'
            ],
            'schedule': [
                'schedule', 'timetable', 'time', 'timing', 'slot', 'slots'
            ],
            'airlines': [
                'airline', 'airlines', 'carrier', 'carriers'
            ],
            'airports': [
                'airport', 'airports', 'terminal', 'hub'
            ],
            'passengers': [
                'passenger', 'passengers', 'people', 'travelers'
            ],
            'throughput': [
                'throughput', 'capacity', 'volume', 'traffic', 'flow'
            ]
        }
    
    def _load_time_patterns(self) -> Dict[str, str]:
        """Load time-related patterns."""
        return {
            'tomorrow': 'tomorrow',
            'today': 'today',
            'yesterday': 'yesterday',
            'next week': 'next_week',
            'this week': 'this_week',
            'morning': 'morning',
            'afternoon': 'afternoon',
            'evening': 'evening',
            'night': 'night',
            'peak': 'peak_hours',
            'rush': 'peak_hours',
            'busy': 'peak_hours'
        }
    
    def parse_query(self, query: str) -> QueryIntent:
        """
        Parse natural language query into structured intent.
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryIntent object
        """
        query = query.lower().strip()
        
        # Extract intent (action)
        action = self._extract_intent(query)
        
        # Extract entity
        entity = self._extract_entity(query)
        
        # Extract filters
        filters = self._extract_filters(query)
        
        # Extract modifiers
        modifiers = self._extract_modifiers(query)
        
        # Calculate confidence
        confidence = self._calculate_confidence(query, action, entity)
        
        return QueryIntent(
            action=action,
            entity=entity,
            filters=filters,
            modifiers=modifiers,
            confidence=confidence
        )
    
    def _extract_intent(self, query: str) -> str:
        """Extract the main intent/action from the query."""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return intent
        
        # Default intent based on question patterns
        if any(word in query for word in ['what', 'which', 'who', 'where', 'when']):
            return 'show'
        elif any(word in query for word in ['how to', 'can you', 'should i']):
            return 'optimize'
        else:
            return 'show'
    
    def _extract_entity(self, query: str) -> str:
        """Extract the main entity being queried."""
        entity_scores = {}
        
        for entity, patterns in self.entity_mappings.items():
            score = 0
            for pattern in patterns:
                if pattern in query:
                    score += 1
            if score > 0:
                entity_scores[entity] = score
        
        if entity_scores:
            return max(entity_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'flights'  # Default entity
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filters from the query."""
        filters = {}
        
        # Time filters
        time_filter = self._extract_time_filter(query)
        if time_filter:
            filters['time'] = time_filter
        
        # Airport filters
        airport_codes = re.findall(r'\b[A-Z]{3}\b', query.upper())
        if airport_codes:
            filters['airports'] = airport_codes
        
        # Airline filters
        airlines = ['indigo', 'spicejet', 'air india', 'vistara', 'go air', 'alliance air']
        found_airlines = [airline for airline in airlines if airline in query]
        if found_airlines:
            filters['airlines'] = found_airlines
        
        # Aircraft type filters
        aircraft_types = ['a320', 'a321', 'a330', 'a350', 'b737', 'b777', 'b787', 'atr72', 'q400']
        found_aircraft = [aircraft for aircraft in aircraft_types if aircraft in query]
        if found_aircraft:
            filters['aircraft_types'] = found_aircraft
        
        # Delay threshold filters
        delay_numbers = re.findall(r'(\d+)\s*(?:minute|min|hour|hr)', query)
        if delay_numbers:
            filters['delay_threshold'] = int(delay_numbers[0])
        
        # Quantity filters
        quantity_patterns = {
            'most': r'most|highest|maximum|top|greatest',
            'least': r'least|lowest|minimum|bottom|smallest',
            'all': r'all|every|each'
        }
        
        for quantity, pattern in quantity_patterns.items():
            if re.search(pattern, query):
                filters['quantity'] = quantity
                break
        
        return filters
    
    def _extract_time_filter(self, query: str) -> Optional[str]:
        """Extract time-related filters."""
        for time_phrase, time_code in self.time_patterns.items():
            if time_phrase in query:
                return time_code
        
        # Extract specific times
        time_matches = re.findall(r'(\d{1,2})(?::|\\s)?(am|pm|:\\d{2})', query)
        if time_matches:
            return f"specific_time_{time_matches[0]}"
        
        # Extract date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2}'  # YYYY-MM-DD
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, query):
                return 'specific_date'
        
        return None
    
    def _extract_modifiers(self, query: str) -> List[str]:
        """Extract modifiers that affect the query scope."""
        modifiers = []
        
        modifier_patterns = {
            'urgent': r'urgent|emergency|critical|immediate',
            'frequent': r'frequent|often|regular|repeated',
            'international': r'international|overseas|foreign',
            'domestic': r'domestic|local|national',
            'heavy_traffic': r'heavy traffic|congested|busy|crowded',
            'efficiency': r'efficient|optimal|best|improved'
        }
        
        for modifier, pattern in modifier_patterns.items():
            if re.search(pattern, query):
                modifiers.append(modifier)
        
        return modifiers
    
    def _calculate_confidence(self, query: str, action: str, entity: str) -> float:
        """Calculate confidence score for the parsed query."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence if clear action words are found
        if any(pattern in query for pattern in self.intent_patterns[action]):
            confidence += 0.2
        
        # Boost confidence if clear entity words are found
        if any(pattern in query for pattern in self.entity_mappings[entity]):
            confidence += 0.2
        
        # Boost confidence for structured queries
        if '?' in query:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def execute_query_on_dataframe(self, intent: QueryIntent, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute query intent on a pandas DataFrame.
        
        Args:
            intent: Parsed query intent
            df: Flight data DataFrame
            
        Returns:
            Filtered/processed DataFrame
        """
        result_df = df.copy()
        
        # Apply filters
        if 'time' in intent.filters:
            result_df = self._apply_time_filter(result_df, intent.filters['time'])
        
        if 'airlines' in intent.filters:
            result_df = result_df[result_df['Airline'].str.lower().isin(
                [airline.lower() for airline in intent.filters['airlines']]
            )]
        
        if 'airports' in intent.filters:
            if 'Origin' in result_df.columns and 'Destination' in result_df.columns:
                airport_filter = (
                    result_df['Origin'].isin(intent.filters['airports']) |
                    result_df['Destination'].isin(intent.filters['airports'])
                )
                result_df = result_df[airport_filter]
        
        if 'delay_threshold' in intent.filters:
            if 'Delay_Minutes' in result_df.columns:
                result_df = result_df[result_df['Delay_Minutes'] > intent.filters['delay_threshold']]
        
        # Apply entity-specific processing
        if intent.entity == 'delays':
            if 'Delay_Minutes' in result_df.columns:
                result_df = result_df[result_df['Delay_Minutes'] > 0]
                result_df = result_df.sort_values('Delay_Minutes', ascending=False)
        
        elif intent.entity == 'runways':
            if 'Runway' in result_df.columns:
                result_df = result_df.groupby('Runway').agg({
                    'Flight_ID': 'count',
                    'Delay_Minutes': 'mean' if 'Delay_Minutes' in result_df.columns else lambda x: 0,
                    'Capacity': 'sum' if 'Capacity' in result_df.columns else lambda x: 0
                }).reset_index()
                result_df.columns = ['Runway', 'Flight_Count', 'Avg_Delay', 'Total_Passengers']
        
        elif intent.entity == 'airlines':
            if 'Airline' in result_df.columns:
                result_df = result_df.groupby('Airline').agg({
                    'Flight_ID': 'count',
                    'Delay_Minutes': 'mean' if 'Delay_Minutes' in result_df.columns else lambda x: 0,
                    'Capacity': 'sum' if 'Capacity' in result_df.columns else lambda x: 0
                }).reset_index()
                result_df.columns = ['Airline', 'Flight_Count', 'Avg_Delay', 'Total_Passengers']
        
        # Apply quantity filters
        if 'quantity' in intent.filters:
            if intent.filters['quantity'] == 'most':
                if intent.entity == 'delays' and 'Delay_Minutes' in result_df.columns:
                    result_df = result_df.nlargest(10, 'Delay_Minutes')
                elif intent.entity in ['runways', 'airlines'] and 'Avg_Delay' in result_df.columns:
                    result_df = result_df.nlargest(10, 'Avg_Delay')
            elif intent.filters['quantity'] == 'least':
                if intent.entity == 'delays' and 'Delay_Minutes' in result_df.columns:
                    result_df = result_df.nsmallest(10, 'Delay_Minutes')
                elif intent.entity in ['runways', 'airlines'] and 'Avg_Delay' in result_df.columns:
                    result_df = result_df.nsmallest(10, 'Avg_Delay')
        
        return result_df
    
    def _apply_time_filter(self, df: pd.DataFrame, time_filter: str) -> pd.DataFrame:
        """Apply time-based filters to DataFrame."""
        if 'Scheduled_Time' not in df.columns:
            return df
            
        df['Scheduled_Time'] = pd.to_datetime(df['Scheduled_Time'])
        df['Hour'] = df['Scheduled_Time'].dt.hour
        df['Date'] = df['Scheduled_Time'].dt.date
        
        today = datetime.now().date()
        
        if time_filter == 'today':
            return df[df['Date'] == today]
        elif time_filter == 'tomorrow':
            tomorrow = today + timedelta(days=1)
            return df[df['Date'] == tomorrow]
        elif time_filter == 'morning':
            return df[df['Hour'].between(6, 11)]
        elif time_filter == 'afternoon':
            return df[df['Hour'].between(12, 17)]
        elif time_filter == 'evening':
            return df[df['Hour'].between(18, 23)]
        elif time_filter == 'night':
            return df[(df['Hour'] >= 0) & (df['Hour'] <= 5) | (df['Hour'] >= 24)]
        elif time_filter == 'peak_hours':
            return df[df['Hour'].isin([7, 8, 9, 18, 19, 20])]
        
        return df
    
    def generate_response(self, intent: QueryIntent, result_df: pd.DataFrame) -> str:
        """
        Generate natural language response based on query results.
        
        Args:
            intent: Original query intent
            result_df: Query results DataFrame
            
        Returns:
            Natural language response string
        """
        if result_df.empty:
            return "No results found for your query."
        
        response_parts = []
        
        # Generate response based on action and entity
        if intent.action == 'show':
            if intent.entity == 'delays':
                if 'quantity' in intent.filters and intent.filters['quantity'] == 'most':
                    response_parts.append(f"Here are the {len(result_df)} most delayed flights:")
                    if 'Delay_Minutes' in result_df.columns:
                        for _, row in result_df.head(5).iterrows():
                            response_parts.append(f"• {row['Flight_ID']}: {row['Delay_Minutes']:.1f} minutes delay")
                else:
                    if 'Delay_Minutes' in result_df.columns:
                        avg_delay = result_df['Delay_Minutes'].mean()
                        response_parts.append(f"Found {len(result_df)} delayed flights with an average delay of {avg_delay:.1f} minutes.")
                    else:
                        response_parts.append(f"Found {len(result_df)} flights.")
            
            elif intent.entity == 'runways':
                response_parts.append(f"Runway utilization summary ({len(result_df)} runways):")
                for _, row in result_df.iterrows():
                    if 'Avg_Delay' in row and 'Flight_Count' in row:
                        response_parts.append(f"• {row['Runway']}: {row['Flight_Count']} flights, {row['Avg_Delay']:.1f} min avg delay")
                    else:
                        response_parts.append(f"• {row['Runway']}: {row.get('Flight_Count', 'N/A')} flights")
            
            elif intent.entity == 'airlines':
                response_parts.append(f"Airline performance summary ({len(result_df)} airlines):")
                for _, row in result_df.head(5).iterrows():
                    if 'Avg_Delay' in row and 'Flight_Count' in row:
                        response_parts.append(f"• {row['Airline']}: {row['Flight_Count']} flights, {row['Avg_Delay']:.1f} min avg delay")
                    else:
                        response_parts.append(f"• {row['Airline']}: {row.get('Flight_Count', 'N/A')} flights")
        
        elif intent.action == 'optimize':
            response_parts.append("Based on the data analysis, here are optimization recommendations:")
            
            if intent.entity == 'schedule':
                response_parts.append("• Consider redistributing flights from peak hours (7-9 AM, 6-8 PM)")
                response_parts.append("• Utilize underused time slots during off-peak hours")
                response_parts.append("• Implement dynamic runway allocation based on aircraft type")
        
        elif intent.action == 'predict':
            response_parts.append("Delay prediction analysis:")
            if 'Delay_Minutes' in result_df.columns:
                future_delay_estimate = result_df['Delay_Minutes'].mean() * 1.1  # Simple prediction
                response_parts.append(f"Expected average delay: {future_delay_estimate:.1f} minutes")
        
        # Add confidence note if low
        if intent.confidence < 0.7:
            response_parts.append(f"\n(Note: Query interpretation confidence: {intent.confidence:.0%})")
        
        return "\n".join(response_parts)
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """
        Generate query suggestions based on partial input.
        
        Args:
            partial_query: Partial query string
            
        Returns:
            List of suggested complete queries
        """
        suggestions = []
        
        # Common query templates
        templates = [
            "Show me the most delayed flights tomorrow",
            "Optimize evening schedule for maximum throughput",
            "Which aircraft types cause most delays?",
            "Analyze runway utilization during peak hours",
            "Predict delays for international flights",
            "Show flights delayed more than 30 minutes",
            "Which runway has the highest traffic?",
            "Optimize schedule for Air India flights",
            "Show morning flight delays",
            "Analyze delay patterns by airline"
        ]
        
        # Filter templates based on partial query
        partial_lower = partial_query.lower()
        for template in templates:
            if partial_lower in template.lower() or any(word in template.lower() for word in partial_lower.split()):
                suggestions.append(template)
        
        return suggestions[:5]  # Return top 5 suggestions

# Create aliases for backward compatibility
NLPQueryProcessor = SimpleNLPQueryProcessor
