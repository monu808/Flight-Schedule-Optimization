
# ✈️ Flight Schedule Optimization System

An AI-powered airline operations management dashboard designed to optimize flight schedules at busy airports like Mumbai (BOM) and Delhi (DEL). This system helps operations teams identify optimal time slots, predict delays, and minimize cascading impacts through intelligent scheduling.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.25+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Problem Statement

Due to capacity limitations and heavy passenger load, flight operations at busy airports are becoming a scheduling nightmare. Controllers and operators need to find efficiency in scheduling within system constraints and find means to de-congest flight traffic.

## 🚀 Features

- **Intelligent Data Transformer**: Auto-normalizes uploaded Excel/CSV files into standardized format
- **AI-Powered Analytics**: Real-time insights using Google Gemini AI integration
- **Delay Prediction**: Machine learning models for predicting flight delays
- **Schedule Optimization**: Recommends optimal time slots to reduce congestion
- **Cascade Impact Analysis**: Identifies flights with highest cascading delay impact
- **Runway Utilization**: Optimizes runway capacity and usage patterns
- **NLP Query Interface**: Natural language processing for operational queries
- **Interactive Dashboard**: Comprehensive Streamlit-based visualization

## 📋 Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows / macOS / Linux
- **Package Manager**: pip
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)

## 🏗️ Project Structure

```
Flight-Schedule-Optimization/
├── app/
│   ├── main.py                 # Main Streamlit application
│   └── main_updated.py         # Updated version with enhancements
├── src/
│   ├── data_processor.py       # Data processing and transformation
│   ├── optimizer.py           # Schedule optimization algorithms
│   ├── predictor.py           # ML models for delay prediction
│   ├── anomaly_detector.py    # Anomaly detection in flight patterns
│   ├── peak_time_analyzer.py  # Peak time analysis
│   ├── cascade_delay_predictor.py  # Cascade delay analysis
│   ├── nlp_query_processor.py # Natural language query processing
│   └── advanced_optimizer.py  # Advanced optimization algorithms
├── data/
│   ├── flight_schedule_data.csv    # Sample flight data
│   ├── optimized_schedule.csv      # Optimized schedule output
│   └── Flight_Data.xlsx           # Sample Excel data
├── notebooks/
│   └── flight_analysis.ipynb     # Jupyter notebook for analysis
├── docs/
│   ├── flight_radar_integration.md
│   └── openai_setup.md
├── requirements.txt              # Python dependencies
├── .env.example                 # Environment variables template
└── README.md                   # This file
```

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/monu808/Flight-Schedule-Optimization.git
cd Flight-Schedule-Optimization
```

### 2. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional
TIME_SLOT_DURATION=15
MIN_TURNAROUND_TIME=45
RUNWAY_09R_27L_CAPACITY=30
RUNWAY_09L_27R_CAPACITY=30
RUNWAY_14_32_CAPACITY=25
```

### 5. Run the Application

```bash
streamlit run app/main.py
```

The application will open in your browser at [http://localhost:8501](http://localhost:8501)

## 📊 Data Format & Intelligent Transformer

### Expected Data Format

The application expects flight schedule data with the following columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `FlightNumber` | String | Yes | Flight identifier (e.g., AI101, 6E234) |
| `Airline` | String | Yes | Airline code or name |
| `Scheduled_Departure` | DateTime | Yes | Scheduled departure time |
| `Scheduled_Arrival` | DateTime | Yes | Scheduled arrival time |
| `Actual_Departure` | DateTime | No | Actual departure time |
| `Actual_Arrival` | DateTime | No | Actual arrival time |
| `Runway` | String | No | Runway identifier |
| `Origin` | String | Yes | Origin airport code |
| `Destination` | String | Yes | Destination airport code |
| `Delay_Minutes` | Numeric | No | Computed if missing |

### Intelligent Data Transformer

The system includes an intelligent data transformer that:

- **Auto-detects** common column names and maps them to canonical names
- **Parses** multiple datetime formats automatically
- **Computes** delay minutes when actual times are available
- **Fills** missing required columns where possible
- **Reports** unmappable columns for manual review

Upload any Excel/CSV file through the dashboard - the transformer will handle format conversion automatically.

## 🤖 AI Integration

### Google Gemini AI

The application supports Google Generative AI (Gemini) for intelligent insights:

1. **API Key Setup**: Set your API key in the `.env` file:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

2. **Response Format**: The system is configured for concise metric responses by default

3. **Common Issues**:
   - **404 models/gemini-pro**: Update model name in `app/main.py` to `gemini-1.5-flash` or `gemini-1.5-pro`
   - **Rate Limits**: The system gracefully falls back to local analysis mode

### Fallback Mode

When AI services are unavailable, the system provides:

- Built-in recommendation engine
- Local statistical analysis
- Basic optimization algorithms

## 📖 How to Use

### Main Features

1. **Upload Data**: Use the sidebar file upload to import Excel/CSV files
2. **Review Mapping**: Check the intelligent transformation preview
3. **Apply Filters**: Filter by airport, date range, airline, etc.
4. **Get AI Insights**: Click quick-action buttons for optimization, risk, and revenue analysis
5. **Schedule Tuning**: Use the simulator to test alternate schedules
6. **Export Results**: Download optimized schedules and analysis reports

### Dashboard Navigation

- **📊 Dashboard**: Overview of flight operations and key metrics
- **🔧 Optimization & AI**: Schedule optimization and AI-powered insights  
- **🔍 Query Interface**: Natural language queries for operational questions
- **📈 Analytics**: Detailed performance analytics and predictions
- **⚙️ Advanced**: ML models, anomaly detection, and cascade analysis

## 🔧 Troubleshooting

### Common Issues

- **Streamlit won't start**: Ensure virtual environment is activated and dependencies installed
- **Large file hangs**: Use smaller data samples for initial testing
- **AI responses too long**: Adjust prompt settings in `app/main.py`
- **Missing packages**: Run `pip install -r requirements.txt` again
- **Memory issues**: Increase system memory or reduce dataset size

### Performance Tips

- Use data filtering to reduce processing load
- Enable caching for repeated operations
- Close unused browser tabs when running large analyses

## 🧪 Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
pytest src/

# Code formatting
black src/ app/

# Linting
flake8 src/ app/
```

### Project Architecture

- **Data Layer**: `src/data_processor.py` handles data ingestion and transformation
- **ML Layer**: `src/predictor.py`, `src/anomaly_detector.py` for predictive analytics
- **Optimization Layer**: `src/optimizer.py`, `src/advanced_optimizer.py` for scheduling
- **UI Layer**: `app/main.py` Streamlit dashboard
- **AI Layer**: NLP processing and Gemini integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Keep changes focused and well-documented
- Add tests for new functionality
- Follow existing code style and conventions
- Update documentation for new features

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🔗 References & Data Sources

- **FlightRadar24**: [https://www.flightradar24.com](https://www.flightradar24.com)
- **FlightAware**: [https://www.flightaware.com](https://www.flightaware.com)
- **Mumbai Airport (BOM)**: [https://www.flightradar24.com/data/airports/bom](https://www.flightradar24.com/data/airports/bom)
- **Delhi Airport (DEL)**: [https://www.flightradar24.com/data/airports/del](https://www.flightradar24.com/data/airports/del)

## 🏆 Hackathon Submission

This project addresses the Honeywell Hackathon challenge for flight schedule optimization by:

1. ✅ **Analyzing flight data** from busy airports using open-source AI tools
2. ✅ **Finding optimal takeoff/landing times** through scheduled vs actual time analysis
3. ✅ **Identifying busiest time slots** to avoid congestion
4. ✅ **Providing schedule tuning models** with delay impact visualization
5. ✅ **Isolating high-impact flights** that cause cascading delays
6. ✅ **Offering NLP interface** for natural language queries

---

<<<<<<< HEAD
**Built with ❤️ for airline operations teams**
=======
>>>>>>> cb8aa5f0bb2a86eafb49ab021c75ac5fe7cc527a

