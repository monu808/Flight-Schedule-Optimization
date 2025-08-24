
# Flight Schedule Optimization — Airline Operations Management

Lightweight AI-powered toolkit and Streamlit dashboard to analyze, predict, and optimize flight schedules for busy airports (example: BOM, DEL). Designed for operations teams to find optimal time slots, identify cascading-delay flights, and run what-if schedule adjustments.

## Quick checklist
- Project overview and purpose — Done
- Local setup and run instructions — Done
- Data format & transformer details — Done
- AI / Gemini key and response notes — Done
- Troubleshooting & contribution notes — Done

## What this project contains
- `app/` — Streamlit dashboard and UI (`app/main.py`).
- `data/` — sample CSVs and generated outputs.
- `src/` — core modules: data processing, optimization, ML, NLP helpers.
- `requirements.txt` — Python dependencies.
- `README.md` — (this file).

## Key features
- Intelligent data transformer: auto-normalizes uploaded Excel/CSV into the project's canonical schema.
- Delay prediction models and schedule tuning simulator.
- Cascade-delay impact detection (network analysis).
- Runway utilization and peak-slot analysis.
- NLP Query interface (AI-powered insights) with concise metric output option.

## Requirements
- Python 3.10+ recommended
- Windows / macOS / Linux
- `pip` available

## Quick start (Windows PowerShell)
1. Create and activate a virtual environment, install deps, and run the app:

```powershell
cd 'C:\Users\ASUS\Desktop\Honeywell_Hackathon\Flight-Schedule-Optimization'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/main.py
```

2. Open the URL printed by Streamlit (usually http://localhost:8501).

## Data: expected format & transformer
The application expects a canonical flight schedule table with (at minimum) these columns:

- `FlightNumber` — flight identifier (string)
- `Airline` — airline code or name
- `Scheduled_Departure` / `Scheduled_Arrival` — ISO-like datetimes or HH:MM
- `Actual_Departure` / `Actual_Arrival` — ISO-like datetimes or HH:MM (optional)
- `Runway` — runway identifier (optional)
- `Origin` / `Destination` — airport ICAO/IATA codes
- `Delay_Minutes` — numeric (optional; computed if missing)

Uploaded files often vary. Use the dashboard file-upload control to upload any Excel/CSV. The built-in intelligent transformer will:

- Auto-detect common column names and map them to the canonical names.
- Parse multiple datetime formats into pandas Timestamps.
- Compute `Delay_Minutes` when actual times are present.
- Fill missing required columns where possible and report unmappable columns.

If an uploaded file cannot be auto-mapped, the transformer will show a short report and an editable mapping preview in the UI.

Developer note: the transformer lives in `src/data_processor.py` (search for `intelligent_transform`) and is invoked by `app/main.py` during upload.

## AI integration (Gemini)
- The app supports Google Generative AI (Gemini). Set your API key in a `.env` file or environment variable named `GEMINI_API_KEY`.
- Example `.env` entry:

```
GEMINI_API_KEY=your_api_key_here
```

- The app requests concise metric-only responses by default (see `app/main.py` prompt section labeled `RESPONSE FORMAT REQUIRED`). If you need longer explanations, toggle verbosity in the UI or edit that prompt.

Common Gemini issues:
- "404 models/gemini-pro" — update the model name in `app/main.py` to a supported model (e.g. `gemini-1.5-flash` / `gemini-1.5-pro` depending on your access).

Fallback: if Gemini isn't available, the app gracefully falls back to a local analysis mode and the built-in recommendation engine.

## How to use the main features
1. Upload raw schedule (Excel/CSV) via the sidebar upload control.
2. Review the transformation preview and confirm mapping.
3. Use filters (airport, date range, airline) to narrow data.
4. Click the AI quick-buttons (Optimization / Risk / Revenue) for concise metrics.
5. Open the Schedule Tuner to simulate alternate scheduled times and see delay impact.

## Troubleshooting
- If Streamlit fails to start: ensure `.venv` activated and `requirements.txt` installed.
- If large uploads hang: check file size, use a sample subset to validate transform, increase memory if needed.
- If AI responses are long: search `RESPONSE FORMAT REQUIRED` in `app/main.py` and adjust the prompt or toggle verbosity in the UI.
- Missing packages: run `pip install -r requirements.txt` again.

## Development notes
- Linting & type-checking: run your preferred tools (e.g. `flake8`, `mypy`) against `src/`.
- Tests: add unit tests for `src/data_processor.py` transformer and core ML modules.

## Contribution
- Fork the repo, create a branch, and submit a PR. Keep changes scoped and provide tests for new logic.

## License
- See `LICENSE` in the repo root.

## Contact / References
- FlightRadar24: https://www.flightradar24.com
- FlightAware: https://www.flightaware.com

---


