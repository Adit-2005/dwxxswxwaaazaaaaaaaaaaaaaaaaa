# Hospital Readmission Predictor — Polished Streamlit App

Features:
- Loads your exported .joblib models from `models/` (uses the first compatible one).
- Professional UI with collapsible sections and color-coded risk card.
- Single patient prediction and batch CSV scoring (download results).
- Streamlit theme configured in `.streamlit/config.toml`.

Run locally:
```
pip install -r requirements.txt
streamlit run app.py
```

Deploy on Streamlit Cloud:
- Push project to GitHub and set main file to `app.py`.