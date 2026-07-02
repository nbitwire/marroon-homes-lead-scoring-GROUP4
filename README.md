# marroon-homes-lead-scoring-GROUP4
Predictive Lead Scoring &amp; Sales Conversion Optimization for Marroon Homes

# Marroon Homes — Predictive Lead Scoring & Sales Conversion Optimization

## Project Overview
This project builds a machine learning system to predict the likelihood that a property lead will convert into a buyer for Marroon Homes, a real estate developer based in Ontario, Canada. The goal is to help the sales team prioritize high-intent leads instead of contacting leads in the order they arrive.

## Business Problem
Marroon Homes generates thousands of monthly leads through digital channels but only converts a small fraction into buyers. Without a structured prioritization system, sales effort is wasted on low-intent leads while high-intent leads risk going cold due to delayed follow-up.

## Data
Four datasets were merged on `lead_id`:
- `leads.csv` — demographics and acquisition source (5,000 records)
- `engagement.csv` — digital behavior (site visits, time on site, downloads)
- `interactions.csv` — sales activity (calls, emails, response time)
- `properties.csv` — property catalog data

## Methodology
1. Data ingestion and validation
2. Merging datasets into one unified lead view
3. Feature engineering and encoding of categorical variables
4. Train/test split (80/20)
5. Baseline Decision Tree model
6. Random Forest ensemble model
7. Class imbalance correction using SMOTE (conversion rate was 14.54%)
8. Threshold tuning to improve recall
9. Feature importance analysis
10. Lead segmentation (High / Medium / Low intent)
11. Streamlit app for real-time scoring

## Key Results
- Final model accuracy: 82.9%
- Top conversion drivers identified: time on site, sales response time, age, property interest type, income
- Leads segmented into High, Medium, and Low intent groups for prioritized outreach
- ROC-AUC: 50.57% — indicates the model performs close to random on ranking quality, a known limitation flagged for future improvement (e.g. additional features, more training data, hyperparameter tuning)

## Tech Stack
- Python, Pandas — data processing
- Scikit-learn — Decision Tree & Random Forest models
- imbalanced-learn (SMOTE) — class balancing
- Matplotlib — visualization
- Streamlit — deployment
- Git/GitHub — version control

## Project Structure
marroon_homes_lead_scoring/

data— CSV datasets and ranked leads output

notebooks — main analysis notebook

models- saved trained model and feature columns

app — Streamlit application

README.md

requirements.txt

## How to Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd app
streamlit run app.py
```

## Limitations & Next Steps
- ROC-AUC suggests the model needs further feature engineering or more historical data to better distinguish converters from non-converters
- Future work could include hyperparameter tuning, testing additional algorithms (e.g. XGBoost), and incorporating more granular behavioral data

## Author
Nicole — Data Science Intern, Group 4
