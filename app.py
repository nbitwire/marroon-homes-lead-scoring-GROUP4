import streamlit as st
import pandas as pd
import joblib
import os

# Load model and feature columns
base_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_dir, '..', 'models', 'lead_scoring_model.pkl'))
feature_columns = joblib.load(os.path.join(base_dir, '..', 'models', 'feature_columns.pkl'))

st.title("Marroon Homes — Lead Scoring Tool")
st.write("Enter lead details below to get a real-time conversion score.")

age = st.slider("Age", 18, 80, 35)
income = st.number_input("Income", min_value=0, value=65000, step=5000)
website_visits = st.number_input("Website Visits", min_value=0, value=1)
time_on_site = st.number_input("Time on Site (seconds)", min_value=0.0, value=60.0)
property_views = st.number_input("Property Views", min_value=0, value=2)
brochure_downloaded = st.selectbox("Brochure Downloaded", [0, 1])
whatsapp_clicks = st.number_input("WhatsApp Clicks", min_value=0, value=0)
agent_calls = st.number_input("Agent Calls", min_value=0, value=0)
emails_sent = st.number_input("Emails Sent", min_value=0, value=0)
meetings_booked = st.number_input("Meetings Booked", min_value=0, value=0)
response_time = st.number_input("Response Time (minutes)", min_value=0.0, value=120.0)

location = st.selectbox("Location", ["Toronto", "Mississauga", "Brampton", "Other"])
source = st.selectbox("Source", ["google_ads", "facebook_ads", "instagram_ads", "organic_search", "listing_portal", "referral", "linkedin_ads", "tiktok_ads"])
interest_type = st.selectbox("Interest Type", ["first_time", "family_home", "investment", "luxury", "off_plan"])

if st.button("Score This Lead"):
    input_data = pd.DataFrame([{
        'age': age, 'income': income, 'website_visits': website_visits,
        'time_on_site': time_on_site, 'property_views': property_views,
        'brochure_downloaded': brochure_downloaded, 'whatsapp_clicks': whatsapp_clicks,
        'agent_calls': agent_calls, 'emails_sent': emails_sent,
        'meetings_booked': meetings_booked, 'response_time': response_time
    }])

    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    loc_col = f"location_{location}"
    src_col = f"source_{source}"
    int_col = f"interest_type_{interest_type}"
    for col in [loc_col, src_col, int_col]:
        if col in input_data.columns:
            input_data[col] = 1

    input_data = input_data[feature_columns]
    score = model.predict_proba(input_data)[:, 1][0]

    st.metric("Conversion Probability", f"{round(score * 100, 1)}%")

    if score >= 0.6:
        st.success("High Intent Lead — Contact Immediately")
    elif score >= 0.3:
        st.warning("Medium Intent Lead")
    else:
        st.info("Low Intent Lead")