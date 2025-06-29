import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import openai

# Load API key from secrets
openai.api_key = st.secrets["api"]["openai_key"]

# --- Load & preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_csv("your_salary_data.csv")
    df.dropna(inplace=True)
    cat_cols = ['experience_level', 'employment_type', 'job_title',
                'employee_residence', 'company_location', 'company_size']
    encoders = {col: LabelEncoder().fit(df[col]) for col in cat_cols}
    for col, le in encoders.items():
        df[col] = le.transform(df[col])
    X = df.drop(['salary', 'salary_currency', 'salary_in_usd'], axis=1)
    y = df['salary_in_usd']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, y, scaler, encoders

df, X_scaled, y, scaler, encoders = load_data()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Train model ---
model = RandomForestRegressor()
model.fit(X_train, y_train)

# --- UI Inputs ---
st.title("💼 Salary Predictor + OpenAI Assistant")
st.markdown("Predict salaries based on job info and get insights from GPT-4!")

col1, col2 = st.columns(2)

with col1:
    work_year = st.slider("Work Year", 2020, 2025, 2025)
    experience_level = st.selectbox("Experience Level", list(encoders['experience_level'].classes_))
    employment_type = st.selectbox("Employment Type", list(encoders['employment_type'].classes_))
    job_title = st.selectbox("Job Title", list(encoders['job_title'].classes_))

with col2:
    employee_residence = st.selectbox("Employee Residence", list(encoders['employee_residence'].classes_))
    remote_ratio = st.slider("Remote Ratio (%)", 0, 100, 100)
    company_location = st.selectbox("Company Location", list(encoders['company_location'].classes_))
    company_size = st.selectbox("Company Size", list(encoders['company_size'].classes_))

if st.button("Predict Salary"):
    # Encode input
    input_data = {
        'work_year': work_year,
        'experience_level': encoders['experience_level'].transform([experience_level])[0],
        'employment_type': encoders['employment_type'].transform([employment_type])[0],
        'job_title': encoders['job_title'].transform([job_title])[0],
        'employee_residence': encoders['employee_residence'].transform([employee_residence])[0],
        'remote_ratio': remote_ratio,
        'company_location': encoders['company_location'].transform([company_location])[0],
        'company_size': encoders['company_size'].transform([company_size])[0]
    }
    X_input = pd.DataFrame([input_data])
    X_input_scaled = scaler.transform(X_input)

    # Predict
    predicted_salary = model.predict(X_input_scaled)[0]
    st.success(f"💰 Predicted Salary: **${predicted_salary:,.2f} USD**")

    # --- OpenAI Explanation ---
    with st.spinner("🔍 Asking GPT-4 for insights..."):
        prompt = f"""
        A {experience_level} {job_title} in {employee_residence} working {employment_type} with {remote_ratio}% remote work,
        at a {company_size}-sized company in {company_location}, in {work_year}, is predicted to earn ${predicted_salary:,.0f} USD annually.
        Explain why this salary might make sense and suggest ways to improve it.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            explanation = response['choices'][0]['message']['content']
            st.markdown("### 🤖 GPT-4 Insight:")
            st.write(explanation)
        except Exception as e:
            st.error(f"OpenAI Error: {e}")
