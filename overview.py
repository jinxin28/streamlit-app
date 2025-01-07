# overview.py
import streamlit as st

def display_overview():
    st.write("### Crop Prices in Malaysia")
    st.write("Crop prices in Malaysia are influenced by various factors including climate, seasonality, and market dynamics.")
    st.write("The most common crops grown in Malaysia are palm oil, rice, and rubber. Crop prices fluctuate based on local and international demand, production levels, and external economic factors.")
    st.write("These fluctuations impact farmers' income and the overall economy, making it essential for farmers and stakeholders to monitor these prices regularly.")

    # Add images (make sure the paths are correct)
    #st.image("images/crop_prices_trends.jpg", caption="Crop Price Trends")
    #st.image("images/crop_market_overview.jpg", caption="Crop Market Overview")

    st.write("### Step-by-Step Guide to Using the Web Interface")
    st.write("1. Upload your CSV file containing crop data in the sidebar using the 'Proceed to upload CSV' button.")
    st.write("2. Click on the 'Data Cleaning & EDA' button to clean and explore the data.")
    st.write("3. Once the data is cleaned, click on the 'Price Prediction' button to predict crop prices using machine learning models.")
    st.write("4. The predictions and insights will be shown in the 'Price Prediction' section based on the cleaned data.")
