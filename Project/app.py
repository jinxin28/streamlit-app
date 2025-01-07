import streamlit as st
import pandas as pd
from data_cleaning import data_cleaning_and_eda
from data_modelling import price_prediction
import overview  # Import the overview.py module


# Streamlit UI
st.title("Crop Data Analysis")

# Initial page setting - show the overview first
if "page" not in st.session_state:
    st.session_state.page = "Overview"  # Default to Overview page when app opens

# Display the Overview page first
if st.session_state.page == "Overview":
    overview.display_overview()

# Sidebar button for navigation after Overview
if st.sidebar.button("Proceed to Upload CSV"):
    st.session_state.page = "Upload CSV"  # Move to the next page where the user can upload CSV

# Sidebar for file upload, now available only after clicking "Proceed to Upload CSV"
if st.session_state.page == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    # Initialize DataFrame in session state
    if "df_cleaned" not in st.session_state:
        st.session_state.df_cleaned = None

    if uploaded_file is not None:
        # Load the dataset from the uploaded CSV
        df = pd.read_csv(uploaded_file)
        # Display preview of the dataset
        st.write("### Data Preview")
        st.write(df.head())

        # Sidebar navigation for Data Cleaning & EDA or Price Prediction
        page = st.sidebar.selectbox("Select a Page", ["Data Cleaning & EDA", "Price Prediction"])

        if page == "Data Cleaning & EDA":
            # Call data cleaning and EDA function
            df_cleaned = data_cleaning_and_eda(df)

            # Check if the returned result is valid
            if isinstance(df_cleaned, pd.DataFrame):
                st.session_state.df_cleaned = df_cleaned  # Save to session state
                st.write("### Cleaned Data")
                st.write(df_cleaned.head())
            else:
                st.error("Data cleaning function did not return a valid DataFrame.")

        elif page == "Price Prediction":
            # Check if cleaned data is available
            if st.session_state.df_cleaned is not None:
                st.write("### Cleaned Data for Price Prediction")
                st.write(st.session_state.df_cleaned.head())  # Show the cleaned DataFrame
                # Call price prediction function with the cleaned data
                price_prediction(st.session_state.df_cleaned)
            else:
                st.warning("Please complete the Data Cleaning & EDA step first.")
