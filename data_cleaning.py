import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from datetime import datetime
import seaborn as sns

# Define the function to detect outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

# Function to create a Date column
def create_date(row):
    try:
        return datetime.strptime(f"{row['Year']}-{row['Months'][:3]}", '%Y-%b')
    except ValueError:
        return None

# File upload handling within Streamlit
def data_cleaning_and_eda(df):
    # Display Data Preview
    #st.write("### Data Preview")
    #st.write(df.head())

    # Group data by Item (Papayas, Pineapples, Watermelons)
    crop_groups = df.groupby('Item')

    cleaned_crops = []

    # Visualize outliers and remove them for each crop
    for crop_name, crop_data in crop_groups:
        # Visualize boxplot for each crop
        #st.write(f"### Boxplot for {crop_name} (Outliers Detection)")
        #fig, ax = plt.subplots(figsize=(10, 6))
        #ax.boxplot(crop_data['Value'].dropna(), vert=False)
        #ax.set_title(f'Boxplot for {crop_name} (Outliers Detection)')
        #ax.set_xlabel('Value (LCU/tonne)')
        #st.pyplot(fig)

        # Detect outliers using IQR method and add an 'Outlier' column to the crop_data
        crop_data['Outlier'] = detect_outliers_iqr(crop_data, 'Value')

        # Display rows with outliers for each crop
        #outliers_df = crop_data[crop_data['Outlier'] == True]
        #st.write(f"Outliers detected for {crop_name}:")
        #st.write(outliers_df)

        # Remove outliers from the dataset
        crop_data_cleaned = crop_data[~crop_data['Outlier']]

       
        # Append cleaned data to the list
        cleaned_crops.append(crop_data_cleaned)


        # Visualize the cleaned data for each crop
        #st.write(f"### Boxplot for {crop_name} (After Removing Outliers)")
        #fig, ax = plt.subplots(figsize=(10, 6))
        #ax.boxplot(crop_data_cleaned['Value'].dropna(), vert=False)
        #ax.set_title(f'Boxplot for {crop_name} (After Removing Outliers)')
        #ax.set_xlabel('Value (LCU/tonne)')
        #st.pyplot(fig)

    # Combine the cleaned crop datasets back into one DataFrame
    df_cleaned = pd.concat(cleaned_crops).sort_values(
        by=["Item", "Year", "Months"]
    ).reset_index(drop=True)

    #st.write("### Cleaned Data (After Removing Outliers):")
    #st.write(df_cleaned.head())

    # Drop 'Outlier' column
    df = df_cleaned.drop(columns=['Outlier'])

    # Define the list of months
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    # Create a DataFrame with all years, months, and items
    unique_years = df["Year"].unique()
    unique_items = df["Item"].unique()
    full_year_month_item = pd.DataFrame(
        [(year, month, item) for year in unique_years for month in months for item in unique_items],
        columns=["Year", "Months", "Item"]
    )

    # Merge with the original data
    merged_df = pd.merge(
        full_year_month_item,
        df,
        how="left",
        on=["Year", "Months", "Item"]
    )

    # Add Missing column and count missing months by year
    merged_df["Missing"] = merged_df["Value"].isnull()
    missing_summary = merged_df[merged_df["Missing"]].groupby("Year")["Months"].count()
    #st.write(missing_summary)

    # Fill missing metadata for each item
    default_values = {
        "Domain": "Producer Prices",
        "Area": "Malaysia",
        "Element": "Producer Price (LCU/tonne)"
    }
    for column, default in default_values.items():
        merged_df[column].fillna(default, inplace=True)

    # Interpolate missing 'Value' for each item (including the last year)
    merged_df["Value"] = (
        merged_df.groupby("Item")["Value"]
        .transform(lambda x: x.interpolate())
    )

    # Create a Date column and handle missing values
    merged_df['Date'] = merged_df.apply(create_date, axis=1)
    merged_df = merged_df.dropna(subset=['Date'])

    # Output the cleaned DataFrame
    #st.write("### Cleaned Data with Date and Interpolation:")
    #st.write(merged_df)

    # Check for missing values
    #st.write("### Missing Values:")
    #st.write(merged_df.isnull().sum())

    # Remove duplicate and null rows
    merged_df = merged_df.drop_duplicates()
    merged_df = merged_df.dropna()

    # Sort for clarity
    months_order = {month: i for i, month in enumerate(months, start=1)}
    merged_df["Month_Order"] = merged_df["Months"].map(months_order)
    merged_df.sort_values(by=["Item", "Year", "Month_Order"], inplace=True)

    # Drop the temporary 'Month_Order' column after sorting
    merged_df.drop(columns=["Month_Order"], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    # Drop the mising values column
    merged_df.drop(columns=['Missing'], inplace=True)

    # Validate data types
    #st.write("### Data Types:")
    #st.write(merged_df.dtypes)

     # Create a full daily date range for each crop, adding 30 days beyond the max date
    full_date_range = pd.DataFrame({
        'Date': pd.date_range(start=merged_df['Date'].min(), 
                              end=merged_df['Date'].max() + pd.Timedelta(days=30))
    })

    expanded_data = []
    for crop in merged_df['Item'].unique():
        crop_data = merged_df[merged_df['Item'] == crop]
        crop_full_range = pd.merge(full_date_range, crop_data, how='left', on='Date')
        crop_full_range['Item'] = crop  # Reassign crop name in case of missing rows
        expanded_data.append(crop_full_range)

    # Combine all expanded datasets
    df_expanded = pd.concat(expanded_data, ignore_index=True)

    # Interpolate missing values for 'Value' column by crop
    df_expanded['Value'] = df_expanded.groupby('Item')['Value'].transform(lambda x: x.interpolate())

    # Round the 'Value' column to 2 decimal places
    df_expanded['Value'] = df_expanded['Value'].apply(lambda x: f"{x:.2f}")

    # Forward fill for 'Year', 'Months', 'Domain', 'Area', and 'Element'
    df_expanded['Year'] = df_expanded['Year'].fillna(method='ffill')
    df_expanded['Months'] = df_expanded['Months'].fillna(method='ffill')
    df_expanded['Domain'] = df_expanded['Domain'].fillna(method='ffill')
    df_expanded['Area'] = df_expanded['Area'].fillna(method='ffill')
    df_expanded['Element'] = df_expanded['Element'].fillna(method='ffill')

    # Final cleaning steps
    df_expanded.drop_duplicates(inplace=True)
    df_expanded.dropna(subset=['Date', 'Value'], inplace=True)

    # Sorting by 'Item' and 'Date' for clarity
    df_expanded.sort_values(by=['Item', 'Date'], inplace=True)

    # Remove duplicate and null rows
    df_expanded.reset_index(drop=True, inplace=True)

    merged_df = df_expanded
    # Ensure 'Value' is numeric
    merged_df['Value'] = pd.to_numeric(merged_df['Value'], errors='coerce')

    # Pivot data to analyze trends across crops
    pivot_data = merged_df.pivot_table(
        index='Date',
        columns='Item',
        values='Value',
        aggfunc='mean'
    )

    

    # Dynamically extract crop options from the "Item" column
    crop_options = ['All Crops'] + merged_df['Item'].unique().tolist()

    # Crop selection slider
    crop_selection = st.selectbox(
        "Select a crop (or all crops)",
        options=crop_options
    )

    # Visualization type selection
    visualization_type = st.selectbox(
        "Select a type of visualization",
        options=["Price Trends Across All Crops", "Yearly Price Trends", "Monthly Price Trends", 
                 "Price Distribution by Year", "Correlation Between Crops", "Histogram of Prices"]
    )

    # Filter data based on selected crop(s)
    if crop_selection != "All Crops":
        df_filtered = merged_df[merged_df['Item'] == crop_selection]
    else:
        df_filtered = merged_df

    # Generate visualizations based on selected type
    if visualization_type == "Price Trends Across All Crops":
        st.write("### Price Trends Across All Crops")
        pivot_data = df_filtered.pivot_table(
            index='Date',
            columns='Item',
            values='Value',
            aggfunc='mean'
        )
        fig, ax = plt.subplots(figsize=(14, 6))
        for crop in pivot_data.columns:
            ax.plot(pivot_data.index, pivot_data[crop], label=crop)
        ax.set_title("Price Trends Across All Crops")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (LCU)")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

    elif visualization_type == "Yearly Price Trends":
        st.write("### Yearly Price Trends for Each Crop")
        yearly_data = df_filtered.groupby(['Year', 'Item'])['Value'].mean().unstack()
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_data.plot(ax=ax)
        ax.set_title("Yearly Price Trends")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Price (LCU/tonne)")
        st.pyplot(fig)

    elif visualization_type == "Monthly Price Trends":
        st.write("### Monthly Price Trends Across All Years")
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        monthly_data = df_filtered.groupby(['Months', 'Item'])['Value'].mean().unstack()
        monthly_data = monthly_data.loc[months]  # Ensure months are in correct order
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_data.plot(ax=ax)
        ax.set_title("Monthly Price Trends Across All Years")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Price (LCU/tonne)")
        st.pyplot(fig)

    elif visualization_type == "Price Distribution by Year":
        st.write("### Price Distribution for Each Crop (By Year)")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_filtered.boxplot(column='Value', by='Year', ax=ax, vert=False, patch_artist=True)
        ax.set_title("Price Distribution by Year")
        ax.set_xlabel("Price (LCU/tonne)")
        ax.set_ylabel("Year")
        st.pyplot(fig)

    elif visualization_type == "Correlation Between Crops":
        st.write("### Correlation Between Crops")
        crops_data = df_filtered.pivot_table(index='Date', columns='Item', values='Value')
        correlation_matrix = crops_data.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
        ax.set_title("Correlation Between Crop Prices")
        st.pyplot(fig)

    elif visualization_type == "Histogram of Prices":
        st.write("### Histogram of Prices for Each Crop")
        for crop in df_filtered['Item'].unique():
            crop_data = df_filtered[df_filtered['Item'] == crop]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(crop_data['Value'], bins=15, color='skyblue', edgecolor='black')
            ax.set_title(f"Price Distribution for {crop}")
            ax.set_xlabel("Price (LCU/tonne)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)





    # Save the cleaned data to CSV
    cleaned_csv_filename = "cleaned_crop_data.csv"
    merged_df.to_csv(cleaned_csv_filename, index=False)

    # Provide a download link for the cleaned data
    st.write("### Download the Cleaned Data")
    st.download_button(
        label="Download Cleaned Data (CSV)",
        data=open(cleaned_csv_filename, "rb").read(),
        file_name=cleaned_csv_filename,
        mime="text/csv"
    
    )

    return merged_df