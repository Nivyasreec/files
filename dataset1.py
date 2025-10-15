import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def preprocess_diabetes_data(file_path):
    """
    Loads, preprocesses, and visualizes the diabetes prediction dataset.

    Args:
        file_path (str): The path to the diabetes prediction dataset CSV file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Load the dataset
    diabetes_df = pd.read_csv("C:\project_1\diabetes_predictor_app\diabetes_prediction_dataset.csv")

    # Display basic info
    print("--- First 5 Rows ---")
    print(diabetes_df.head())
    print("\n--- Dataset Info ---")
    print(diabetes_df.info())
    print("\n--- Descriptive Statistics ---")
    print(diabetes_df.describe())

    # Check for 0s in relevant columns before replacement
    print("\n--- Count of 0s before NaN replacement ---")
    print((diabetes_df[['bmi', 'HbA1c_level', 'blood_glucose_level']] == 0).sum())

    # Replace 0s with NaN in the specified columns
    for col in ['bmi', 'HbA1c_level', 'blood_glucose_level']:
        diabetes_df[col] = diabetes_df[col].replace(0, np.nan)

    # Check how many missing values are now present
    print("\n--- Count of Missing Values (NaN) after replacement ---")
    print(diabetes_df.isnull().sum())

    print("\n--- Distribution of Diabetes Status ---")
    print(diabetes_df['diabetes'].value_counts())
    print(diabetes_df['diabetes'].value_counts(normalize=True) * 100)

    # Plot Distribution of Diabetes Status
    plt.figure(figsize=(6, 4))
    sns.countplot(x='diabetes', data=diabetes_df)
    plt.title('Distribution of Diabetes (Target Variable)')
    plt.savefig(os.path.join(output_dir, 'diabetes_distribution.png')) # Save plot
    plt.show()
    
    print("\n--- Distribution of Gender ---")
    print(diabetes_df['gender'].value_counts())
    plt.figure(figsize=(7, 5))
    sns.countplot(x='gender', hue='diabetes', data=diabetes_df)
    plt.title('Diabetes Status by Gender')
    plt.savefig(os.path.join(output_dir, 'diabetes_by_gender.png')) # Save plot
    plt.show()

    print("\n--- Distribution of Smoking History ---")
    print(diabetes_df['smoking_history'].value_counts())
    plt.figure(figsize=(8, 5))
    sns.countplot(x='smoking_history', hue='diabetes', data=diabetes_df)
    plt.title('Diabetes Status by Smoking History')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'diabetes_by_smoking_history.png')) # Save plot
    plt.show()

    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

    print("\n--- Histograms for Numerical Features ---")
    diabetes_df[numerical_cols].hist(bins=30, figsize=(15, 10))
    plt.suptitle('Histograms of Numerical Features')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'numerical_features_histograms.png')) # Save plot
    plt.show()

    print("\n--- Box Plots for Numerical Features by Diabetes Status ---")
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x='diabetes', y=col, data=diabetes_df)
        plt.title(f'{col} by Diabetes Status')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'numerical_features_boxplots.png')) # Save plot
    plt.show()

    print("\n--- Correlation Matrix ---")
    plt.figure(figsize=(10, 8))
    sns.heatmap(diabetes_df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Features')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png')) # Save plot
    plt.show()

    print("✅ Diabetes data preprocessing complete!")
    return diabetes_df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust the path to 'diabetes_prediction_dataset.csv' relative to the script
    data_path = os.path.join(current_dir, '..', 'data', 'diabetes_prediction_dataset.csv')
    
    # Ensure the 'figures' directory exists to save plots
    output_dir = os.path.join(current_dir, '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    cleaned_df = preprocess_diabetes_data(data_path)
    # You might want to save this cleaned_df for the next training script
    cleaned_df.to_csv(os.path.join(current_dir, '..', 'data', 'diabetes_cleaned.csv'), index=False)
    print(f"Cleaned diabetes data saved to {os.path.join(current_dir, '..', 'data', 'diabetes_cleaned.csv')}")

import streamlit as st
import joblib
import pandas as pd # Needed if your models expect a specific data structure

# Load the trained models and vectorizer
try:
    calorie_model = joblib.load('calorie_model.pkl')
    sugar_model = joblib.load('sugar_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'calorie_model.pkl', 'sugar_model.pkl', and 'tfidf_vectorizer.pkl' are in the same directory.")
    st.stop() # Stop the app if models aren't found

st.title("Indian Food Nutrition Predictor")

dish_name = st.text_input("Enter a dish name:", "Butter Chicken")

if dish_name:
    # Transform the input dish name using the loaded TF-IDF vectorizer
    # Ensure the input is in a list as expected by transform
    dish_features = tfidf_vectorizer.transform([dish_name]).toarray()

    # Predict calories and sugar
    predicted_calories = calorie_model.predict(dish_features)[0]
    predicted_sugar = sugar_model.predict(dish_features)[0]

    st.subheader(f"Nutrition Prediction for '{dish_name}':")
    st.metric(label="Predicted Calories (kcal)", value=f"{predicted_calories:.2f}")
    st.metric(label="Predicted Free Sugar (g)", value=f"{predicted_sugar:.2f}")

    st.markdown("---")
    st.info("Note: Predictions are based on the training data and may vary for new or complex dish names.")