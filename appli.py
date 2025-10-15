# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Configuration ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# --- Load Model and Features ---
@st.cache_resource
def load_model():
    """Loads the trained Random Forest model."""
    try:
        with open('diabetes_rf_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('model_features.pkl', 'rb') as file:
            features = pickle.load(file)
        return model, features
    except FileNotFoundError:
        st.error("Model files not found. Please run `train_model.py` first.")
        st.stop() # Stop the app if model isn't found

model, model_features = load_model()

# --- Meal Data (Simple Lookup) ---
# In a real app, this would be a more sophisticated ML model or larger database
food_nutrition_db = {
    "apple": {"calories": 95, "sugar": 19},
    "banana": {"calories": 105, "sugar": 14},
    "chicken breast (100g)": {"calories": 165, "sugar": 0},
    "brown rice (1 cup cooked)": {"calories": 215, "sugar": 0},
    "broccoli (1 cup chopped)": {"calories": 55, "sugar": 2},
    "soda (1 can)": {"calories": 150, "sugar": 39},
    "salad with dressing": {"calories": 250, "sugar": 5},
    "pizza (1 slice)": {"calories": 285, "sugar": 4},
    "yogurt (plain, 1 cup)": {"calories": 150, "sugar": 12},
    "oatmeal (1 cup cooked)": {"calories": 150, "sugar": 1},
    "orange juice (1 cup)": {"calories": 112, "sugar": 21},
}

# --- Functions for Prediction and Suggestions ---
def predict_diabetes_risk(input_data):
    """Predicts diabetes risk based on user input."""
    # Ensure input_data matches the features the model was trained on
    input_df = pd.DataFrame([input_data], columns=model_features)

    # Impute 0s for prediction in the same way as training
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if col in input_df.columns and input_df[col].iloc[0] == 0:
            # For simplicity, we'll use a fixed mean or a more robust value if we had a full dataset available here
            # For this demo, let's use a dummy mean. In train_model.py, it uses the actual training mean.
            # In a production app, you'd save the means from training and use them here.
            input_df[col] = input_df[col].replace(0, 120 if col == 'Glucose' else (70 if col == 'BloodPressure' else (30 if col == 'SkinThickness' else (150 if col == 'Insulin' else 32))))

    prediction_proba = model.predict_proba(input_df)[0] # Get probabilities for [No Diabetes, Diabetes]
    prediction = np.argmax(prediction_proba) # 0 for no diabetes, 1 for diabetes

    risk_levels = {
        0: {"level": "LOW", "color": "green", "message": "Your risk of diabetes is currently low. Keep up the good work!"},
        1: {"level": "HIGH", "color": "red", "message": "Your risk of diabetes is high. It's crucial to consult a healthcare professional for further evaluation and guidance."}
    }

    # Custom thresholds for medium risk based on probability, if we only have two classes
    # This is a simplification; a multi-class model would be better for distinct Low/Medium/High
    if prediction == 0 and prediction_proba[1] > 0.3: # If classified as 0, but probability of 1 is significant
        return {"level": "MEDIUM", "color": "orange", "message": "Your risk of diabetes appears moderate. Lifestyle adjustments and regular check-ups are recommended."}
    elif prediction == 0:
        return risk_levels[0]
    else:
        return risk_levels[1]


def get_lifestyle_suggestions(risk_level, total_calories, total_sugar):
    """Generates personalized lifestyle suggestions."""
    suggestions = []

    if risk_level == "HIGH":
        suggestions.append("🛑 **Immediate action required:** Consult your doctor or an endocrinologist as soon as possible for a comprehensive assessment and management plan.")
        suggestions.append("🥗 Focus on a whole-food diet, rich in vegetables, lean proteins, and complex carbohydrates.")
        suggestions.append("💧 Hydrate with water instead of sugary drinks.")
        suggestions.append("🏃‍♀️ Engage in at least 150 minutes of moderate-intensity exercise per week, if cleared by your doctor.")
    elif risk_level == "MEDIUM":
        suggestions.append("🩺 Schedule a check-up with your doctor to discuss your risk factors.")
        suggestions.append("🍎 Incorporate more fruits, vegetables, and whole grains into your diet.")
        suggestions.append("🍰 Limit intake of processed foods, sugary beverages, and refined carbohydrates.")
        suggestions.append("🚶‍♀️ Aim for regular physical activity, even short walks can make a difference.")
        suggestions.append("😴 Ensure you get adequate sleep (7-9 hours per night).")
    else: # LOW risk
        suggestions.append("✅ **Keep it up!** Continue your healthy lifestyle choices.")
        suggestions.append("🍏 Maintain a balanced diet with plenty of fiber.")
        suggestions.append("💪 Stay physically active and manage stress levels.")
        suggestions.append("Regularly monitor your health and glucose levels.")

    if total_sugar > 50: # Example threshold
        suggestions.append("⚠️ Your recent meal logs show high sugar intake. Consider reducing added sugars in your diet.")
    if total_calories > 2500: # Example threshold
        suggestions.append("⚠️ Your recent meal logs show high calorie intake. Pay attention to portion sizes and calorie density.")
    if total_calories < 1200: # Example threshold
        suggestions.append("⚠️ Your recent meal logs show low calorie intake. Ensure you're meeting your body's energy needs.")

    return suggestions

# --- Streamlit UI ---
st.title("🏥 AI-Powered Diabetes Risk Predictor")
st.markdown("---")

# Layout with columns
col1, col2 = st.columns([1.5, 2]) # Adjust column widths as needed

with col1:
    st.header("Your Clinical Data")
    st.markdown("Please provide your clinical information for risk assessment.")

    # Input fields for clinical data
    pregnancies = st.number_input("Number of Pregnancies (0-17)", min_value=0, max_value=17, value=0)
    glucose = st.number_input("Glucose Level (mg/dL) (70-200)", min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input("Blood Pressure (mmHg) (40-120)", min_value=0, max_value=120, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm) (0-100)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin (muU/ml) (0-850)", min_value=0, max_value=850, value=80)
    bmi = st.number_input("BMI (kg/m²) (0-70)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function (0.0-2.5)", min_value=0.0, max_value=2.5, value=0.4, step=0.01)
    age = st.number_input("Age (Years) (20-100)", min_value=20, max_value=100, value=30)

    # Note: Family History is often represented by DiabetesPedigreeFunction in datasets.
    # If you wanted a separate 'Yes/No' input for Family History, you'd need to adapt model_features and training.
    # For this demo, we'll rely on DiabetesPedigreeFunction.

    # User input dictionary, ensuring order matches model_features
    user_input = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }

    if st.button("Predict Diabetes Risk"):
        st.session_state['show_prediction'] = True
        st.session_state['prediction_result'] = predict_diabetes_risk(user_input)

with col2:
    st.header("Meal Log & Analysis")
    st.markdown("Log your meals to estimate calorie and sugar intake.")

    meal_options = list(food_nutrition_db.keys())
    selected_meals = st.multiselect(
        "Select meal items:",
        options=meal_options,
        default=[]
    )

    total_calories = 0
    total_sugar = 0
    if selected_meals:
        for meal in selected_meals:
            total_calories += food_nutrition_db[meal]["calories"]
            total_sugar += food_nutrition_db[meal]["sugar"]

    st.write(f"**Estimated Calories from Selected Meals:** {total_calories} kcal")
    st.write(f"**Estimated Sugar from Selected Meals:** {total_sugar} g")

    # If you had an advanced image recognition model, this is where you'd put the file uploader:
    # st.subheader("Upload a Meal Photo (Advanced - Not Implemented in this demo)")
    # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    # if uploaded_file is not None:
    #     st.image(uploaded_file, caption="Uploaded Meal Image.", use_column_width=True)
    #     # Process image with ML model here...
    #     st.warning("Image recognition for meals is a complex feature and not included in this basic demo.")

    # --- Display Prediction Results (only after button click) ---
    if st.session_state.get('show_prediction', False):
        st.markdown("---")
        st.subheader("Your Risk Assessment")
        result = st.session_state['prediction_result']

        if result["level"] == "LOW":
            st.success(f"**Risk Group: {result['level']}**")
        elif result["level"] == "MEDIUM":
            st.warning(f"**Risk Group: {result['level']}**")
        else: # HIGH
            st.error(f"**Risk Group: {result['level']}**")

        st.info(result["message"])

        st.markdown("---")
        st.subheader("Personalized Lifestyle Suggestions")
        suggestions = get_lifestyle_suggestions(result["level"], total_calories, total_sugar)
        for i, suggestion in enumerate(suggestions):
            st.markdown(f"{i+1}. {suggestion}")

st.markdown("---")
st.caption("Disclaimer: This application is for informational purposes only and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for any health concerns or before making any decisions related to your health or treatment.")