import streamlit as st
import pandas as pd
from backend.ml_models import load_model, FoodNutrientEstimator
from backend.data_preprocessing import clean_meal_text
import joblib # For loading the scaler
import os

# --- Configuration ---
MODEL_DIR = "backend/models"
DIABETES_MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_risk_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "diabetes_scaler.pkl") # Assuming you save the scaler too
FOOD_NUTRIENT_DATA_PATH = "data/food_nutrients.csv"

# --- Load Models and Data ---
@st.cache_resource # Cache the model loading for efficiency
def get_diabetes_assets():
    try:
        model = load_model(DIABETES_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error("Diabetes model or scaler not found. Please train and save them first.")
        return None, None

@st.cache_resource
def get_food_estimator_assets():
    try:
        food_df = pd.read_csv(FOOD_NUTRIENT_DATA_PATH)
        # Assuming further processing for food_df if needed before passing to estimator
        estimator = FoodNutrientEstimator(food_df)
        return estimator
    except FileNotFoundError:
        st.error("Food nutrient data not found. Please ensure 'data/food_nutrients.csv' exists.")
        return None

diabetes_model, diabetes_scaler = get_diabetes_assets()
food_estimator = get_food_estimator_assets()

# --- Streamlit UI ---
st.set_page_config(page_title="Diabetes Risk & Lifestyle Manager", layout="centered")

st.title("Diabetes Risk & Lifestyle Management App")
st.markdown("""
    This application helps you understand your diabetes risk based on clinical data
    and provides insights into your meal's nutritional content.
""")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Diabetes Risk Predictor", "Meal Tracker & Nutrition"])

if app_mode == "Diabetes Risk Predictor":
    st.header("Diabetes Risk Prediction")
    st.write("Please enter your clinical data:")

    if diabetes_model and diabetes_scaler:
        # Input fields for clinical data
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
        age = st.number_input("Age (Years)", min_value=0, max_value=120, value=30)

        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                     insulin, bmi, dpf, age]],
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        if st.button("Predict Diabetes Risk"):
            # Scale the input data
            scaled_input = diabetes_scaler.transform(input_data)
            prediction = diabetes_model.predict(scaled_input)
            prediction_proba = diabetes_model.predict_proba(scaled_input)

            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.error(f"**High Risk of Diabetes!** (Probability: {prediction_proba[0][1]*100:.2f}%)")
                st.write("It is highly recommended to consult a healthcare professional.")
                st.write("---")
                st.subheader("Personalized Lifestyle Suggestions:")
                st.info("""
                    *   **Diet:** Focus on whole grains, lean proteins, and plenty of vegetables. Limit processed foods, sugary drinks, and unhealthy fats.
                    *   **Exercise:** Aim for at least 150 minutes of moderate-intensity aerobic activity per week, plus muscle-strengthening activities on 2 or more days.
                    *   **Weight Management:** Work towards a healthy weight. Even a modest weight loss can significantly reduce risk.
                    *   **Regular Check-ups:** Schedule regular appointments with your doctor for monitoring.
                    *   **Stress Management:** Practice relaxation techniques like yoga or meditation.
                """)
            else:
                st.success(f"**Low Risk of Diabetes.** (Probability: {prediction_proba[0][0]*100:.2f}%)")
                st.write("Maintain your healthy lifestyle to keep the risk low.")
                st.write("---")
                st.subheader("General Healthy Lifestyle Tips:")
                st.info("""
                    *   **Balanced Diet:** Continue eating a variety of fruits, vegetables, and whole grains.
                    *   **Stay Active:** Regular physical activity is key for overall health.
                    *   **Hydration:** Drink plenty of water throughout the day.
                    *   **Adequate Sleep:** Aim for 7-9 hours of quality sleep per night.
                    *   **Monitor Health:** Keep track of your general health indicators.
                """)
    else:
        st.warning("Cannot perform prediction. Please ensure models are trained and available.")

elif app_mode == "Meal Tracker & Nutrition":
    st.header("Meal Tracker & Calorie/Sugar Estimator")
    st.write("Enter your meal details below. The system will estimate its calorie and sugar content.")

    if food_estimator:
        meal_input = st.text_area("What did you eat for your meal?", height=150,
                                  placeholder="e.g., 'A bowl of oatmeal with berries and a spoon of honey'")

        if st.button("Analyze Meal"):
            if meal_input:
                cleaned_meal = clean_meal_text(meal_input)
                # In a BERT setup, you'd pass cleaned_meal to your BERT model
                # For this conceptual code, we use the simple estimator
                nutrients = food_estimator.estimate_nutrients(cleaned_meal)

                st.subheader("Nutritional Estimation:")
                st.write(f"**Estimated Calories:** {nutrients['calories']} kcal")
                st.write(f"**Estimated Sugar:** {nutrients['sugar_g']} grams")

                if nutrients['matched_foods']:
                    st.info(f"Identified food items: {', '.join(nutrients['matched_foods'])}")
                else:
                    st.warning("Could not identify specific food items from your input using simple matching.")

                st.subheader("Dietary Feedback:")
                if nutrients['sugar_g'] > 30: # Example threshold
                    st.warning("High sugar content detected. Consider reducing sugary items in your next meal.")
                elif nutrients['calories'] > 800: # Example threshold
                    st.warning("This seems like a high-calorie meal. Monitor portion sizes.")
                else:
                    st.info("This meal appears to have a moderate nutritional profile.")
                
                st.write("---")
                st.subheader("General Meal Planning Suggestions:")
                st.write("""
                    *   **Portion Control:** Be mindful of serving sizes to manage calorie intake.
                    *   **Balanced Plates:** Aim for half your plate to be vegetables, a quarter lean protein, and a quarter whole grains.
                    *   **Limit Processed Foods:** These are often high in unhealthy fats, sugar, and sodium.
                    *   **Hydrate:** Drink water instead of sugary beverages.
                    *   **Mindful Eating:** Pay attention to your body's hunger and fullness cues.
                """)
            else:
                st.warning("Please enter some meal details to analyze.")
    else:
        st.warning("Cannot analyze meals. Food nutrient estimator not available.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed as a conceptual AI Healthcare Application.")