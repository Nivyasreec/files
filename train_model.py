# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Starting model training...")

# 1. Load the dataset
try:
    df = pd.read_csv("C:\project_1\diabetes_predictor_app\diabetes_data.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: diabetes_data.csv not found. Please create it first.")
    exit()

# 2. Define features (X) and target (y)
# We'll use all columns except 'Outcome' as features for training
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Handle missing values (simple imputation for demonstration)
# Replace 0s in relevant columns with the mean of that column
# In a real scenario, you'd do more robust imputation.
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    X[col] = X[col].replace(0, X[col].mean())

print("Missing values (0s) imputed.")

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

# 5. Initialize and Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Random Forest model trained.")

# 6. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on test set: {accuracy:.2f}")

# 7. Save the trained model
with open('diabetes_rf_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Trained model saved as diabetes_rf_model.pkl")

# Save the feature column names that the model expects during prediction
# This is crucial for ensuring the Streamlit app provides inputs in the correct order
model_features = X.columns.tolist()
with open('model_features.pkl', 'wb') as file:
    pickle.dump(model_features, file)
print("Model feature names saved as model_features.pkl")

print("Model training script finished.")