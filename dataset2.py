import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load the dataset
file_path = "C:/project_1/diabetes_predictor_app/Indian_Food_Nutrition_Processed.csv"
df_food = pd.read_csv(file_path)
df_food.columns = df_food.columns.str.strip()

# Use 'Dish Name' as the text feature
text_column = 'Dish Name'
calorie_column = 'Calories (kcal)'
sugar_column = 'Free Sugar (g)'

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=500)
X_food_features = tfidf_vectorizer.fit_transform(df_food[text_column]).toarray()

# Prepare targets
y_calories = df_food[calorie_column]
y_sugar = df_food[sugar_column]

# Train-test split for calories
X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(X_food_features, y_calories, test_size=0.2, random_state=42)
calorie_model = LinearRegression()
calorie_model.fit(X_train_cal, y_train_cal)
y_pred_cal = calorie_model.predict(X_test_cal)

# Train-test split for sugar
X_train_sug, X_test_sug, y_train_sug, y_test_sug = train_test_split(X_food_features, y_sugar, test_size=0.2, random_state=42)
sugar_model = LinearRegression()
sugar_model.fit(X_train_sug, y_train_sug)
y_pred_sug = sugar_model.predict(X_test_sug)

# Plot predictions vs actual for calories
plt.figure(figsize=(10, 5))
plt.scatter(y_test_cal, y_pred_cal, alpha=0.7, color='blue')
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Calories Prediction: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig("calories_prediction.png")

# Plot predictions vs actual for sugar
plt.figure(figsize=(10, 5))
plt.scatter(y_test_sug, y_pred_sug, alpha=0.7, color='green')
plt.xlabel("Actual Sugar (g)")
plt.ylabel("Predicted Sugar (g)")
plt.title("Sugar Prediction: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig("sugar_prediction.png")

# Feature importance (coefficients)
calorie_importance = pd.Series(calorie_model.coef_, index=tfidf_vectorizer.get_feature_names_out()).sort_values(ascending=False)
sugar_importance = pd.Series(sugar_model.coef_, index=tfidf_vectorizer.get_feature_names_out()).sort_values(ascending=False)

# Plot top 20 important features for calories
plt.figure(figsize=(12, 6))
calorie_importance.head(20).plot(kind='bar', color='blue')
plt.title("Top 20 Important Features for Calorie Prediction")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("calorie_feature_importance.png")

# Plot top 20 important features for sugar
plt.figure(figsize=(12, 6))
sugar_importance.head(20).plot(kind='bar', color='green')
plt.title("Top 20 Important Features for Sugar Prediction")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("sugar_feature_importance.png")

print("✅ Visualizations saved successfully!")

