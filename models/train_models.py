import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

data_path = '../data/preprocessed_data.csv'

# Load dataset
df = pd.read_csv(data_path, low_memory=False)

# Define target columns and features
diet_target_column = 'Diet_type_dash' 
workout_target_column = 'Type_Cardio' 

# Prepare features and targets for diet recommendation
diet_features = df.drop(columns=[diet_target_column])
diet_target = df[diet_target_column]

# Prepare features and targets for workout recommendation
workout_features = df.drop(columns=[workout_target_column])
workout_target = df[workout_target_column]

# Convert columns to numeric where applicable
diet_features = diet_features.apply(pd.to_numeric, errors='coerce')
workout_features = workout_features.apply(pd.to_numeric, errors='coerce')

# Fill missing values with 0
diet_features.fillna(0, inplace=True)
workout_features.fillna(0, inplace=True)

# Encode categorical features
diet_features = pd.get_dummies(diet_features, drop_first=True)
workout_features = pd.get_dummies(workout_features, drop_first=True)

# Convert target variables to numeric
diet_target, diet_target_mapping = pd.factorize(diet_target)
workout_target, workout_target_mapping = pd.factorize(workout_target)

# Train-test split
X_train_diet, X_test_diet, y_train_diet, y_test_diet = train_test_split(diet_features, diet_target, test_size=0.2, random_state=42)
X_train_workout, X_test_workout, y_train_workout, y_test_workout = train_test_split(workout_features, workout_target, test_size=0.2, random_state=42)

# Standardize features
scaler_diet = StandardScaler()
X_train_diet = scaler_diet.fit_transform(X_train_diet)
X_test_diet = scaler_diet.transform(X_test_diet)

scaler_workout = StandardScaler()
X_train_workout = scaler_workout.fit_transform(X_train_workout)
X_test_workout = scaler_workout.transform(X_test_workout)

# Compute class weights
class_weights_diet = compute_class_weight('balanced', classes=np.unique(diet_target), y=diet_target)
class_weights_diet = dict(enumerate(class_weights_diet))

class_weights_workout = compute_class_weight('balanced', classes=np.unique(workout_target), y=workout_target)
class_weights_workout = dict(enumerate(class_weights_workout))

# Train diet recommendation model with class weights
diet_model = RandomForestClassifier(class_weight=class_weights_diet, random_state=42)
diet_model.fit(X_train_diet, y_train_diet)

# Train workout recommendation model with class weights
workout_model = RandomForestClassifier(class_weight=class_weights_workout, random_state=42)
workout_model.fit(X_train_workout, y_train_workout)

# Evaluate diet recommendation model
y_pred_diet = diet_model.predict(X_test_diet)
print("Diet Recommendation Model Evaluation:")
print(classification_report(y_test_diet, y_pred_diet, target_names=diet_target_mapping, zero_division=1))
print("Accuracy:", accuracy_score(y_test_diet, y_pred_diet))

# Evaluate workout recommendation model
y_pred_workout = workout_model.predict(X_test_workout)
print("Workout Recommendation Model Evaluation:")
print(classification_report(y_test_workout, y_pred_workout, target_names=workout_target_mapping, zero_division=1))
print("Accuracy:", accuracy_score(y_test_workout, y_pred_workout))

# Save the models and scalers
joblib.dump(diet_model, 'diet_recommendation_model.pkl')
joblib.dump(workout_model, 'workout_recommendation_model.pkl')
joblib.dump(scaler_diet, 'scaler_diet.pkl')
joblib.dump(scaler_workout, 'scaler_workout.pkl')
