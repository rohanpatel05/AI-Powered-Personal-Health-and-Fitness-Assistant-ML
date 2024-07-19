import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load datasets
diets_all = pd.read_csv("../data/diets/All_Diets.csv")
gym_exercises = pd.read_csv("../data/megaGymDataset.csv")
nutrition = pd.read_csv("../data/nutrition.csv")

# Handle missing values
diets_all.dropna(inplace=True)
gym_exercises.dropna(inplace=True)
nutrition.dropna(inplace=True)

# Function to convert non-numeric entries to numeric
def convert_to_numeric(value):
    try:
        return float(value)
    except ValueError:
        numeric_value = ''.join([c for c in value if c.isdigit() or c == '.'])
        return float(numeric_value) if numeric_value else 0

# Convert non-numeric values in the nutrition dataset
nutrition.iloc[:, 2:] = nutrition.iloc[:, 2:].applymap(convert_to_numeric).astype(float)

# Normalize numerical features
scaler_diets = StandardScaler()
diets_all[['Protein(g)', 'Carbs(g)', 'Fat(g)']] = scaler_diets.fit_transform(diets_all[['Protein(g)', 'Carbs(g)', 'Fat(g)']])

scaler_nutrition = StandardScaler()
nutrition.iloc[:, 2:] = scaler_nutrition.fit_transform(nutrition.iloc[:, 2:])

# One-hot encoding for categorical variables
diets_all = pd.get_dummies(diets_all, columns=['Diet_type', 'Cuisine_type'])
gym_exercises = pd.get_dummies(gym_exercises, columns=['Type', 'BodyPart', 'Equipment', 'Level'])

# Combine all datasets
final_dataset = pd.concat([diets_all, gym_exercises, nutrition], axis=1)

# Handle remaining NaN values by filling them with 0
final_dataset.fillna(0, inplace=True)

# Downcast object dtype arrays explicitly
final_dataset = final_dataset.infer_objects()

# Display final dataset
print(final_dataset.head())

# Save the final preprocessed dataset
final_dataset.to_csv('../data/preprocessed_data.csv', index=False)
