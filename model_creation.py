import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- 1. Data Loading and Cleaning ---
def clean_data(df):
    """Cleans the survey data."""
    # Clean Age
    df = df[(df['Age'] >= 18) & (df['Age'] <= 75)]

    # Clean Gender
    df['Gender'] = df['Gender'].str.lower()
    male_terms = ['m', 'male', 'cis male', 'mal', 'male (cis)', 'maile', 'make', 'man']
    female_terms = ['f', 'female', 'cis female', 'woman', 'femake', 'femail']
    
    df.loc[df['Gender'].isin(male_terms), 'Gender'] = 'Male'
    df.loc[df['Gender'].isin(female_terms), 'Gender'] = 'Female'
    df.loc[~df['Gender'].isin(['Male', 'Female']), 'Gender'] = 'Other'

    return df

# Load data
try:
    data = pd.read_csv('survey.csv')
    data = clean_data(data)
except FileNotFoundError:
    print("Error: survey.csv not found. Please place it in the same directory.")
    exit()

# --- 2. Feature Selection and Preprocessing ---

# Select features and target
features = ['Age', 'Gender', 'family_history', 'work_interfere', 'benefits', 'care_options', 'anonymity']
target = 'treatment'

# Drop rows with missing values in the selected columns
df_model = data[features + [target]].dropna()

# Handle 'Don't know' as a separate category, not a missing value for some columns
# For simplicity here, we'll keep them as is since OneHotEncoder will handle them.

# Separate features (X) and target (y)
X = df_model[features]
y = df_model[target]

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Define preprocessing steps for columns
# We will use OneHotEncoder for all categorical features
categorical_features = ['Gender', 'family_history', 'work_interfere', 'benefits', 'care_options', 'anonymity']
numeric_features = ['Age']

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 3. Model Training ---

# Create the pipeline with the preprocessor and the classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train the model
model_pipeline.fit(X_train, y_train)

# --- 4. Model Evaluation ---

# Make predictions
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print("--- Model Training Complete ---")
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# --- 5. Save the Model ---
joblib.dump(model_pipeline, 'mental_health_model.joblib')
print("\nModel saved to 'mental_health_model.joblib'")
joblib.dump(le, 'target_encoder.joblib')
print("Target encoder saved to 'target_encoder.joblib'")