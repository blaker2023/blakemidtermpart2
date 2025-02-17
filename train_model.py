import pandas as pd

# Load dataset
file_path = "StudentsPerformance.csv"
df = pd.read_csv(file_path)

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Check data types
print(df.info())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Keep only relevant columns
df = df[['math score', 'reading score', 'writing score', 'race/ethnicity']]

# Encode target variable
label_encoder = LabelEncoder()
df['race/ethnicity'] = label_encoder.fit_transform(df['race/ethnicity'])

# Split dataset
X = df[['math score', 'reading score', 'writing score']]
y = df['race/ethnicity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save model, label encoder, and scaler
with open("model.pkl", "wb") as f:
    pickle.dump((model, label_encoder, scaler), f)

print("✅ Model trained and saved as model.pkl!")