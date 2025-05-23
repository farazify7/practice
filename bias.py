import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in double_scalars")

# Load the Adult dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                'marital-status', 'occupation', 'relationship', 'race', 'sex', 
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, names=column_names, sep=', ', engine='python')

# Prepare data
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Check for bias in original data
print("Original data distribution by sex:")
print(data.groupby(['sex', 'income']).size())

# Create a copy of the sex column before encoding
sex_column = data['sex'].copy()

# One-hot encode categorical features
data_encoded = pd.get_dummies(data, columns=['workclass', 'education', 'marital-status', 
                                           'occupation', 'relationship', 'race', 
                                           'native-country'])

# Split data
X = data_encoded.drop(['income', 'sex'], axis=1)
y = data_encoded['income']

# Use the original sex column for stratification and analysis
X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
    X, y, sex_column, test_size=0.2, random_state=42, stratify=sex_column)

# Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Check accuracy by sex
y_pred = model.predict(X_test_scaled)

# Create masks for male and female samples
male_mask = sex_test == ' Male'
female_mask = sex_test == ' Female'

# Calculate accuracy only if there are samples in each group
print("\nBefore mitigation:")
print(f"Overall accuracy: {accuracy_score(y_test, y_pred):.4f}")

if sum(male_mask) > 0:
    male_accuracy = accuracy_score(y_test[male_mask], y_pred[male_mask])
    print(f"Male accuracy: {male_accuracy:.4f}")
else:
    print("No male samples in test set")

if sum(female_mask) > 0:
    female_accuracy = accuracy_score(y_test[female_mask], y_pred[female_mask])
    print(f"Female accuracy: {female_accuracy:.4f}")
else:
    print("No female samples in test set")

# Mitigate bias using reweighting
male_count = sum(sex_train == ' Male')
female_count = sum(sex_train == ' Female')

print(f"\nTraining data distribution: {male_count} males, {female_count} females")

# Check if we have samples from both genders
if male_count == 0 or female_count == 0:
    print("Warning: Missing gender samples in training data. Using equal weights.")
    sample_weights = np.ones(len(y_train))
else:
    # Calculate class weights
    male_weight = 1.0
    female_weight = male_count / female_count
    print(f"Using weights: Male={male_weight:.2f}, Female={female_weight:.2f}")
    
    # Apply weights
    sample_weights = np.array([male_weight if sex == ' Male' else female_weight for sex in sex_train])

# Retrain with weights
model_mitigated = LogisticRegression(max_iter=1000)
model_mitigated.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# Check accuracy after mitigation
y_pred_mitigated = model_mitigated.predict(X_test_scaled)

print("\nAfter mitigation:")
print(f"Overall accuracy: {accuracy_score(y_test, y_pred_mitigated):.4f}")

if sum(male_mask) > 0:
    male_accuracy = accuracy_score(y_test[male_mask], y_pred_mitigated[male_mask])
    print(f"Male accuracy: {male_accuracy:.4f}")
else:
    print("No male samples in test set")

if sum(female_mask) > 0:
    female_accuracy = accuracy_score(y_test[female_mask], y_pred_mitigated[female_mask])
    print(f"Female accuracy: {female_accuracy:.4f}")
else:
    print("No female samples in test set")