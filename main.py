import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load and prepare data
data = pd.read_csv('credit_scoring_dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Define and tune model
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(rf, param_grid, scoring='f1', cv=5, n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model
best_rf = grid_search.best_estimator_

# Predict on test set
y_pred = best_rf.predict(X_test_scaled)
y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# Feature importance
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]
sorted_features = feature_names[indices]
sorted_importances = importances[indices] * 100

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
for i, v in enumerate(sorted_importances):
    plt.text(v + 1, i, f"{int(round(v))}%", va='center')
plt.title("What Affects Your Creditworthiness?")
plt.xlabel("Relative Importance (%)")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Save model and scaler
joblib.dump(best_rf, 'credit_scoring_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved.")

# Final model metrics
print("\nFinal Model Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Save processed data
data.to_csv('credit_scoring_dataset_final.csv', index=False)
print("Processed dataset saved.")

