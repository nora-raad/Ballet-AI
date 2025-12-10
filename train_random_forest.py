import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Switched from SVM
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import numpy as np

#preprocessed training features CSV 
train_csv = 'ballet_features1_combined.csv'  

df = pd.read_csv(train_csv)

# Prepare data
features = df.drop(['label', 'video_id'], axis=1)  # Drop label and any non-feature columns
labels = df['label']


features = features.fillna(0)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale features (handles mixed scales like vis counts)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train RandomForest model with basic tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
clf.fit(X_train_scaled, y_train)

print(f'Best params: {clf.best_params_}')
print(f'Cross-validation accuracy: {np.mean(cross_val_score(clf.best_estimator_, X_train_scaled, y_train, cv=5)):.2f}')

# Validate on holdout set
preds_val = clf.predict(X_val_scaled)
print(f'Validation Accuracy: {accuracy_score(y_val, preds_val):.2f}')
print(classification_report(y_val, preds_val))

# Feature importances (to see what's helping/hurting)
importances = clf.best_estimator_.feature_importances_
feature_names = features.columns
sorted_idx = np.argsort(importances)[::-1]
print('Feature Importances:')
for i in sorted_idx:
    print(f'{feature_names[i]}: {importances[i]:.4f}')

# Plot importances
plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx], importances[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()  # Pops up a plot—close it to continue

# Save model and scaler
joblib.dump(clf.best_estimator_, 'ballet_rf_model1.pkl')  # Updated filename
joblib.dump(scaler, 'ballet_rf_scaler1.pkl')

print('Model and scaler saved successfully!')