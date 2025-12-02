import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the preprocessed training features CSV (generated from your updated preprocessing script)
train_csv = 'ballet_features1_combined.csv'  # Update to your actual training CSV path, e.g., 'train_ballet_features2.csv'

df = pd.read_csv(train_csv)

# Prepare data
features = df.drop(['label', 'video_id'], axis=1)  # Drop label and any non-feature columns
labels = df['label']

# Split into train and validation sets (optional; you can use all for training if no val needed)
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train SVM model (you can tweak hyperparameters like kernel, C, etc.)
clf = SVC(kernel='rbf', C=1.0, random_state=42)  # Example: RBF kernel for non-linear separation
clf.fit(X_train_scaled, y_train)

# Optional: Validate
if 'X_val' in locals():
    X_val_scaled = scaler.transform(X_val)
    preds_val = clf.predict(X_val_scaled)
    print(f'Validation Accuracy: {accuracy_score(y_val, preds_val):.2f}')
    print(classification_report(y_val, preds_val))

# Save model and scaler
joblib.dump(clf, 'ballet_svm_model1.pkl')
joblib.dump(scaler, 'ballet_scaler1.pkl')

print('Model and scaler saved successfully!')