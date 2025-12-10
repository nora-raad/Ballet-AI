import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# preprocessed training features CSV 
train_csv = 'ballet_features1_combined.csv'  

df = pd.read_csv(train_csv)

# Prepare data
features = df.drop(['label', 'video_id'], axis=1)  # Drop label and any non-feature columns
labels = df['label']

# Split into train and validation sets 
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train SVM model 
clf = SVC(kernel='rbf', C=1.0, random_state=42)  
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