import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ========= تحميل الداتا =========
df = pd.read_csv(
    r"C:\Users\hashe\OneDrive\Desktop\files\senior project\data set\TON_merged_final.csv",
    low_memory=False
)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ========= الأعمدة الخاصة بالجهاز =========
device_cols = [
    "device_Fridge",
    "device_Motion_Light",
    "device_Thermostat"
]

# تأكد أنها موجودة
missing = [col for col in device_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing device columns: {missing}")

# ========= تحويل True/False النصية إلى 1/0 إذا لزم =========
df = df.replace({
    "True": 1,
    "False": 0,
    True: 1,
    False: 0
})

# ========= إعادة تكوين عمود device =========
# نأخذ اسم العمود الذي قيمته 1
y = df[device_cols].idxmax(axis=1)

# تنظيف الأسماء
y = y.replace({
    "device_Fridge": "Fridge",
    "device_Motion_Light": "Motion_Light",
    "device_Thermostat": "Thermostat"
})

print("\nDevice distribution:")
print(y.value_counts())

# ========= حذف أعمدة الهدف من الـ features =========
X = df.drop(columns=device_cols + ["label"], errors="ignore")

# ========= تنظيف =========
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = X.astype("float64")

print("\nX shape:", X.shape)

# ========= تقسيم =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ========= تدريب المودل =========
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Model 1 (Device Identification)...")
model.fit(X_train, y_train)

# ========= التنبؤ =========
y_pred = model.predict(X_test)

# ========= التقييم =========
print("\n=== MODEL 1 RESULTS ===")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ========= أهم الـ features =========
importances = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop 20 Important Features:")
print(feature_importance.head(20))

import joblib

joblib.dump(model, "model1.pkl")
joblib.dump(list(X.columns), "model1_columns.pkl")

print("Model 1 saved with columns")