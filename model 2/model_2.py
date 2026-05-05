import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ========= تحميل الداتا =========
df = pd.read_csv(
    r"C:\Users\hashe\OneDrive\Desktop\files\senior project\data set\FINAL_DATASET.csv",
    low_memory=False
)

# ========= خذي عينة أولية أخف للتجربة =========
df = df.sample(n=300000, random_state=42)

print("Dataset shape:", df.shape)

# ========= فصل البيانات =========
X = df.drop("label", axis=1)
y = df["label"]

# ========= تحويل True/False إلى 1/0 =========
X = X.replace({
    "True": 1,
    "False": 0,
    True: 1,
    False: 0
})

# ========= تحويل الأعمدة النصية المتبقية =========
X = pd.get_dummies(X)

# ========= معالجة القيم اللانهائية =========
X = X.replace([np.inf, -np.inf], np.nan)

# ========= تعبئة القيم الفارغة =========
X = X.fillna(0)

# ========= اختياري: تحويل كل القيم إلى float64 =========
X = X.astype("float64")

print("X shape after encoding:", X.shape)

# ========= للتأكد =========
print("Any NaN left?", X.isnull().values.any())
print("Any +inf left?", np.isinf(X.to_numpy()).any())

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
    n_estimators=50,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# ========= التنبؤ =========
y_pred = model.predict(X_test)

# ========= التقييم =========
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import pandas as pd

# ========= استخراج أهمية الميزات =========
importances = model.feature_importances_

# ========= ربطها مع أسماء الأعمدة =========
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
})

# ========= ترتيبهم =========
feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

# ========= عرض أهم 20 =========
print("\nTop 20 Important Features:")
print(feature_importance.head(20))

import joblib

joblib.dump(model, "model2.pkl")
joblib.dump(list(X.columns), "model2_columns.pkl")

print("Model 2 saved with columns")