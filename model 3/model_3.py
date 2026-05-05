import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ========= تحميل الداتا =========
df = pd.read_csv(
    r"C:\Users\hashe\OneDrive\Desktop\files\senior project\data set\CIC_merged_small.csv",
    low_memory=False
)

# ========= أخذ عينة أولية =========
df = df.sample(n=300000, random_state=42)

print("Original dataset shape:", df.shape)

# ========= توحيد اسم اللابل =========
if "Label" in df.columns:
    df = df.rename(columns={"Label": "label"})

# ========= تجميع الهجمات =========
def map_attack(label):
    label = str(label).upper()

    if "DDOS" in label or "DOS" in label:
        return "DoS"
    elif "MITM" in label:
        return "MITM"
    elif "SPOOF" in label:
        return "Spoofing"
    elif "RECON" in label:
        return "Recon"
    elif "BENIGN" in label:
        return "Benign"
    else:
        return "Other"

df["attack_type"] = df["label"].apply(map_attack)

print("\nAttack distribution before filtering:")
print(df["attack_type"].value_counts())

# ========= حذف الفئات غير المطلوبة =========
df = df[~df["attack_type"].isin(["Other", "Benign"])].copy()

print("\nAttack distribution after removing Other and Benign:")
print(df["attack_type"].value_counts())

# ========= Balancing لفئة DoS فقط =========
dos_target = 30000

df_dos = df[df["attack_type"] == "DoS"]
df_other_attacks = df[df["attack_type"] != "DoS"]

if len(df_dos) > dos_target:
    df_dos = df_dos.sample(n=dos_target, random_state=42)

df_balanced = pd.concat([df_dos, df_other_attacks], ignore_index=True)

print("\nAttack distribution after DoS balancing:")
print(df_balanced["attack_type"].value_counts())

print("\nBalanced dataset shape:", df_balanced.shape)

# ========= فصل الميزات والهدف =========
X = df_balanced.drop(columns=["label", "attack_type"], errors="ignore")
y = df_balanced["attack_type"]

# ========= تنظيف وتحويل =========
X = X.replace({
    "True": 1,
    "False": 0,
    True: 1,
    False: 0
})

X = pd.get_dummies(X)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = X.astype("float64")

print("\nX shape after encoding:", X.shape)
print("Any NaN left?", X.isnull().values.any())
print("Any +inf left?", np.isinf(X.to_numpy()).any())

# ========= تحويل الفئات النصية إلى أرقام =========
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded classes:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i} -> {cls}")

# ========= تقسيم =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# ========= مودل XGBoost =========
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss"
)

print("\nTraining balanced attack-type model with XGBoost...")
model.fit(X_train, y_train)

# ========= التنبؤ =========
y_pred = model.predict(X_test)

# ========= التقييم =========
print("\n=== XGBOOST MODEL 3 RESULTS ===")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# ========= أهم الـ features =========
importances = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop 20 Important Features:")
print(feature_importance.head(20))

# ========= حفظ أهم الـ features =========
feature_output = r"C:\Users\hashe\OneDrive\Desktop\files\senior project\data set\model3_xgboost_top_features.csv"
feature_importance.to_csv(feature_output, index=False)

print("\nTop features saved to:")
print(feature_output)

import joblib

joblib.dump(model, "model3.pkl")
joblib.dump(list(X.columns), "model3_columns.pkl")
joblib.dump(label_encoder, "model3_label_encoder.pkl")

print("Model 3 saved with columns and label encoder")