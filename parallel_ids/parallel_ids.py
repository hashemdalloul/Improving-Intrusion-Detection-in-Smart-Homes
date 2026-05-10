import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ================= تحميل الموديلات =================
print("Parallel IDS started")

model1 = joblib.load(r"C:\Users\hashe\source\repos\model 1\model1.pkl")
model2 = joblib.load(r"C:\Users\hashe\source\repos\model 2\model2.pkl")
model3 = joblib.load(r"C:\Users\hashe\source\repos\model 3\model3.pkl")

model1_cols = joblib.load(r"C:\Users\hashe\source\repos\model 1\model1_columns.pkl")
model2_cols = joblib.load(r"C:\Users\hashe\source\repos\model 2\model2_columns.pkl")
model3_cols = joblib.load(r"C:\Users\hashe\source\repos\model 3\model3_columns.pkl")

model3_encoder = joblib.load(r"C:\Users\hashe\source\repos\model 3\model3_label_encoder.pkl")

print("All models loaded successfully")

# ================= تحميل بيانات اختبار =================
df = pd.read_csv(
    r"C:\Users\hashe\OneDrive\Desktop\files\senior project\data set\FINAL_DATASET.csv",
    low_memory=False
)

print("Dataset loaded:", df.shape)

# أخذ عينة صغيرة للتجربة
df = df.sample(n=10, random_state=42)

# ================= تجهيز البيانات =================
X = df.drop("label", axis=1, errors="ignore")

print("Test sample:", X.shape)

X = X.replace({
    "True": 1,
    "False": 0,
    True: 1,
    False: 0
})

X = pd.get_dummies(X)

# إزالة أي أعمدة مكررة
X = X.loc[:, ~X.columns.duplicated()]

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = X.astype("float64")

# ================= التنبؤ =================
print("\n========= IDS RESULTS =========\n")

for i in range(len(X)):

    sample = X.iloc[[i]]

    try:
        # ترتيب الأعمدة حسب كل مودل
        sample1 = sample.reindex(columns=model1_cols, fill_value=0)
        sample2 = sample.reindex(columns=model2_cols, fill_value=0)
        sample3 = sample.reindex(columns=model3_cols, fill_value=0)

        # التنبؤ من كل مودل
        device_prediction = model1.predict(sample1.to_numpy())
        malicious_prediction = model2.predict(sample2.to_numpy())
        attack_prediction = model3.predict(sample3.to_numpy())

        # تحويل رقم الهجوم إلى اسم الهجوم
        attack_name = model3_encoder.inverse_transform(
            [int(attack_prediction[0])]
        )[0]

        print(f"Sample {i+1}")
        print("Device Type:", device_prediction[0])
        print("Malicious/Normal:", malicious_prediction[0])
        print("Attack Type:", attack_name)
        print("-----------------------------------")

    except Exception as e:
        print(f"Error in sample {i+1}: {e}")

print("\nParallel IDS finished")