import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ================= تحميل الموديلات =================
print("Parallel IDS started")

BASE_PATH = r"C:\Users\hashe\source\repos\parallel_ids"

model1 = joblib.load(BASE_PATH + r"\model1.pkl")
model2 = joblib.load(BASE_PATH + r"\model2.pkl")
model3 = joblib.load(BASE_PATH + r"\model3.pkl")

model1_cols = joblib.load(BASE_PATH + r"\model1_columns.pkl")
model2_cols = joblib.load(BASE_PATH + r"\model2_columns.pkl")
model3_cols = joblib.load(BASE_PATH + r"\model3_columns.pkl")

model3_label_encoder = joblib.load(BASE_PATH + r"\model3_label_encoder.pkl")

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

# تحويل True/False إلى 1/0
X = X.replace({
    "True": 1,
    "False": 0,
    True: 1,
    False: 0
})

# تحويل النصوص إلى أعمدة رقمية
X = pd.get_dummies(X)

# إزالة الأعمدة المكررة
X = X.loc[:, ~X.columns.duplicated()]

# تنظيف البيانات
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

        # ================= Model 1 =================
        device_prediction = model1.predict(sample1)

        # ================= Model 2 =================
        malicious_prediction = model2.predict(sample2)

        # تحويل 0/1 إلى نص
        if malicious_prediction[0] == 1:

            malicious_text = "Malicious"

            # ================= Model 3 =================
            attack_prediction = model3.predict(sample3)

            attack_name = model3_label_encoder.inverse_transform(
                [int(attack_prediction[0])]
            )[0]

        else:

            malicious_text = "Normal"

            # إذا طبيعي لا يوجد هجوم
            attack_name = "No Attack"

        # ================= الطباعة =================
        print(f"Sample {i+1}")
        print("Device Type:", device_prediction[0])
        print("Malicious/Normal:", malicious_text)
        print("Attack Type:", attack_name)
        print("-----------------------------------")

    except Exception as e:
        print(f"Error in sample {i+1}: {e}")

        print("\nParallel IDS finished")
        # طباعة النتائج
        print(f"Sample {i+1}")
        print("Device Type:", device_prediction[0])
        print("Malicious/Normal:", malicious_text)
        print("Attack Type:", attack_name)
        print("-----------------------------------")

    except Exception as e:
        print(f"Error in sample {i+1}: {e}")

print("\nParallel IDS finished")
