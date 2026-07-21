import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. عنوان التطبيق
st.title("Breast Cancer Classification App 🎗️")
st.write(" Random Forest")

# 2. تحميل البيانات
@st.cache_data
def load_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    return df, cancer.target_names

df, target_names = load_data()

# عرض جزء من البيانات للمستخدم
if st.checkbox("show part of data"):
    st.write(df.head())

# 3. فصل الميزات (Features) والهدف (Target)
X = df.drop("target", axis=1)
y = df["target"]

# 4. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. تدريب النموذج
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# حساب الدقة
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"(Model Accuracy): `{acc * 100:.2f}%`")

# 6. مدخلات المستخدم للتنبؤ
st.sidebar.header("Sample Inputs")
def user_input_features():
    # استخدام متوسط القيمة لكل ميزة كقيمة افتراضية
    mean_radius = st.sidebar.slider("Mean Radius", float(X["mean radius"].min()), float(X["mean radius"].max()), float(X["mean radius"].mean()))
    mean_texture = st.sidebar.slider("Mean Texture", float(X["mean texture"].min()), float(X["mean texture"].max()), float(X["mean texture"].mean()))
    mean_perimeter = st.sidebar.slider("Mean Perimeter", float(X["mean perimeter"].min()), float(X["mean perimeter"].max()), float(X["mean perimeter"].mean()))
    mean_area = st.sidebar.slider("Mean Area", float(X["mean area"].min()), float(X["mean area"].max()), float(X["mean area"].mean()))
    
    data = {
        "mean radius": mean_radius,
        "mean texture": mean_texture,
        "mean perimeter": mean_perimeter,
        "mean area": mean_area
    }
    return pd.DataFrame(data, index=[0])

# إعداد البيانات التي أدخلها المستخدم والتنبؤ بها
input_df = user_input_features()

# تكملة باقي الميزات بالمتوسط لتفادي اختلاف عدد الأعمدة
full_input = X.mean().to_frame().T
full_input.update(input_df)

# 7. التنبؤ وعرض النتيجة
prediction = model.predict(full_input)
prediction_proba = model.predict_proba(full_input)

st.subheader("Predicted Output")
result = target_names[prediction[0]]
st.write(f"Expected Output **{result}**")

st.subheader("Prediction")
st.write(f"Probability that the tumor is benign  (Benign): `{prediction_proba[0][1]*100:.1f}%`")
st.write(f"Probability that the tumor is malignant(Malignant): `{prediction_proba[0][0]*100:.1f}%`")