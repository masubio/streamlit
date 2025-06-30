# هذا الكود مخصص للعمل في بيئة محلية تحتوي على مكتبة Streamlit
# في حال لم تتوفر streamlit في البيئة الحالية، يُنصح بتشغيل الكود محليًا

# لتحويل النموذج إلى واجهة تفاعلية باستخدام Streamlit

try:
    import streamlit as st
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier

    # تحميل البيانات
    @st.cache_data
    def load_data():
        df = pd.read_csv("نموذج_محاكاة_الهضم_AI.csv")
        encoders = {}
        for column in df.columns:
            if df[column].dtype == 'object':
                enc = LabelEncoder()
                df[column] = enc.fit_transform(df[column])
                encoders[column] = enc
        return df, encoders

    df, encoders = load_data()
    X = df.drop("التقييم النهائي", axis=1)
    y = df["التقييم النهائي"]

    # تدريب النموذج
    model = DecisionTreeClassifier()
    model.fit(X, y)

    # واجهة المستخدم
    st.title("تقييم تركيبة عشبية لهضم صحي")

    new_input = {}
    for col in X.columns:
        options = encoders[col].classes_.tolist()
        selected = st.selectbox(f"{col}:", options)
        new_input[col] = selected

    if st.button("تقييم النتيجة"):
        # ترميز البيانات الجديدة
        input_encoded = [encoders[col].transform([new_input[col]])[0] for col in X.columns]
        prediction = model.predict([input_encoded])
        result = encoders["التقييم النهائي"].inverse_transform(prediction)[0]
        st.success(f"✅ النتيجة المتوقعة: {result}")

except ModuleNotFoundError:
    print("⚠️ مكتبة Streamlit غير مثبتة. يرجى تشغيل هذا الكود على جهازك المحلي بعد تثبيت المكتبة باستخدام:")
    print("pip install streamlit")
