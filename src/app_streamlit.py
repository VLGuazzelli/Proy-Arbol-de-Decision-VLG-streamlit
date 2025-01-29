from pickle import load
import streamlit as st

model = load(open("/workspaces/Proy-Arbol-de-Decision-VLG-streamlit/src/decision_tree_classifier_default_42.sav", "rb"))
scaler = load(open("/workspaces/Proy-Arbol-de-Decision-VLG-streamlit/src/model_scaler.sav", "rb"))  # Asegúrate de haber guardado el scaler

class_dict = {
    "0": "Sin Diabetes",
    "1": "Con Diabetes"
}

st.title("Diabetes - Model prediction")


preg = st.slider("Pregnancies", min_value = 0.0, max_value = 20.0, step = 1.0)
gluc = st.slider("Glucose", min_value = 0.0, max_value = 200.0, step = 1.0)
bloodpress = st.slider("BloodPressure", min_value = 0.0, max_value = 130.0, step = 1.0)
skinthick = st.slider("SkinThickness", min_value = 0.0, max_value = 100.0, step = 1.0)
ins = st.slider("Insulin", min_value = 0.0, max_value = 850.0, step = 1.0)
bmi = st.slider("BMI", min_value = 0.0, max_value = 70.0, step = 0.5)
DiaPedFun = st.slider("DiabetesPedigreeFunction", min_value = 0.0, max_value = 3.0, step = 0.1)
age = st.slider("Age", min_value = 18.0, max_value = 90.0, step = 1.0)

data = scaler.transform([[preg, gluc, bloodpress, skinthick, ins, bmi, DiaPedFun, age]])

if st.button("Predict"):
    prediction = str(model.predict(data)[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)