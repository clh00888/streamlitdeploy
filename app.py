import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import shap

title = "Sarcopenic Obesity Risk Prediction System for Hemodialysis Patients"

@st.cache_resource
def load_data():
    model = joblib.load("lgb_model.pkl")
    df = pd.read_csv("traindata.csv")
    explainer = shap.KernelExplainer(lambda X: model.predict_proba(X), df[model.feature_names_in_])
    return model, df, explainer
    
model, df, explainer = load_data()

names = model.feature_names_in_
importances = model.feature_importances_

inputs = {i:j for i,j in zip(names,importances)}
inputs = dict(sorted(inputs.items(), key=lambda x: x[1], reverse=True))

BOOL = {"yes":1, "no":0}

def shap_plot(d, class_index):
    tabs = st.tabs(["**Force Plot**", "**Waterfall Plot**"])
    # 获取单样本的 SHAP 解释
    shap_values = explainer(d.iloc[[0]])  # 注意：用双括号 [[1]] 保持二维形状

    # 绘制 waterfall plot
    plt.figure(figsize=(15, 12), dpi=300)
    
    # shap_values[0] 获取第一个样本，[:, 1] 选择第二个类别
    shap.plots.waterfall(shap_values[0, :, class_index], show=False)
    plt.xlabel(f"SHAP Waterfall Plot for Class {class_index}")
    
    with tabs[1]:
        col = st.columns([1, 3, 1])
        col[1].pyplot(plt.gcf(), dpi=300, use_container_width=True)
    
    plt.clf()
    
    # 绘制 force plot
    plt.figure(figsize=(15, 4), dpi=300)  # force plot 通常用较窄的高度
    
    # shap_values[0] 获取第一个样本，[:, 1] 选择第二个类别
    shap.plots.force(shap_values[0, :, class_index], show=False, matplotlib=True)
    plt.xlabel(f"SHAP Force Plot for Class {class_index}")
    
    with tabs[0]:
        col = st.columns([1, 6, 1])
        col[1].pyplot(plt.gcf(), dpi=300, use_container_width=True)

st.set_page_config(page_title=f"{title}", layout="wide", page_icon="🖥️")

st.markdown("""
<style>
.index_name, [scope="row"] {
    display: none;
}
th {
    background: #01579B;
}
th p {
    color: white;
}
th p, tr p{
    text-align: center;
    font-weight: bold;
}
[kind="primaryFormSubmit"] {
    background: #01579B;
    border-color: #01579B;
}
[kind="primaryFormSubmit"]:hover {
    background: #01579B;
    border-color: #3478CA;
}
[data-baseweb="tab-list"] {
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f'''
    <h1 style="text-align: center; font-size: 26px; font-weight: bold; color: white; background: #01579B; border-radius: 0.5rem; margin-bottom: 15px;">
        {title}
    </h1>''', unsafe_allow_html=True)
    
if "predata" not in st.session_state:
    st.session_state.predata = None
else:
    pass

data = {}
with st.form("inputform"):
    st.markdown("""<div style="color: red; font-size: 18px; text-align: center; border-bottom: 1px solid black; margin-bottom: 10px;">Model Input</div>""", unsafe_allow_html=True)
    
    col = st.columns(6)
    
    k = 0
    for i in inputs:
        if i in ["Male"]:
            data[i] = BOOL[col[k%5].selectbox(f"**{i}**" + f" :red[(importance: {round(float(inputs[i]), 3)})]", BOOL, index=list(BOOL.values()).index(df.iloc[0][i]))]
        elif i in ["Age"]:
            data[i] = col[k%5].number_input(f"**{i}**" + f" :red[(importance: {round(float(inputs[i]), 3)})]", step=1, value=int(df.iloc[0][i]))
        else:
            data[i] = col[k%5].number_input(f"**{i}**" + f" :red[(importance: {round(float(inputs[i]), 3)})]", step=0.001, value=round(df.iloc[0][i], 3)+0.001-0.001)
        k = k+1

    st.session_state.predata = data
    
    output = col[5].selectbox("**:red[Output class]**", [1, 0])
    
    pred_data = pd.DataFrame([st.session_state.predata])
    pred_data = pred_data[names]
    
    pred_data1 = pred_data.copy()
    
    for i in pred_data1.columns.tolist():
        pred_data1[i] = pred_data1[i].apply(lambda x: str(round(x, 2)) if isinstance(x, float) else str(x))
    
    st.html("<div style='font-weight: bold; text-align: center;'>User inputs</div><hr>")
    st.table(pred_data1)
    
    c1 = st.columns(3)
    bt = c1[1].form_submit_button("**Start prediction**", use_container_width=True, type="primary")

def prefun(output_index):
    r_p = float(model.predict_proba(pred_data)[0][output_index])
    
    proba = round(r_p*100, 3)
    
    with st.expander("**Result of prediction**", True):
        st.html(f"<div style='font-weight: bold; text-align: center;'>{'Probability of sarcopenic obesity:' if output_index else 'Probability of non-sarcopenic obesity:'} {proba}%.</div><div style='text-align: center; color: red;'>Note:Prediction results are for reference only and do not replace clinical diagnosis. Clinical decisions should be made by physicians.</div><hr>")
        shap_plot(pred_data, output_index)
        
if bt:
    st.session_state.predata = data
    prefun(output)
else:
    prefun(output)
    
with st.expander("**System Introduction**", True):
    st.info("This web-based tool is based on a machine learning model for predicting the risk of sarcopenic obesity in hemodialysis patients. The model integrates five key clinical variables: age, sex, albumin (Alb), fasting blood glucose (FBG), and phase angle (PhA). It provides individualized risk predictions to facilitate early identification of high-risk individuals and to support clinical decision-making.")
