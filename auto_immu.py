import streamlit as st
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
import pickle
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
from lime import lime_tabular
import streamlit.components.v1 as components

st.set_page_config(page_title="Drug-induced Autoimmunity (DIA) Prediction with LIME", layout="wide")

# 加载模型和标准化器
with open('scaler_and_model.pkl', 'rb') as f:
    scaler, best_estimator_eec = pickle.load(f)
# 加载训练数据（如果之前有存储的话）
with open('Xtrain_std.pkl', 'rb') as f:
    Xtrain_std = pickle.load(f)

# 将Xtrain_std转换为NumPy数组
Xtrain_std_np = Xtrain_std.values  # 转换为NumPy数组

# 65个分子描述符名称
descriptor_names = ['BalabanJ', 'Chi0', 'EState_VSA1', 'EState_VSA10', 'EState_VSA4', 'EState_VSA6', 
                    'EState_VSA9', 'HallKierAlpha', 'Ipc', 'Kappa3', 'NHOHCount', 'NumAliphaticHeterocycles',
                    'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticRings', 'PEOE_VSA10',
                    'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7', 
                    'PEOE_VSA9', 'RingCount', 'SMR_VSA10', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA9', 
                    'SlogP_VSA10', 'SlogP_VSA5', 'SlogP_VSA8', 'VSA_EState8', 'fr_ArN', 'fr_Ar_NH', 'fr_C_O', 
                    'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_alkyl_carbamate', 'fr_allylic_oxid', 'fr_amide', 
                    'fr_aryl_methyl', 'fr_azo', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_dihydropyridine', 'fr_epoxide', 
                    'fr_ether', 'fr_furan', 'fr_guanido', 'fr_hdrzone', 'fr_imide', 'fr_ketone_Topliss', 'fr_lactam', 
                    'fr_methoxy', 'fr_morpholine', 'fr_nitro_arom', 'fr_para_hydroxylation', 'fr_phos_ester', 'fr_piperdine', 
                    'fr_pyridine', 'fr_sulfide', 'fr_term_acetylene', 'fr_unbrch_alkane']

# 提取分子描述符的函数
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptors = calculator.CalcDescriptors(mol)
    return np.array(descriptors)

# LIME解释器初始化
explainer = lime_tabular.LimeTabularExplainer(
    training_data=Xtrain_std_np,  # 使用NumPy数组
    feature_names=descriptor_names,
    class_names=['DIA_negative', 'DIA_positive'],
    mode='classification'
)

# Streamlit app
st.title("Drug-induced Autoimmunity (DIA) Prediction with LIME")

# 输入 SMILES 结构
smiles_input = st.text_input("Enter a drug SMILES structure")

if st.button("Predict"):
    if smiles_input:
        descriptors = get_descriptors(smiles_input)
        
        if descriptors is None:
            st.error("Invalid SMILES structure")
        else:
            # 显示原始描述符
            st.subheader("Original Descriptors (before scaling)")
            descriptors_df = pd.DataFrame([descriptors], columns=descriptor_names)
            st.write(descriptors_df)

            # 将描述符标准化
            descriptors_std = scaler.transform([descriptors])

            # 显示标准化后的描述符
            st.subheader("Scaled Descriptors (after scaling)")
            descriptors_std_df = pd.DataFrame(descriptors_std, columns=descriptor_names)
            st.write(descriptors_std_df)

            # 使用模型进行预测并获取概率值
            prediction_prob = best_estimator_eec.predict_proba(descriptors_std)[0]

            # 显示预测的概率值
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                "Class 0 (DIA_negative)": prediction_prob[0],
                "Class 1 (DIA_positive)": prediction_prob[1]
            }, index=[0])
            st.write(prob_df)

            # 最终预测结果
            if prediction_prob[1] > prediction_prob[0]:
                st.markdown(
                    f"<div style='background-color: lightgreen; border: 2px solid green; padding: 10px; border-radius: 5px;'>"
                    f"<h3 style='font-size: 30px;'>The drug is predicted to be associated with autoimmune disease.</h3>"
                    f"<span style='color:red; font-size: 20px;'>Probability: {prediction_prob[1]:.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background-color: lightgreen; border: 2px solid green; padding: 10px; border-radius: 5px;'>"
                    f"<h3 style='font-size: 30px;'>The drug is predicted NOT to be associated with autoimmune disease.</h3>"
                    f"<span style='color:red; font-size: 20px;'>Probability: {prediction_prob[0]:.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # 使用LIME解释预测结果
            np.random.seed(42)  # 设定随机种子

            st.subheader("LIME Explanation")
            exp = explainer.explain_instance(descriptors_std[0], best_estimator_eec.predict_proba, num_features=10)
            components.html(exp.as_html(), height=1000, width=2000, scrolling=True)
