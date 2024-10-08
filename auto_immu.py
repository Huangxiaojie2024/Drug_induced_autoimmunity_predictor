import streamlit as st
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型和标准化器
with open('scaler_and_model.pkl', 'rb') as f:
    scaler, best_estimator_eec = pickle.load(f)

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

# Streamlit app
st.title("Drug-induced Autoimmune Disease Prediction")

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
                "Class 0 (No Autoimmune)": prediction_prob[0],
                "Class 1 (Autoimmune)": prediction_prob[1]
            }, index=[0])
            st.write(prob_df)

            # 最终预测结果
            if prediction_prob[1] > prediction_prob[0]:
                st.success("The drug is predicted to be associated with autoimmune disease.")
            else:
                st.success("The drug is predicted NOT to be associated with autoimmune disease.")

            # ------------------------------
            # SHAP 解释
            # 使用SHAP的KernelExplainer进行解释
            # 使用模型的标准化训练数据作为背景数据
            background_data = scaler.transform([descriptors])  # 使用标准化后的数据
            explainer = shap.KernelExplainer(
                best_estimator_eec.predict_proba, background_data)  # 使用模型的概率预测函数
            
            # 解释当前药物的预测
            shap_values = explainer.shap_values(descriptors_std[0])  # 获取SHAP值

            # 选择前10个贡献最大的特征（如果特征过多，导致图像过大，可以减少特征数量）
            top_n = 10
            shap_values_top = shap_values[1][:top_n]  # 选择类1的前top_n个贡献
            descriptor_names_top = descriptor_names[:top_n]  # 选择前top_n个特征名称

            # 将 SHAP 值转化为 SHAP Explanation 对象
            shap_explanation = shap.Explanation(values=shap_values_top, 
                                               base_values=explainer.expected_value[1], 
                                               data=descriptors_std[0][:top_n], 
                                               feature_names=descriptor_names_top)

            # SHAP瀑布图显示
            st.subheader("SHAP Explanation (Waterfall Plot)")

            # 使用matplotlib设置图像大小以避免超出显示限制
            plt.figure(figsize=(8, 6))  # 设定合适的图像大小
            shap.initjs()  # 初始化SHAP的js可视化
            st.pyplot(shap.waterfall_plot(shap_explanation))  # 显示Class 1的SHAP瀑布图

            # 在SHAP图中显示原始特征值
            st.subheader("SHAP Explanation with Original Feature Values")

            # 创建一个DataFrame以显示每个特征的原始值和贡献
            shap_explanation_df = pd.DataFrame({
                'Feature': descriptor_names_top,
                'Original Value': descriptors[:top_n],  # 原始特征值
                'SHAP Value': shap_values_top  # SHAP贡献值
            })

            # 显示带有原始值和SHAP贡献的解释
            st.write(shap_explanation_df)
    else:
        st.error("Please enter a valid SMILES structure.")
