import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components

st.set_page_config(page_title="Drug-induced Autoimmunity (DIA) Prediction", layout="wide")

# 加载模型和标准化器
@st.cache_resource
def load_model():
    with open('scaler_and_model.pkl', 'rb') as f:
        scaler, best_estimator_eec = pickle.load(f)
    with open('Xtrain_std.pkl', 'rb') as f:
        Xtrain_std = pickle.load(f)
    return scaler, best_estimator_eec, Xtrain_std

scaler, best_estimator_eec, Xtrain_std = load_model()

# 65个最佳分子描述符名称
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

# Streamlit app
st.title("Drug-induced Autoimmunity (DIA) Prediction")

# 文件上传
uploaded_file = st.sidebar.file_uploader("Upload ChemDes descriptors CSV file", type=['csv'])

if uploaded_file is not None:
    # 读取CSV文件
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Successfully loaded file with {len(df)} compounds")
        
        # 检查是否包含所需的描述符
        missing_descriptors = [desc for desc in descriptor_names if desc not in df.columns]
        if missing_descriptors:
            st.error(f"Missing required descriptors: {', '.join(missing_descriptors)}")
        else:
            # 提取65个最佳描述符
            X = df[descriptor_names].values
            
            # 显示原始描述符
            with st.expander("Show Original Descriptors"):
                st.dataframe(pd.DataFrame(X, columns=descriptor_names))

            # 标准化描述符
            X_std = scaler.transform(X)
            
            # 显示标准化后的描述符
            with st.expander("Show Scaled Descriptors"):
                st.dataframe(pd.DataFrame(X_std, columns=descriptor_names))

            # 批量预测
            predictions_prob = best_estimator_eec.predict_proba(X_std)
            
            # 创建结果DataFrame
            results_df = pd.DataFrame({
                "Compound_ID": range(1, len(df) + 1),
                "DIA_negative_prob": predictions_prob[:, 0],
                "DIA_positive_prob": predictions_prob[:, 1],
                "Prediction": ["DIA Positive" if p > 0.5 else "DIA Negative" for p in predictions_prob[:, 1]]
            })
            
            # 显示预测结果
            st.subheader("Prediction Results")
            st.dataframe(results_df)
            
            # 下载预测结果
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Prediction Results",
                data=csv,
                file_name="dia_predictions.csv",
                mime="text/csv"
            )
            
            # SHAP值分析
            st.subheader("SHAP Analysis")
            
            # 创建列布局
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.write("Select a compound:")
                # 为每个化合物创建按钮
                for i in range(len(df)):
                    compound_status = f"DIA {'Positive' if predictions_prob[i, 1] > 0.5 else 'Negative'}"
                    if st.button(
                        f"Compound {i+1}\n({compound_status})",
                        key=f"compound_{i}",
                        help=f"Probability of DIA: {predictions_prob[i, 1]:.3f}"
                    ):
                        st.session_state.selected_compound = i

            with col2:
                if 'selected_compound' in st.session_state:
                    compound_index = st.session_state.selected_compound
                    
                    # 显示选中化合物的详细信息
                    st.write(f"### Compound {compound_index + 1} Details")
                    st.write(f"DIA Prediction: {'Positive' if predictions_prob[compound_index, 1] > 0.5 else 'Negative'}")
                    st.write(f"Probability of DIA: {predictions_prob[compound_index, 1]:.3f}")
                    
                    # 创建SHAP解释器
                    explainer = shap.KernelExplainer(best_estimator_eec.predict_proba, Xtrain_std)
                    
                    # 计算SHAP值
                    with st.spinner('Calculating SHAP values...'):
                        shap_values = explainer.shap_values(X_std[compound_index:compound_index+1], nsamples=150)
                        
                        # 保存 SHAP 力图为 HTML 文件，显示特征名称和原始数值
                        force_plot = shap.force_plot(
                            explainer.expected_value[1], 
                            shap_values[1][0],  # 使用第二个类别（正类）的SHAP值
                            X[compound_index],  # 使用原始描述符
                            feature_names=descriptor_names,  # 显示特征名称
                            show=False
                        )
                        
                        html_file = f"shap_force_plot_{compound_index}.html"
                        shap.save_html(html_file, force_plot)
                        
                        # 在 Streamlit 中显示 HTML
                        with open(html_file) as f:
                            components.html(f.read(), height=500, scrolling=True)
                        
                        st.success('SHAP analysis completed!')
                else:
                    st.info("Please select a compound to view its SHAP explanation.")
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file containing molecular descriptors from ChemDes.")
