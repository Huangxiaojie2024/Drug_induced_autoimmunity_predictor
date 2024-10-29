import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# 设置页面配置
st.set_page_config(
    page_title="Drug-induced Autoimmunity (DIA) Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            width: 100%;
            background-color: #0d6efd;
            color: white;
            border-radius: 5px;
            padding: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .stProgress .st-bo {
            background-color: #0d6efd;
        }
        .plot-container {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #0d6efd;
            text-align: center;
            padding: 1rem;
        }
        h2 {
            color: #0d6efd;
            padding: 0.5rem 0;
        }
        .stAlert {
            background-color: #e7f3fe;
            border-left-color: #0d6efd;
        }
    </style>
""", unsafe_allow_html=True)

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

# 页面标题和介绍
st.title("🔬 Drug-induced Autoimmunity (DIA) Predictor")
st.markdown("""
    <div style='background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <p style='font-size: 1.1em; color: #666;'>
            Welcome to the DIA Predictor, an advanced machine learning-based tool for predicting drug-induced autoimmunity. 
            This tool analyzes molecular descriptors to assess the potential risk of drugs causing autoimmune responses.
        </p>
    </div>
""", unsafe_allow_html=True)

# 侧边栏设计
with st.sidebar:
    st.header("📊 Data Input")
    uploaded_file = st.file_uploader("Upload ChemDes descriptors CSV", type=['csv'])
    
    if uploaded_file:
        st.success(f"File uploaded successfully!")
        
    st.markdown("---")
    st.markdown("""
        ### 📖 Instructions
        1. Prepare your CSV file with molecular descriptors
        2. Upload the file using the button above
        3. View predictions and analysis in the main panel
        4. Explore individual compounds using SHAP analysis
    """)
    
    st.markdown("---")
    st.markdown("""
        ### 🎯 Model Information
        - **Algorithm**: Ensemble Classifier
        - **Features**: 65 molecular descriptors
        - **Output**: Binary classification (DIA positive/negative)
    """)

# 主要内容
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # 数据验证
        missing_descriptors = [desc for desc in descriptor_names if desc not in df.columns]
        if missing_descriptors:
            st.error(f"Missing required descriptors: {', '.join(missing_descriptors)}")
        else:
            # 数据处理和预测
            X = df[descriptor_names].values
            X_std = scaler.transform(X)
            predictions_prob = best_estimator_eec.predict_proba(X_std)
            
            # 创建结果DataFrame
            results_df = pd.DataFrame({
                "Compound_ID": range(1, len(df) + 1),
                "DIA_negative_prob": predictions_prob[:, 0],
                "DIA_positive_prob": predictions_prob[:, 1],
                "Prediction": ["DIA Positive" if p > 0.5 else "DIA Negative" for p in predictions_prob[:, 1]]
            })
            
            # 创建三列布局用于显示关键指标
            col1, col2, col3 = st.columns(3)
            
            with col1:
                positive_count = sum(predictions_prob[:, 1] > 0.5)
                st.metric("DIA Positive Compounds", positive_count, 
                         f"{positive_count/len(df)*100:.1f}% of total")
                
            with col2:
                high_risk = sum(predictions_prob[:, 1] > 0.8)
                st.metric("High Risk Compounds (>80%)", high_risk,
                         f"{high_risk/len(df)*100:.1f}% of total")
                
            with col3:
                avg_prob = np.mean(predictions_prob[:, 1])
                st.metric("Average DIA Probability", f"{avg_prob:.2f}",
                         f"±{np.std(predictions_prob[:, 1]):.2f} SD")

            # 预测结果可视化
            st.subheader("📈 Prediction Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=predictions_prob[:, 1], 
                                     nbinsx=30,
                                     name="DIA Probability Distribution",
                                     marker_color='#0d6efd'))
            fig.update_layout(
                title="Distribution of DIA Probabilities",
                xaxis_title="Probability of DIA",
                yaxis_title="Number of Compounds",
                template="plotly_white",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # 显示详细结果表格
            st.subheader("📋 Detailed Results")
            st.dataframe(results_df.style.background_gradient(subset=['DIA_positive_prob'], cmap='RdYlBu_r'))
            
            # 下载按钮
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="dia_predictions.csv",
                mime="text/csv"
            )
            
            # SHAP Analysis
            st.subheader("🔍 SHAP Analysis")
            
            # 创建两列布局
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Select Compound")
                # 创建一个更美观的选择界面
                compounds_df = pd.DataFrame({
                    "ID": range(1, len(df) + 1),
                    "Prediction": ["🔴" if p > 0.5 else "🟢" for p in predictions_prob[:, 1]],
                    "Probability": [f"{p:.2f}" for p in predictions_prob[:, 1]]
                })
                
                selected_compound = st.selectbox(
                    "Choose a compound to analyze:",
                    range(len(df)),
                    format_func=lambda x: f"Compound {x+1} {compounds_df.loc[x, 'Prediction']} (P={compounds_df.loc[x, 'Probability']})"
                )
                
                if selected_compound is not None:
                    st.session_state.selected_compound = selected_compound

            with col2:
                if 'selected_compound' in st.session_state:
                    compound_index = st.session_state.selected_compound
                    
                    # 创建SHAP解释器
                    with st.spinner('Analyzing molecular features...'):
                        explainer = shap.KernelExplainer(best_estimator_eec.predict_proba, Xtrain_std)
                        shap_values = explainer.shap_values(X_std[compound_index:compound_index+1], nsamples=150)
                        
                        force_plot = shap.force_plot(
                            explainer.expected_value[1], 
                            shap_values[0,:,1],
                            X[compound_index],
                            feature_names=descriptor_names,
                            show=False
                        )
                        
                        html_file = f"shap_force_plot_{compound_index}.html"
                        shap.save_html(html_file, force_plot)
                        
                        with open(html_file) as f:
                            components.html(f.read(), height=500, scrolling=True)
                        
                        # 添加特征重要性条形图
                        st.markdown("### Top Contributing Features")
                        feature_importance = pd.DataFrame({
                            'Feature': descriptor_names,
                            'Importance': np.abs(shap_values[0][0,:,1])
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        fig = px.bar(feature_importance, 
                                   x='Importance', 
                                   y='Feature',
                                   orientation='h',
                                   title='Top 10 Most Influential Molecular Descriptors')
                        fig.update_layout(template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.success('Analysis completed successfully!')

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    # 显示欢迎信息和示例
    st.info("👆 Please upload your ChemDes descriptors CSV file to begin the analysis.")
    
    # 添加示例数据的展示
    st.markdown("""
        ### 📝 Example Data Format
        Your CSV file should contain the following 65 molecular descriptors:
        ```
        BalabanJ, Chi0, EState_VSA1, ... [and other descriptors]
        ```
        
        ### 🎯 Key Features
        - Advanced machine learning model for DIA prediction
        - Comprehensive molecular descriptor analysis
        - Interactive SHAP explanations
        - Detailed statistical analysis
        - Beautiful data visualizations
    """)
