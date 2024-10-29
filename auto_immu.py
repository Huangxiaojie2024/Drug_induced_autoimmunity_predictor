import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

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
        div[data-testid="stMetricValue"] {
            font-size: 24px;
            color: #0d6efd;
        }
        .metric-container {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        .shap-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
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
    uploaded_file = st.file_uploader("Upload RDKit descriptors CSV", type=['csv'])
    
    if uploaded_file:
        st.success(f"File uploaded successfully!")
        
    st.markdown("---")
    st.markdown("""
        ### 📖 Instructions
        1. Visit http://www.scbdd.com/rdk_desc/index/ to calculate and download 196 RDKit molecular descriptors for your compounds
        2. The model will automatically select and use the 65 optimal descriptors for prediction
        3. Upload the descriptors file using the button above
        4. View predictions and analysis in the main panel
    """)
    
    st.markdown("---")
    st.markdown("""
        ### 🎯 Model Information
        - **Algorithm**: Easy Ensemble Classifier
        - **Input Features**: 65 selected RDKit molecular descriptors
        - **Output**: Binary classification (DIA positive/negative)
        - **Data Source**: RDKit descriptors calculated from http://www.scbdd.com/rdk_desc/index/
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
                "Prediction": ["DIA Positive" if p > 0.5 else "DIA Negative" for p in predictions_prob[:, 1]],
                "Risk_Level": pd.cut(predictions_prob[:, 1], 
                                   bins=[0, 0.2, 0.5, 0.8, 1.0],
                                   labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
            })
            
            # 显示关键指标
            st.subheader("📊 Prediction Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                positive_count = sum(predictions_prob[:, 1] > 0.5)
                st.metric("DIA Positive", 
                         f"{positive_count}",
                         f"{positive_count/len(df)*100:.1f}%")
            
            with col2:
                high_risk = sum(predictions_prob[:, 1] > 0.8)
                st.metric("High Risk",
                         f"{high_risk}",
                         f"{high_risk/len(df)*100:.1f}%")
            
            with col3:
                med_risk = sum((predictions_prob[:, 1] > 0.5) & (predictions_prob[:, 1] <= 0.8))
                st.metric("Medium Risk",
                         f"{med_risk}",
                         f"{med_risk/len(df)*100:.1f}%")
            
            with col4:
                avg_prob = np.mean(predictions_prob[:, 1])
                st.metric("Avg. Probability",
                         f"{avg_prob:.3f}",
                         f"±{np.std(predictions_prob[:, 1]):.3f}")

            # 风险分布可视化
            st.subheader("📈 Risk Distribution Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # 饼图显示预测结果分布
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['DIA Positive', 'DIA Negative'],
                    values=[positive_count, len(df) - positive_count],
                    hole=.3,
                    marker_colors=['#ff6b6b', '#4ecdc4']
                )])
                fig_pie.update_layout(
                    title="Prediction Distribution",
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # 风险等级条形图
                risk_counts = results_df['Risk_Level'].value_counts().sort_index()
                fig_risk = go.Figure(data=[go.Bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    marker_color=['#4ecdc4', '#ffe66d', '#ff9f1c', '#ff6b6b']
                )])
                fig_risk.update_layout(
                    title="Risk Level Distribution",
                    xaxis_title="Risk Level",
                    yaxis_title="Number of Compounds",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_risk, use_container_width=True)

            # 显示详细结果表格
            st.subheader("📋 Detailed Results")
            st.dataframe(results_df.style.background_gradient(
                subset=['DIA_positive_prob'],
                cmap='RdYlBu_r'
            ))
            
            # 下载结果
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Results",
                data=csv,
                file_name="dia_predictions.csv",
                mime="text/csv"
            )
            
            # SHAP Analysis
            st.markdown("---")
            st.subheader("🔍 SHAP Analysis")
            
            # 选择化合物
            selected_compound = st.selectbox(
                "Select a compound for detailed analysis:",
                range(len(df)),
                format_func=lambda x: (
                    f"Compound {x+1} "
                    f"({'🔴' if predictions_prob[x,1] > 0.5 else '🟢'}) "
                    f"Risk: {results_df.iloc[x]['Risk_Level']} "
                    f"(P={predictions_prob[x,1]:.3f})"
                )
            )

            if selected_compound is not None:
                st.session_state.selected_compound = selected_compound
                
                # 显示所选化合物的预测详情
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Prediction", 
                        "DIA Positive" if predictions_prob[selected_compound,1] > 0.5 else "DIA Negative",
                        f"P={predictions_prob[selected_compound,1]:.3f}"
                    )
                with col2:
                    st.metric(
                        "Risk Level",
                        results_df.iloc[selected_compound]['Risk_Level']
                    )
                with col3:
                    confidence = abs(predictions_prob[selected_compound,1] - 0.5) * 2
                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.1%}"
                    )
                
                # SHAP Analysis
                with st.spinner('Analyzing molecular features...'):
                    # 创建SHAP解释器
                    explainer = shap.KernelExplainer(
                        best_estimator_eec.predict_proba,
                        Xtrain_std
                    )
                    
                    # 计算SHAP值
                    shap_values = explainer.shap_values(
                        X_std[selected_compound:selected_compound+1],
                        nsamples=150
                    )
                    
                    # 特征重要性分析
                    st.markdown("### Top Contributing Features")
                    feature_importance = pd.DataFrame({
                        'Feature': descriptor_names,
                        'Importance': np.abs(shap_values[0][0,:,1])
                    }).sort_values('Importance', ascending=True).tail(10)
                    
                    fig = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Most Influential Molecular Descriptors'
                    )
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # SHAP力图
                    st.markdown("### SHAP Force Plot")
                    force_plot = shap.force_plot(
                        explainer.expected_value[1],
                        shap_values[0,:,1],
                        X[selected_compound],
                        feature_names=descriptor_names,
                        show=False
                    )
                    
                    html_file = f"shap_force_plot_{selected_compound}.html"
                    shap.save_html(html_file, force_plot)
                    
                    with open(html_file) as f:
                        components.html(f.read(), height=300, scrolling=True)
                    
                    st.success('Analysis completed successfully!')

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    # 显示欢迎信息
    st.info("👆 Please upload your RDKit descriptors CSV file to begin the analysis.")
    
    # 添加示例说明
    st.markdown("""
        ### 📝 Data Requirements
        1. Calculate RDKit descriptors from http://www.scbdd.com/rdk_desc/index/
        2. Your CSV file should contain all 196 RDKit molecular descriptors
        3. The model will automatically select the 65 optimal descriptors
        
        ### 🎯 Key Features
        - Advanced Easy Ensemble Classifier for DIA prediction
        - Comprehensive molecular descriptor analysis
        - Interactive SHAP explanations
        - Risk level assessment
        - Detailed feature importance analysis
    """)
