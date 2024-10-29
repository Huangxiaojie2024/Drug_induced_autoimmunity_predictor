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

# [前面的CSS样式代码保持不变...]

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

# 主体内容修改部分
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # [数据处理代码保持不变...]
        
        # 替换原来的Prediction Distribution
        st.subheader("📊 Prediction Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            # 创建饼图显示正负样本分布
            pos_count = sum(predictions_prob[:, 1] > 0.5)
            neg_count = len(df) - pos_count
            fig_pie = go.Figure(data=[go.Pie(
                labels=['DIA Positive', 'DIA Negative'],
                values=[pos_count, neg_count],
                hole=.3,
                marker_colors=['#ff6b6b', '#4ecdc4']
            )])
            fig_pie.update_layout(title="Distribution of Predictions")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            # 创建风险等级分布
            risk_levels = pd.cut(predictions_prob[:, 1], 
                               bins=[0, 0.2, 0.5, 0.8, 1.0],
                               labels=['Low Risk', 'Medium-Low Risk', 'Medium-High Risk', 'High Risk'])
            risk_counts = risk_levels.value_counts()
            fig_bar = go.Figure(data=[go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker_color=['#4ecdc4', '#ffe66d', '#ff9f1c', '#ff6b6b']
            )])
            fig_bar.update_layout(title="Risk Level Distribution",
                                xaxis_title="Risk Level",
                                yaxis_title="Number of Compounds")
            st.plotly_chart(fig_bar, use_container_width=True)

        # SHAP Analysis - 修改布局
        st.subheader("🔍 SHAP Analysis")
        
        # 选择化合物
        selected_compound = st.selectbox(
            "Choose a compound to analyze:",
            range(len(df)),
            format_func=lambda x: f"Compound {x+1} ({'🔴' if predictions_prob[x,1] > 0.5 else '🟢'}) P={predictions_prob[x,1]:.2f}"
        )
        
        if selected_compound is not None:
            st.session_state.selected_compound = selected_compound
            
            # 显示所选化合物的预测详情
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", 
                         "DIA Positive" if predictions_prob[selected_compound,1] > 0.5 else "DIA Negative",
                         f"P={predictions_prob[selected_compound,1]:.3f}")
            with col2:
                risk_level = "High" if predictions_prob[selected_compound,1] > 0.8 else \
                            "Medium" if predictions_prob[selected_compound,1] > 0.5 else "Low"
                st.metric("Risk Level", risk_level)
            with col3:
                confidence = abs(predictions_prob[selected_compound,1] - 0.5) * 2
                st.metric("Prediction Confidence", f"{confidence:.1%}")
            
            # SHAP Analysis
            with st.spinner('Analyzing molecular features...'):
                explainer = shap.KernelExplainer(best_estimator_eec.predict_proba, Xtrain_std)
                shap_values = explainer.shap_values(X_std[selected_compound:selected_compound+1], nsamples=150)
                
                # SHAP Force Plot
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
                
                # 特征重要性 - 修改排序方向
                st.markdown("### Feature Importance Analysis")
                feature_importance = pd.DataFrame({
                    'Feature': descriptor_names,
                    'Importance': np.abs(shap_values[0,:,1])
                }).sort_values('Importance', ascending=True).tail(10)  # 改为ascending=True并使用tail
                
                fig = px.bar(feature_importance, 
                           x='Importance', 
                           y='Feature',
                           orientation='h',
                           title='Top 10 Most Influential Molecular Descriptors')
                fig.update_layout(yaxis={'categoryorder':'total ascending'},  # 确保最重要的特征在顶部
                                height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.success('Analysis completed successfully!')

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    # [欢迎信息代码保持不变...]
    pass
