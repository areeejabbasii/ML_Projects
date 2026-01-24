import streamlit as st
import pandas as pd
import numpy as np
from model import Model
import pickle
import os

# Set page config
st.set_page_config(page_title="Instagram Influencer Analytics", layout="wide")

st.title("📊 Instagram Influencer Analytics Dashboard")
st.sidebar.header("Configuration")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar options
menu_option = st.sidebar.radio("Select Option", ["Train Model", "Influencer Details", "View Data", "Make Predictions", "Model Evaluation"])

# Data path
data_path = "top_insta_influencers_data.csv"
model_path = "trained_model.pkl"

if menu_option == "Train Model":
    st.header("Train the Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Data", key="load_data"):
            try:
                model = Model(data_path)
                model.load_data()
                st.session_state.model = model
                st.session_state.data = model.data
                st.success("✅ Data loaded successfully!")
                st.dataframe(model.data.head())
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    with col2:
        target_column = st.selectbox("Select Target Column", options=st.session_state.data.columns.tolist() if st.session_state.data is not None else [])
    
    if st.button("Preprocess Data", key="preprocess"):
        if st.session_state.model is not None:
            try:
                st.session_state.model.preprocess_data()
                st.success("✅ Data preprocessed successfully!")
            except Exception as e:
                st.error(f"Error preprocessing data: {e}")
    
    if st.button("Split Data", key="split"):
        if st.session_state.model is not None and target_column:
            try:
                st.session_state.model.split_data(target_column)
                st.success("✅ Data split successfully!")
                st.info(f"Training set size: {len(st.session_state.model.X_train)}")
                st.info(f"Test set size: {len(st.session_state.model.X_test)}")
            except Exception as e:
                st.error(f"Error splitting data: {e}")
    
    if st.button("Train Model", key="train"):
        if st.session_state.model is not None:
            try:
                with st.spinner("Training model..."):
                    st.session_state.model.train_model()
                st.success("✅ Model trained successfully!")
                
                # Save model
                with open(model_path, 'wb') as f:
                    pickle.dump(st.session_state.model, f)
                st.success(f"Model saved as '{model_path}'")
            except Exception as e:
                st.error(f"Error training model: {e}")

elif menu_option == "View Data":
    st.header("Data Overview")
    
    try:
        data = pd.read_csv(data_path)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Total Features", len(data.columns))
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        st.subheader("Dataset Preview")
        st.dataframe(data, use_container_width=True)
        
        st.subheader("Data Statistics")
        st.dataframe(data.describe(), use_container_width=True)
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes,
            'Non-Null Count': data.notna().sum(),
            'Null Count': data.isnull().sum()
        })
        st.dataframe(col_info, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")

elif menu_option == "Influencer Details":
    st.header("📸 Influencer Details")
    
    try:
        data = pd.read_csv(data_path)
        
        # Select influencer
        influencer = st.selectbox("Select Influencer", options=data['channel_info'].tolist())
        
        # Get influencer data
        influencer_data = data[data['channel_info'] == influencer].iloc[0]
        
        # Select what details to view
        st.subheader("Select Details to View")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_followers = st.checkbox("👥 Followers", value=True)
        with col2:
            show_likes = st.checkbox("❤️ Likes", value=True)
        with col3:
            show_posts = st.checkbox("📸 Posts", value=True)
        
        # Display selected metrics
        st.subheader(f"Details for {influencer}")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        if show_followers:
            with metric_col1:
                st.metric("👥 Followers", influencer_data.get('followers', 'N/A'))
        
        if show_likes:
            with metric_col2:
                st.metric("❤️ Avg Likes", influencer_data.get('avg_likes', 'N/A'))
        
        if show_posts:
            with metric_col3:
                st.metric("📸 Posts", influencer_data.get('posts', 'N/A'))
        
        # Display additional info
        st.divider()
        st.subheader("📊 Full Influencer Profile")
        
        profile_data = pd.DataFrame({
            'Field': data.columns,
            'Value': influencer_data.values
        })
        st.dataframe(profile_data, use_container_width=True)
        
        # Display influencer image/profile
        st.divider()
        st.subheader("📷 Influencer Profile")
        
        col_img, col_info = st.columns(2)
        
        with col_img:
            # Display Instagram link
            instagram_url = f"https://instagram.com/{influencer}"
            st.markdown(f"""
            ### {influencer}
            
            **Click to visit Instagram Profile:**
            
            [@{influencer}]({instagram_url})
            
            ---
            
            **Key Stats:**
            - **Followers:** {influencer_data.get('followers', 'N/A')}
            - **Posts:** {influencer_data.get('posts', 'N/A')}
            - **Avg Likes per Post:** {influencer_data.get('avg_likes', 'N/A')}
            - **Influence Score:** {influencer_data.get('influence_score', 'N/A')}
            - **Engagement Rate:** {influencer_data.get('60_day_eng_rate', 'N/A')}
            - **Country:** {influencer_data.get('country', 'N/A')}
            """)
        
        with col_info:
            # Try to load Instagram profile image
            st.info(f"""
            **Profile Information**
            
            Visit the link above to see:
            - Profile picture
            - Recent posts
            - Follower engagement
            - Full profile details
            """)
        
    except Exception as e:
        st.error(f"Error loading influencer details: {e}")

elif menu_option == "Make Predictions":
    st.header("Make Predictions")
    
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            data = pd.read_csv(data_path)
            
            st.info("Select an influencer from the dataset to make a prediction")
            
            influencer = st.selectbox("Select Influencer", options=data.get('channel_info', data.iloc[:, 0]).tolist())
            
            if st.button("Get Prediction"):
                # Get the row for this influencer
                if 'channel_info' in data.columns:
                    row_idx = data[data['channel_info'] == influencer].index[0]
                else:
                    row_idx = data[data.iloc[:, 0] == influencer].index[0]
                
                st.success(f"Selected Influencer: {influencer}")
                st.dataframe(data.iloc[[row_idx]], use_container_width=True)
        else:
            st.warning("No trained model found. Please train the model first!")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

elif menu_option == "Model Evaluation":
    st.header("Model Evaluation")
    
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            if hasattr(model, 'X_test') and hasattr(model, 'y_test'):
                st.info("Model Performance Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Evaluate Model"):
                        with st.spinner("Evaluating model..."):
                            try:
                                model.evaluate_model()
                                st.success("✅ Evaluation complete!")
                            except Exception as e:
                                st.error(f"Error: {e}")
                
                # Display feature importance
                if hasattr(model.model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    feature_importance = pd.DataFrame({
                        'Feature': model.X_train.columns,
                        'Importance': model.model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.bar_chart(feature_importance.set_index('Feature'))
            else:
                st.warning("Model has not been trained yet!")
        else:
            st.warning("No trained model found. Please train the model first!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
