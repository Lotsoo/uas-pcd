import streamlit as st
import os
import cv2
import numpy as np
import processing
import classifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car vs Bike Classification", layout="wide")

# --- Initialize Session State ---
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'accuracy' not in st.session_state:
    st.session_state['accuracy'] = 0.0
if 'confusion_matrix' not in st.session_state:
    st.session_state['confusion_matrix'] = None

# --- Sidebar Configuration ---
st.sidebar.title("Configuration")
dataset_path = st.sidebar.text_input("Dataset Path", value=os.path.join(os.getcwd(), "dataset"))
feature_method = st.sidebar.selectbox("Feature Extraction", ["Color Histogram", "Edge Features", "HOG Features"])
algo_choice = st.sidebar.selectbox("Classification Algorithm", ["KNN", "SVM"])
k_neighbors = 3
if algo_choice == "KNN":
    k_neighbors = st.sidebar.slider("K Neighbors", 1, 15, 3)

# --- Helper Function to Load Dataset ---
@st.cache_data
def load_dataset_features(path, method):
    classes = os.listdir(path)
    X = []
    y = []
    class_names = []
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    total_files = sum([len(files) for r, d, files in os.walk(path)])
    processed = 0
    
    for label in classes:
        class_dir = os.path.join(path, label)
        if not os.path.isdir(class_dir):
            continue
            
        class_names.append(label)
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, file)
                img = processing.load_image(img_path)
                
                if img is not None:
                    if method == "Color Histogram":
                        feat = processing.extract_color_histogram(img)
                    elif method == "Edge Features":
                        feat = processing.extract_edge_features(img)
                    elif method == "HOG Features":
                        feat = processing.extract_hog_features(img)
                    else:
                        feat = processing.flatten_image(img)
                    
                    X.append(feat)
                    y.append(label)
            
            processed += 1
            progress_bar.progress(min(processed / max(total_files, 1), 1.0))
            status_text.text(f"Processing {label}: {file}")
            
    status_text.empty()
    progress_bar.empty()
    
    return np.array(X), np.array(y), class_names

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["📂 Dataset", "⚙️ Training", "🖼️ Testing"])

with tab1:
    st.header("Dataset Overview")
    if os.path.exists(dataset_path):
        classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        st.write(f"Classes found: **{classes}**")
        
        cols = st.columns(len(classes))
        for idx, cls in enumerate(classes):
            with cols[idx]:
                st.subheader(cls)
                class_dir = os.path.join(dataset_path, cls)
                files = os.listdir(class_dir)
                st.write(f"Count: {len(files)} images")
                if files:
                    sample_img = processing.load_image(os.path.join(class_dir, files[0]))
                    st.image(sample_img, channels="BGR", caption=f"Sample {cls}")
    else:
        st.error(f"Dataset path not found: {dataset_path}")

with tab2:
    st.header("Model Training")
    st.write(f"**Settings:** {algo_choice} | {feature_method}")
    
    if st.button("Train Model"):
        # Load Data
        with st.spinner("Loading and extracting features..."):
            X, y, class_names = load_dataset_features(dataset_path, feature_method)
        
        st.success(f"Loaded {len(X)} images.")
        
        # Train
        with st.spinner("Training model..."):
            model, acc, cm, test_data = classifier.train_model(X, y, algo_choice, k_neighbors)
            st.session_state['model'] = model
            st.session_state['accuracy'] = acc
            st.session_state['confusion_matrix'] = cm
            st.session_state['classes'] = class_names
                    
    if st.session_state['model'] is not None:
        st.info(f"Model Accuration: **{st.session_state['accuracy']*100:.2f}%**")
        
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = st.session_state['confusion_matrix']
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        
        # Add labels
        classes = st.session_state.get('classes', ['Class 0', 'Class 1'])
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
            
        st.pyplot(fig)

with tab3:
    st.header("Testing New Image")
    
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Save temp file
        with open("temp_test.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            if st.session_state['model'] is None:
                st.warning("Please train the model first in the 'Training' tab.")
            else:
                img = processing.load_image("temp_test.jpg")
                if img is not None:
                    # Extract features
                    if feature_method == "Color Histogram":
                        feat = processing.extract_color_histogram(img)
                    elif feature_method == "Edge Features":
                        feat = processing.extract_edge_features(img)
                    elif feature_method == "HOG Features":
                        feat = processing.extract_hog_features(img)
                    else:
                        feat = processing.flatten_image(img)
                    
                    # Predict
                    pred_class, probs = classifier.predict_image(st.session_state['model'], feat)
                    
                    st.metric(label="Prediction", value=pred_class)
                    if probs is not None:
                        st.subheader("Confidence Score")
                        classes = st.session_state.get('classes', [])
                        # Probs is a list of lists [[p1, p2]], take first one
                        proba = probs[0]
                        
                        for idx, p in enumerate(proba):
                            label = classes[idx] if idx < len(classes) else f"Class {idx}"
                            st.progress(modfied_p := float(p), text=f"{label}: {modfied_p*100:.2f}%")
