import streamlit as st
import cv2
import os
import uuid
from roboflow import Roboflow
from PIL import Image, ExifTags
import pandas as pd
import numpy as np
from datetime import datetime
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Complete Earring Detection & Counting",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Roboflow client
@st.cache_resource
def get_roboflow_client():
    rf = Roboflow(api_key="0lfsVUdCmiTvLOVlDejI")
    return rf

# Create directories
def create_directories():
    directories = [
        'complete_detection_results',
        'complete_detection_results/front_cropped',
        'complete_detection_results/top_cropped',
        'complete_detection_results/processed_images',
        'complete_detection_results/excel_reports'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return directories

def fix_image_orientation(image_path):
    """Fix image orientation based on EXIF data"""
    try:
        with Image.open(image_path) as pil_image:
            exif = pil_image._getexif()
            
            if exif is not None:
                for tag, value in exif.items():
                    if ExifTags.TAGS.get(tag) == 'Orientation':
                        orientation = value
                        break
                else:
                    orientation = 1
            else:
                orientation = 1
            
            if orientation == 3:
                pil_image = pil_image.rotate(180, expand=True)
            elif orientation == 6:
                pil_image = pil_image.rotate(270, expand=True)
            elif orientation == 8:
                pil_image = pil_image.rotate(90, expand=True)
            
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return opencv_image
            
    except Exception as e:
        st.warning(f"Could not read EXIF data: {e}. Using original image.")
        return cv2.imread(image_path)

def draw_label(img, text, x, y, color=(0, 255, 0)):
    """Draw label with background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    cv2.rectangle(img, (x, y - 25), (x + text_size[0] + 10, y + 5), color, -1)
    cv2.putText(img, text, (x + 5, y - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def group_detections_by_rows(predictions, y_threshold=100):
    """Group detections into rows by Y proximity and sort each row right to left."""
    rows = []
    predictions.sort(key=lambda p: p['y'])  # Top to bottom

    for pred in predictions:
        added = False
        for row in rows:
            avg_y = sum(p['y'] for p in row) / len(row)
            if abs(pred['y'] - avg_y) <= y_threshold:
                row.append(pred)
                added = True
                break
        if not added:
            rows.append([pred])

    # Sort each row right to left (x descending)
    for row in rows:
        row.sort(key=lambda p: p['x'], reverse=True)

    # Flatten the list row by row
    return [pred for row in rows for pred in row]

def process_front_image(file, confidence=0.50):
    """Process front image to detect and classify earrings"""
    rf = get_roboflow_client()
    
    # Save uploaded file
    original_filename = file.name
    save_name = str(uuid.uuid4()) + "_" + original_filename
    save_path = os.path.join('complete_detection_results', save_name)
    
    with open(save_path, "wb") as f:
        f.write(file.getbuffer())
    
    # Fix orientation
    img = fix_image_orientation(save_path)
    height, width = img.shape[:2]
    
    # Save corrected image
    corrected_path = save_path.replace('.jpg', '_corrected.jpg').replace('.jpeg', '_corrected.jpg').replace('.png', '_corrected.png')
    cv2.imwrite(corrected_path, img)
    
    # Get detection model
    detection_model = rf.workspace().project("earring-crop-ugfno").version(1).model
    
    # Run detection
    st.info("🔍 Detecting earrings in front image...")
    detection_result = detection_model.predict(corrected_path, confidence=int(confidence * 100)).json()
    
    predictions = []
    id_counter = 1
    min_conf_threshold = 0.50
    
    if detection_result.get('predictions'):
        sorted_detections = group_detections_by_rows(detection_result['predictions'], y_threshold=100)
        
        for pred in sorted_detections:
            x, y = int(pred['x']), int(pred['y'])
            w, h = int(pred['width']), int(pred['height'])
            start_x, start_y = max(x - w // 2, 0), max(y - h // 2, 0)
            end_x, end_y = min(x + w // 2, width), min(y + h // 2, height)

            # Draw bounding box
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            draw_label(img, f"Earring-ID-{id_counter}", start_x, start_y)

            # Crop and classify
            crop = img[start_y:end_y, start_x:end_x]
            crop_filename = f"front_crop_{id_counter}_{uuid.uuid4().hex[:8]}.jpg"
            crop_path = os.path.join('complete_detection_results/front_cropped', crop_filename)
            cv2.imwrite(crop_path, crop)

            # Use dual models for classification
            class_model_1 = rf.workspace().project("earring-id-new").version(5).model
            class_model_2 = rf.workspace().project("earring-id-detect").version(2).model
            
            # Run both models
            classification_result_1 = class_model_1.predict(crop_path, confidence=30).json()
            classification_result_2 = class_model_2.predict(crop_path, confidence=30).json()
            
            # Find best prediction
            best_prediction = None
            best_confidence = 0
            best_model_id = ""
            
            if classification_result_1.get('predictions'):
                for pred in classification_result_1['predictions']:
                    if pred.get('confidence', 0) > best_confidence:
                        best_prediction = pred
                        best_confidence = pred.get('confidence', 0)
                        best_model_id = "earring-id-new-v5"
            
            if classification_result_2.get('predictions'):
                for pred in classification_result_2['predictions']:
                    if pred.get('confidence', 0) > best_confidence:
                        best_prediction = pred
                        best_confidence = pred.get('confidence', 0)
                        best_model_id = "earring-id-detect-v2"

            if best_prediction and best_confidence >= min_conf_threshold:
                predictions.append({
                    'id': f'Earring-ID-{id_counter}',
                    'class': best_prediction['class'],
                    'confidence': best_confidence,
                    'model_used': best_model_id,
                    'image_path': crop_path,
                    'crop_filename': crop_filename
                })
            else:
                predictions.append({
                    'id': f'Earring-ID-{id_counter}',
                    'class': 'Unknown',
                    'confidence': best_confidence if best_prediction else 0.0,
                    'model_used': best_model_id if best_prediction else 'No Model',
                    'image_path': crop_path,
                    'crop_filename': crop_filename
                })

            id_counter += 1

    # Save processed image
    processed_filename = f"front_processed_{uuid.uuid4().hex[:8]}.jpg"
    processed_path = os.path.join('complete_detection_results/processed_images', processed_filename)
    cv2.imwrite(processed_path, img)
    
    # Clean up
    if os.path.exists(corrected_path):
        os.remove(corrected_path)
    if os.path.exists(save_path):
        os.remove(save_path)
    
    return {
        'original_filename': original_filename,
        'processed_image_path': processed_path,
        'predictions': predictions,
        'total_detections': len(predictions)
    }

def process_top_image(file, confidence=0.50):
    """Process top image to count items in pipes"""
    rf = get_roboflow_client()
    
    # Save uploaded file
    original_filename = file.name
    save_name = str(uuid.uuid4()) + "_" + original_filename
    save_path = os.path.join('complete_detection_results', save_name)
    
    with open(save_path, "wb") as f:
        f.write(file.getbuffer())
    
    # Fix orientation
    img = fix_image_orientation(save_path)
    height, width = img.shape[:2]
    
    # Save corrected image
    corrected_path = save_path.replace('.jpg', '_corrected.jpg').replace('.jpeg', '_corrected.jpg').replace('.png', '_corrected.png')
    cv2.imwrite(corrected_path, img)
    
    # Get crop model
    crop_model = rf.workspace().project("earring-crop-ugfno").version(1).model
    
    # Run crop detection
    st.info("🔍 Detecting pipe regions in top image...")
    crop_result = crop_model.predict(corrected_path, confidence=int(confidence * 100)).json()
    
    id_counter = 1
    pipe_count = []
    
    if crop_result.get('predictions'):
        sorted_detections = group_detections_by_rows(crop_result['predictions'], y_threshold=100)
        
        for pred in sorted_detections:
            x, y = int(pred['x']), int(pred['y'])
            w, h = int(pred['width']), int(pred['height'])
            start_x, start_y = max(x - w // 2, 0), max(y - h // 2, 0)
            end_x, end_y = min(x + w // 2, width), min(y + h // 2, height)

            draw_label(img, f"Pipe-ID-{id_counter}", start_x, start_y)

            # Save cropped images
            crop = img[start_y:end_y, start_x:end_x]
            crop_filename = f"top_crop_{id_counter}_{uuid.uuid4().hex[:8]}.jpg"
            crop_path = os.path.join('complete_detection_results/top_cropped', crop_filename)
            cv2.imwrite(crop_path, crop)
            
            # Count items in pipe
            count_model = rf.workspace().project("earring-box-count-0cf1h").version(1).model
            count_result = count_model.predict(crop_path, confidence=30).json()
            
            if count_result.get('predictions'):
                for i, pred in enumerate(count_result['predictions']):
                    x, y = int(pred['x']), int(pred['y'])
                    w, h = int(pred['width']), int(pred['height'])
                    start_x, start_y = max(x - w // 2, 0), max(y - h // 2, 0)
                    end_x, end_y = min(x + w // 2, width), min(y + h // 2, height)
                    
                    cv2.rectangle(crop, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    draw_label(crop, f"item", start_x, start_y, (0, 255, 0))
                
                # Update the cropped image with bounding boxes
                cv2.imwrite(crop_path, crop)

            pipe_count.append({
                'pipe_id': f'Pipe-{id_counter}',
                'item_count': len(count_result.get('predictions', [])),
                'image_path': crop_path,
                'crop_filename': crop_filename
            })

            id_counter += 1

    # Save processed image
    processed_filename = f"top_processed_{uuid.uuid4().hex[:8]}.jpg"
    processed_path = os.path.join('complete_detection_results/processed_images', processed_filename)
    cv2.imwrite(processed_path, img)
    
    # Clean up
    if os.path.exists(corrected_path):
        os.remove(corrected_path)
    if os.path.exists(save_path):
        os.remove(save_path)
    
    return {
        'original_filename': original_filename,
        'processed_image_path': processed_path,
        'pipe_groups': pipe_count,
        'total_count': len(pipe_count)
    }

def map_ids(front_predictions, top_pipes, stand_name):
    """Map front detections to top counts"""
    mapped_results = []
    
    for i, front_pred in enumerate(front_predictions):
        if i < len(top_pipes):
            mapped_results.append({
                'mapped_id': front_pred['id'],
                'detected_class': front_pred['class'],
                'confidence': front_pred['confidence'],
                'model_used': front_pred['model_used'],
                'count': top_pipes[i]['item_count'],
                'stand_name': stand_name,
                'front_image': front_pred['crop_filename'],
                'top_image': top_pipes[i]['crop_filename']
            })
        else:
            mapped_results.append({
                'mapped_id': front_pred['id'],
                'detected_class': front_pred['class'],
                'confidence': front_pred['confidence'],
                'model_used': front_pred['model_used'],
                'count': 0,
                'stand_name': stand_name,
                'front_image': front_pred['crop_filename'],
                'top_image': 'N/A'
            })
    
    return mapped_results

def create_excel_report(mapped_results, include_model_info=False):
    """Create Excel report with detection results"""
    df_data = []
    for result in mapped_results:
        row_data = {
            'Mapped ID': result['mapped_id'],
            'Detected Class': result['detected_class'],
            'Confidence': round(result['confidence'], 3),
            'Item Count': result['count'],
            'Stand Name': result['stand_name'],
            'Front Image': result['front_image'],
            'Top Image': result['top_image'],
            'Detection Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if include_model_info:
            row_data['Model Used'] = result.get('model_used', 'Unknown')
        
        df_data.append(row_data)
    
    df = pd.DataFrame(df_data)
    excel_filename = f"complete_earring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    excel_path = os.path.join('complete_detection_results/excel_reports', excel_filename)
    df.to_excel(excel_path, index=False, engine='openpyxl')
    
    return excel_path, df

def main():
    # Header
    st.markdown('<h1 class="main-header">💎 Complete Earring Detection & Counting System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Create necessary directories
    create_directories()
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Configuration")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Minimum confidence for detection"
    )
    
    show_model_info = st.sidebar.checkbox(
        "🔧 Show Model Info (Internal)", 
        value=False,
        help="Show which model was used for each prediction"
    )
    
    st.sidebar.markdown("## 📁 Output Folders")
    st.sidebar.info("""
    - **Front Cropped**: Individual earring crops from front view
    - **Top Cropped**: Pipe crops from top view
    - **Processed Images**: Original images with bounding boxes
    - **Excel Reports**: Complete detection results
    """)
    
    # Main content
    st.markdown('<h2 class="section-header">📤 Upload Images</h2>', unsafe_allow_html=True)
    
    # Upload mode selection
    upload_mode = st.radio(
        "Select upload mode:",
        ["Both Images (Complete Analysis)", "Front Image Only (Detection)", "Top Image Only (Counting)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Upload sections based on mode
    front_file = None
    top_file = None
    
    if upload_mode in ["Both Images (Complete Analysis)", "Front Image Only (Detection)"]:
        st.markdown("### Front Image (Earring Detection)")
        st.markdown("Upload an image showing earrings from the front view for classification.")
        front_file = st.file_uploader(
            "Choose front image",
            type=['png', 'jpg', 'jpeg'],
            key="front_upload"
        )
    
    if upload_mode in ["Both Images (Complete Analysis)", "Top Image Only (Counting)"]:
        st.markdown("### Top Image (Counting)")
        st.markdown("Upload an image showing the stand from the top view for counting items in pipes.")
        top_file = st.file_uploader(
            "Choose top image",
            type=['png', 'jpg', 'jpeg'],
            key="top_upload"
        )
    
    # Process button
    process_text = {
        "Both Images (Complete Analysis)": "🚀 Process Both Images",
        "Front Image Only (Detection)": "🔍 Process Front Image",
        "Top Image Only (Counting)": "📊 Process Top Image"
    }
    
    if st.button(process_text[upload_mode], type="primary", use_container_width=True):
        # Validate uploads based on mode
        if upload_mode == "Both Images (Complete Analysis)" and (not front_file or not top_file):
            st.error("❌ Please upload both front and top images.")
            return
        elif upload_mode == "Front Image Only (Detection)" and not front_file:
            st.error("❌ Please upload a front image.")
            return
        elif upload_mode == "Top Image Only (Counting)" and not top_file:
            st.error("❌ Please upload a top image.")
            return
        
        with st.spinner("Processing images... This may take a few moments."):
            try:
                st.markdown('<h2 class="section-header">🔄 Processing Images</h2>', unsafe_allow_html=True)
                
                front_result = None
                top_result = None
                mapped_results = []
                
                # Process based on selected mode
                if upload_mode in ["Both Images (Complete Analysis)", "Front Image Only (Detection)"]:
                    with st.expander("Front Image Processing", expanded=True):
                        front_result = process_front_image(front_file, confidence_threshold)
                
                if upload_mode in ["Both Images (Complete Analysis)", "Top Image Only (Counting)"]:
                    with st.expander("Top Image Processing", expanded=True):
                        top_result = process_top_image(top_file, confidence_threshold)
                
                # Map results only if both images are processed
                if upload_mode == "Both Images (Complete Analysis)" and front_result and top_result:
                    st.markdown('<h2 class="section-header">🔗 Mapping Results</h2>', unsafe_allow_html=True)
                    mapped_results = map_ids(
                        front_result['predictions'], 
                        top_result['pipe_groups'], 
                        front_file.name
                    )
                
                # Display results
                st.markdown('<h2 class="section-header">📊 Detection Results</h2>', unsafe_allow_html=True)
                
                # Metrics based on mode
                if upload_mode == "Both Images (Complete Analysis)":
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Earrings Detected", front_result['total_detections'])
                    with col2:
                        st.metric("Pipes Found", top_result['total_count'])
                    with col3:
                        total_items = sum([r['count'] for r in mapped_results])
                        st.metric("Total Items", total_items)
                    with col4:
                        unique_classes = len(set([r['detected_class'] for r in mapped_results]))
                        st.metric("Unique Classes", unique_classes)
                
                elif upload_mode == "Front Image Only (Detection)":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Earrings Detected", front_result['total_detections'])
                    with col2:
                        unique_classes = len(set([p['class'] for p in front_result['predictions']]))
                        st.metric("Unique Classes", unique_classes)
                    with col3:
                        avg_confidence = np.mean([p['confidence'] for p in front_result['predictions']])
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                
                elif upload_mode == "Top Image Only (Counting)":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pipes Found", top_result['total_count'])
                    with col2:
                        total_items = sum([p['item_count'] for p in top_result['pipe_groups']])
                        st.metric("Total Items", total_items)
                    with col3:
                        avg_items_per_pipe = total_items / top_result['total_count'] if top_result['total_count'] > 0 else 0
                        st.metric("Avg Items/Pipe", f"{avg_items_per_pipe:.1f}")
                
                # Display processed images based on mode
                if upload_mode == "Both Images (Complete Analysis)":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Front Image Results**")
                        front_img = cv2.imread(front_result['processed_image_path'])
                        front_img_rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
                        st.image(front_img_rgb, caption="Detected Earrings", use_container_width=True)
                    
                    with col2:
                        st.markdown("**Top Image Results**")
                        top_img = cv2.imread(top_result['processed_image_path'])
                        top_img_rgb = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)
                        st.image(top_img_rgb, caption="Detected Pipes", use_container_width=True)
                
                elif upload_mode == "Front Image Only (Detection)":
                    st.markdown("**Front Image Results**")
                    front_img = cv2.imread(front_result['processed_image_path'])
                    front_img_rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
                    st.image(front_img_rgb, caption="Detected Earrings", use_container_width=True)
                
                elif upload_mode == "Top Image Only (Counting)":
                    st.markdown("**Top Image Results**")
                    top_img = cv2.imread(top_result['processed_image_path'])
                    top_img_rgb = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)
                    st.image(top_img_rgb, caption="Detected Pipes", use_container_width=True)
                
                # Display cropped images based on mode
                if upload_mode == "Both Images (Complete Analysis)":
                    st.markdown("**Cropped Images**")
                    cols = st.columns(4)
                    for i, (front_pred, top_pipe) in enumerate(zip(front_result['predictions'], top_result['pipe_groups'])):
                        if i < len(cols):
                            with cols[i]:
                                # Front crop
                                front_crop = cv2.imread(front_pred['image_path'])
                                front_crop_rgb = cv2.cvtColor(front_crop, cv2.COLOR_BGR2RGB)
                                st.image(front_crop_rgb, caption=f"Front: {front_pred['id']}")
                                
                                # Top crop
                                top_crop = cv2.imread(top_pipe['image_path'])
                                top_crop_rgb = cv2.cvtColor(top_crop, cv2.COLOR_BGR2RGB)
                                st.image(top_crop_rgb, caption=f"Top: {top_pipe['pipe_id']} ({top_pipe['item_count']} items)")
                
                elif upload_mode == "Front Image Only (Detection)":
                    st.markdown("**Cropped Earring Images**")
                    cols = st.columns(4)
                    for i, front_pred in enumerate(front_result['predictions']):
                        if i < len(cols):
                            with cols[i]:
                                front_crop = cv2.imread(front_pred['image_path'])
                                front_crop_rgb = cv2.cvtColor(front_crop, cv2.COLOR_BGR2RGB)
                                st.image(front_crop_rgb, caption=f"{front_pred['id']}: {front_pred['class']}")
                
                elif upload_mode == "Top Image Only (Counting)":
                    st.markdown("**Cropped Pipe Images**")
                    cols = st.columns(4)
                    for i, top_pipe in enumerate(top_result['pipe_groups']):
                        if i < len(cols):
                            with cols[i]:
                                top_crop = cv2.imread(top_pipe['image_path'])
                                top_crop_rgb = cv2.cvtColor(top_crop, cv2.COLOR_BGR2RGB)
                                st.image(top_crop_rgb, caption=f"{top_pipe['pipe_id']}: {top_pipe['item_count']} items")
                
                # Results table based on mode
                if upload_mode == "Both Images (Complete Analysis)":
                    st.markdown("**Mapped Results Table**")
                    df_data = []
                    for result in mapped_results:
                        row_data = {
                            'Mapped ID': result['mapped_id'],
                            'Detected Class': result['detected_class'],
                            'Confidence': f"{result['confidence']:.3f}",
                            'Item Count': result['count'],
                            'Status': '✅ High' if result['confidence'] > 0.7 else '⚠️ Medium' if result['confidence'] > 0.5 else '❌ Low'
                        }
                        
                        if show_model_info:
                            row_data['Model Used'] = result.get('model_used', 'Unknown')
                        
                        df_data.append(row_data)
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                
                elif upload_mode == "Front Image Only (Detection)":
                    st.markdown("**Earring Detection Results Table**")
                    df_data = []
                    for pred in front_result['predictions']:
                        row_data = {
                            'Earring ID': pred['id'],
                            'Detected Class': pred['class'],
                            'Confidence': f"{pred['confidence']:.3f}",
                            'Status': '✅ High' if pred['confidence'] > 0.7 else '⚠️ Medium' if pred['confidence'] > 0.5 else '❌ Low'
                        }
                        
                        if show_model_info:
                            row_data['Model Used'] = pred.get('model_used', 'Unknown')
                        
                        df_data.append(row_data)
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                
                elif upload_mode == "Top Image Only (Counting)":
                    st.markdown("**Pipe Counting Results Table**")
                    df_data = []
                    for pipe in top_result['pipe_groups']:
                        row_data = {
                            'Pipe ID': pipe['pipe_id'],
                            'Item Count': pipe['item_count'],
                            'Status': '✅ High' if pipe['item_count'] > 5 else '⚠️ Medium' if pipe['item_count'] > 2 else '❌ Low'
                        }
                        df_data.append(row_data)
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                
                # Create Excel report based on mode
                if upload_mode == "Both Images (Complete Analysis)":
                    excel_path, excel_df = create_excel_report(mapped_results, include_model_info=show_model_info)
                elif upload_mode == "Front Image Only (Detection)":
                    # Create front-only report
                    front_data = []
                    for pred in front_result['predictions']:
                        front_data.append({
                            'Earring ID': pred['id'],
                            'Detected Class': pred['class'],
                            'Confidence': pred['confidence'],
                            'Model Used': pred.get('model_used', 'Unknown') if show_model_info else 'N/A',
                            'Image File': pred['crop_filename'],
                            'Detection Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    df = pd.DataFrame(front_data)
                    excel_filename = f"front_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    excel_path = os.path.join('complete_detection_results/excel_reports', excel_filename)
                    df.to_excel(excel_path, index=False, engine='openpyxl')
                elif upload_mode == "Top Image Only (Counting)":
                    # Create top-only report
                    top_data = []
                    for pipe in top_result['pipe_groups']:
                        top_data.append({
                            'Pipe ID': pipe['pipe_id'],
                            'Item Count': pipe['item_count'],
                            'Image File': pipe['crop_filename'],
                            'Detection Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    df = pd.DataFrame(top_data)
                    excel_filename = f"top_counting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    excel_path = os.path.join('complete_detection_results/excel_reports', excel_filename)
                    df.to_excel(excel_path, index=False, engine='openpyxl')
                
                # Download buttons based on mode
                st.markdown("**📥 Download Results**")
                
                if upload_mode == "Both Images (Complete Analysis)":
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        with open(excel_path, "rb") as f:
                            st.download_button(
                                label="📊 Download Excel Report",
                                data=f.read(),
                                file_name=os.path.basename(excel_path),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                    with col2:
                        with open(front_result['processed_image_path'], "rb") as f:
                            st.download_button(
                                label="🖼️ Download Front Image",
                                data=f.read(),
                                file_name=f"front_{os.path.basename(front_result['processed_image_path'])}",
                                mime="image/jpeg"
                            )
                    
                    with col3:
                        with open(top_result['processed_image_path'], "rb") as f:
                            st.download_button(
                                label="🖼️ Download Top Image",
                                data=f.read(),
                                file_name=f"top_{os.path.basename(top_result['processed_image_path'])}",
                                mime="image/jpeg"
                            )
                
                elif upload_mode == "Front Image Only (Detection)":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with open(excel_path, "rb") as f:
                            st.download_button(
                                label="📊 Download Excel Report",
                                data=f.read(),
                                file_name=os.path.basename(excel_path),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                    with col2:
                        with open(front_result['processed_image_path'], "rb") as f:
                            st.download_button(
                                label="🖼️ Download Front Image",
                                data=f.read(),
                                file_name=f"front_{os.path.basename(front_result['processed_image_path'])}",
                                mime="image/jpeg"
                            )
                
                elif upload_mode == "Top Image Only (Counting)":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with open(excel_path, "rb") as f:
                            st.download_button(
                                label="📊 Download Excel Report",
                                data=f.read(),
                                file_name=os.path.basename(excel_path),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                    with col2:
                        with open(top_result['processed_image_path'], "rb") as f:
                            st.download_button(
                                label="🖼️ Download Top Image",
                                data=f.read(),
                                file_name=f"top_{os.path.basename(top_result['processed_image_path'])}",
                                mime="image/jpeg"
                            )
                
                st.success("✅ Processing completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing images: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>💎 Complete Earring Detection & Counting System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
