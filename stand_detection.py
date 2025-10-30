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
    page_title="Earring Stand Detection",
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
        'stand_detection_results',
        'stand_detection_results/cropped_images',
        'stand_detection_results/processed_images',
        'stand_detection_results/excel_reports'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return directories

def fix_image_orientation(image_path):
    """Fix image orientation based on EXIF data"""
    try:
        # Open image with PIL to read EXIF data
        with Image.open(image_path) as pil_image:
            # Get EXIF data
            exif = pil_image._getexif()
            
            if exif is not None:
                # Find orientation tag
                for tag, value in exif.items():
                    if ExifTags.TAGS.get(tag) == 'Orientation':
                        orientation = value
                        break
                else:
                    orientation = 1
            else:
                orientation = 1
            
            # Apply rotation based on orientation
            if orientation == 3:
                pil_image = pil_image.rotate(180, expand=True)
            elif orientation == 6:
                pil_image = pil_image.rotate(270, expand=True)
            elif orientation == 8:
                pil_image = pil_image.rotate(90, expand=True)
            
            # Convert PIL image to OpenCV format
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

def process_stand_image(image_path, confidence_threshold=0.5):
    """Process stand image to detect and crop earrings, then classify them"""
    
    # Initialize client
    rf = get_roboflow_client()
    
    # Load image with proper orientation handling
    st.info("🔄 Loading and fixing image orientation...")
    img = fix_image_orientation(image_path)
    height, width = img.shape[:2]
    
    # Save the corrected image temporarily for Roboflow processing
    corrected_image_path = image_path.replace('.jpg', '_corrected.jpg').replace('.jpeg', '_corrected.jpg').replace('.png', '_corrected.png')
    cv2.imwrite(corrected_image_path, img)
    
    # Step 1: Detect earring regions using crop model
    st.info("🔍 Detecting earring regions using crop model...")
    
    # Get the crop model
    crop_model = rf.workspace().project("earring-crop-ugfno").version(1).model
    
    # Run inference on corrected image
    crop_result = crop_model.predict(corrected_image_path, confidence=int(confidence_threshold * 100)).json()
    
    if not crop_result.get('predictions'):
        st.warning("No earrings detected in the image.")
        return None
    
    st.success(f"✅ Found {len(crop_result['predictions'])} earring regions!")
    
    # Process each detection
    results = []
    cropped_images = []
    
    for i, pred in enumerate(crop_result['predictions']):
        # Extract coordinates
        x, y = int(pred['x']), int(pred['y'])
        w, h = int(pred['width']), int(pred['height'])
        
        # Calculate crop boundaries
        start_x = max(x - w // 2, 0)
        start_y = max(y - h // 2, 0)
        end_x = min(x + w // 2, width)
        end_y = min(y + h // 2, height)
        
        # Draw bounding box on original image
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        draw_label(img, f"Region-{i+1}", start_x, start_y)
        
        # Crop the earring
        cropped_img = img[start_y:end_y, start_x:end_x]
        
        # Save cropped image
        crop_filename = f"earring_region_{i+1}_{uuid.uuid4().hex[:8]}.jpg"
        crop_path = os.path.join('stand_detection_results/cropped_images', crop_filename)
        cv2.imwrite(crop_path, cropped_img)
        
        # Step 2: Classify the cropped earring using multiple models and pick the best result
        st.info(f"🔍 Classifying earring region {i+1}...")
        
        # Get both classification models
        class_model_1 = rf.workspace().project("earring-id-new").version(5).model  # earring-id-new/5
        class_model_2 = rf.workspace().project("earring-id-detect").version(2).model  # earring-id-detect/2
        
        # Run classification with both models
        classification_result_1 = class_model_1.predict(crop_path, confidence=30).json()
        classification_result_2 = class_model_2.predict(crop_path, confidence=30).json()
        
        # Find best prediction from both models
        best_prediction = None
        best_confidence = 0
        best_model_id = ""
        
        # Check results from model 1
        if classification_result_1.get('predictions'):
            for pred in classification_result_1['predictions']:
                if pred.get('confidence', 0) > best_confidence:
                    best_prediction = pred
                    best_confidence = pred.get('confidence', 0)
                    best_model_id = "earring-id-new-v5"
        
        # Check results from model 2
        if classification_result_2.get('predictions'):
            for pred in classification_result_2['predictions']:
                if pred.get('confidence', 0) > best_confidence:
                    best_prediction = pred
                    best_confidence = pred.get('confidence', 0)
                    best_model_id = "earring-id-detect-v2"
        
        # Process the best prediction
        if best_prediction:
            result = {
                'region_id': f"Region-{i+1}",
                'earring_id': best_prediction.get('class', 'Unknown'),
                'confidence': best_prediction.get('confidence', 0.0),
                'model_used': best_model_id,
                'crop_image_path': crop_path,
                'crop_filename': crop_filename,
                'bounding_box': {
                    'x': x, 'y': y, 'width': w, 'height': h
                }
            }
        else:
            result = {
                'region_id': f"Region-{i+1}",
                'earring_id': 'Unknown',
                'confidence': 0.0,
                'model_used': 'No Model',
                'crop_image_path': crop_path,
                'crop_filename': crop_filename,
                'bounding_box': {
                    'x': x, 'y': y, 'width': w, 'height': h
                }
            }
        
        results.append(result)
        cropped_images.append(cropped_img)
    
    # Save processed image with bounding boxes
    processed_filename = f"processed_stand_{uuid.uuid4().hex[:8]}.jpg"
    processed_path = os.path.join('stand_detection_results/processed_images', processed_filename)
    cv2.imwrite(processed_path, img)
    
    # Clean up temporary corrected image
    if os.path.exists(corrected_image_path):
        os.remove(corrected_image_path)
    
    return {
        'results': results,
        'processed_image_path': processed_path,
        'total_detections': len(results),
        'cropped_images': cropped_images
    }

def create_excel_report(results, original_filename, include_model_info=False):
    """Create Excel report with detection results"""
    df_data = []
    for result in results:
        row_data = {
            'Region ID': result['region_id'],
            'Earring ID': result['earring_id'],
            'Confidence': round(result['confidence'], 3),
            'Crop Filename': result['crop_filename'],
            'Original Image': original_filename,
            'Detection Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add model info only if requested (for internal use)
        if include_model_info:
            row_data['Model Used'] = result.get('model_used', 'Unknown')
        
        df_data.append(row_data)
    
    df = pd.DataFrame(df_data)
    excel_filename = f"earring_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    excel_path = os.path.join('stand_detection_results/excel_reports', excel_filename)
    df.to_excel(excel_path, index=False, engine='openpyxl')
    
    return excel_path, df

def main():
    # Header
    st.markdown('<h1 class="main-header">💎 Earring Stand Detection System</h1>', unsafe_allow_html=True)
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
    
    # Internal debugging option (hidden from normal users)
    show_model_info = st.sidebar.checkbox(
        "🔧 Show Model Info (Internal)", 
        value=False,
        help="Show which model was used for each prediction"
    )
    
    st.sidebar.markdown("## 📁 Output Folders")
    st.sidebar.info("""
    - **Cropped Images**: Individual earring crops
    - **Processed Images**: Original image with bounding boxes
    - **Excel Reports**: Detection results in spreadsheet format
    """)
    
    # Main content
    st.markdown('<h2 class="section-header">📤 Upload Stand Image</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a stand image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of an earring stand for detection and classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Original Image**")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Stand Image", use_container_width=True)
        
        # Save uploaded file
        temp_path = f"temp_upload_{uuid.uuid4().hex}.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process button
        if st.button("🚀 Process Image", type="primary", use_container_width=True):
            with st.spinner("Processing image... This may take a few moments."):
                try:
                    # Process the image
                    result = process_stand_image(temp_path, confidence_threshold)
                    
                    if result:
                        st.markdown('<h2 class="section-header">📊 Detection Results</h2>', unsafe_allow_html=True)
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Detections", result['total_detections'])
                        with col2:
                            unique_ids = len(set([r['earring_id'] for r in result['results']]))
                            st.metric("Unique Earring IDs", unique_ids)
                        with col3:
                            avg_confidence = np.mean([r['confidence'] for r in result['results']])
                            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                        with col4:
                            high_conf = len([r for r in result['results'] if r['confidence'] > 0.7])
                            st.metric("High Confidence", high_conf)
                        
                        # Display processed image
                        st.markdown("**Processed Image with Detections**")
                        processed_img = cv2.imread(result['processed_image_path'])
                        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                        st.image(processed_img_rgb, caption="Detected Earring Regions", use_container_width=True)
                        
                        # Display cropped images
                        st.markdown("**Cropped Earring Images**")
                        cols = st.columns(3)
                        for i, (result_item, col) in enumerate(zip(result['results'], cols * (len(result['results']) // 3 + 1))):
                            if i < len(result['results']):
                                with col:
                                    crop_img = cv2.imread(result_item['crop_image_path'])
                                    crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                                    st.image(crop_img_rgb, caption=f"{result_item['region_id']}: {result_item['earring_id']}")
                        
                        # Create and display results table
                        st.markdown("**Detection Results Table**")
                        df_data = []
                        for result_item in result['results']:
                            row_data = {
                                'Region ID': result_item['region_id'],
                                'Earring ID': result_item['earring_id'],
                                'Confidence': f"{result_item['confidence']:.3f}",
                                'Status': '✅ High' if result_item['confidence'] > 0.7 else '⚠️ Medium' if result_item['confidence'] > 0.5 else '❌ Low'
                            }
                            
                            # Add model info only if debugging is enabled
                            if show_model_info:
                                row_data['Model Used'] = result_item.get('model_used', 'Unknown')
                            
                            df_data.append(row_data)
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Create Excel report
                        excel_path, excel_df = create_excel_report(result['results'], uploaded_file.name, include_model_info=show_model_info)
                        
                        # Download buttons
                        st.markdown("**📥 Download Results**")
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
                            with open(result['processed_image_path'], "rb") as f:
                                st.download_button(
                                    label="🖼️ Download Processed Image",
                                    data=f.read(),
                                    file_name=os.path.basename(result['processed_image_path']),
                                    mime="image/jpeg"
                                )
                        
                        st.success("✅ Processing completed successfully!")
                        
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>💎 Earring Stand Detection System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
