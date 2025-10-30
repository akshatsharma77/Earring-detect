# 💎 Earring Stand Detection System

A Streamlit-based application for detecting and classifying earrings from stand images using Roboflow AI models.

## 🚀 Features

- **Stand Image Processing**: Upload images of earring stands
- **Automatic Detection**: Uses `earring-crop-ugfno/1` model to detect earring regions
- **Earring Classification**: Uses `earring-id-new/5` model to classify detected earrings
- **Interactive UI**: Beautiful Streamlit interface with real-time processing
- **Results Export**: Download Excel reports and processed images
- **Organized Output**: Automatically creates folders for different output types

## 📁 Output Structure

```
stand_detection_results/
├── cropped_images/          # Individual earring crops
├── processed_images/        # Original image with bounding boxes
└── excel_reports/          # Detection results in Excel format
```

## 🛠 Installation

1. Install required packages:
```bash
pip install -r requirements_streamlit.txt
```

2. Run the application:
```bash
streamlit run stand_detection.py
```

## 📖 Usage

1. **Upload Image**: Select a stand image (PNG, JPG, JPEG)
2. **Configure**: Adjust confidence threshold in the sidebar
3. **Process**: Click "Process Image" to start detection
4. **View Results**: See detected regions, classifications, and metrics
5. **Download**: Export Excel reports and processed images

## 🎯 Workflow

1. **Detection Phase**: 
   - Uses `earring-crop-ugfno/1` model to find earring regions
   - Draws bounding boxes around detected areas

2. **Classification Phase**:
   - Crops each detected region
   - Uses `earring-id-new/5` model to classify each earring
   - Extracts earring ID and confidence score

3. **Results Phase**:
   - Displays processed image with annotations
   - Shows cropped images with classifications
   - Generates Excel report with all results
   - Provides download options

## 📊 Features

- **Real-time Processing**: Live updates during detection
- **Confidence Metrics**: Visual indicators for detection quality
- **Batch Results**: Process multiple detections at once
- **Export Options**: Excel reports and processed images
- **Responsive UI**: Works on desktop and mobile devices

## 🔧 Configuration

- **Confidence Threshold**: Adjust detection sensitivity (0.1 - 1.0)
- **Model Selection**: Currently uses fixed Roboflow models
- **Output Format**: Excel reports with detailed metadata

## 📝 Output Format

Excel reports include:
- Region ID
- Earring ID (classification result)
- Confidence Score
- Crop Filename
- Original Image Name
- Detection Timestamp


