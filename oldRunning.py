from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from flask import Flask, request, render_template
import os
import cv2
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Roboflow API
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="OSI2VBzabQevFjlXWxyO",
    
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def draw_label(img, text, x, y, color=(0, 255, 0)):
    """Draw label with background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    cv2.rectangle(img, (x, y - 20), (x + text_size[0], y), color, -1)
    cv2.putText(img, text, (x, y - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def process_image(file, confidence=0.00):
    filename = secure_filename(file.filename)
    unique_filename = str(uuid.uuid4()) + "_" + filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    # Custom config
    custom_configuration = InferenceConfiguration(confidence_threshold=0.2, iou_threshold=0.5)

    with CLIENT.use_configuration(custom_configuration):
        result = CLIENT.infer(filepath, model_id="sku-110k/4")

    img = cv2.imread(filepath)
    cropped_images = []

    for i, pred in enumerate(result['predictions']):
        if pred['confidence'] < confidence:
            continue

        x, y, w, h = map(int, [pred['x'], pred['y'], pred['width'], pred['height']])
        start_x, start_y = max(0, x - w // 2), max(0, y - h // 2)
        end_x, end_y = min(img.shape[1], x + w // 2), min(img.shape[0], y + h // 2)

        # Crop detected region
        cropped_img = img[start_y:end_y, start_x:end_x]
        cropped_filename = f"cropped_{i}_{unique_filename}"
        cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
        cv2.imwrite(cropped_path, cropped_img)
        cropped_images.append(cropped_path)

        # Draw box
        color = (0, 255, 0)
        label = f"{pred['class']} ({pred['confidence'] * 100:.1f}%)"
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), color, 2)
        draw_label(img, label, start_x, start_y, color=color)

    result_path = os.path.join(app.config['UPLOAD_FOLDER'], "result_" + unique_filename)
    cv2.imwrite(result_path, img)

    total_count = len([pred for pred in result['predictions'] if pred['confidence'] >= confidence])
    return {
        'original_filename': filename,
        'result_image': result_path,
        'cropped_images': cropped_images,
        'predictions': [pred for pred in result['predictions'] if pred['confidence'] >= confidence],
        'total_count': total_count
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'images' not in request.files:
            return render_template("index.html", error="No file part")

        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            return render_template("index.html", error="No selected files")

        try:
            confidence = float(request.form.get('confidence', 0.10))
        except ValueError:
            confidence = 0.10

        results = []
        for file in files:
            if allowed_file(file.filename):
                try:
                    result = process_image(file, confidence=confidence)
                    results.append(result)
                except Exception as e:
                    print(f"Error: {str(e)}")
                    continue

        if not results:
            return render_template("index.html", error="No valid files processed")
        return render_template("index.html", results=results)

    return render_template("index.html")

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
