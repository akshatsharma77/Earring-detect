from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from flask import Flask, request, render_template, url_for
import os
import cv2
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['CROPPED_FOLDER'] = 'static/cropped'
app.config['RESULT_FOLDER'] = 'static/results'


# Roboflow API clients
DETECTOR_CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="IPLBBwOtBmABU9HMwtQ1"
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

def group_predictions_by_columns(predictions, x_threshold=50, fixed_width=100, fixed_height=200):
    """Groups predictions into columns (pipes) based on X-coordinate proximity and merges within fixed box size"""
    predictions = sorted(predictions, key=lambda p: p['x'])  # Sort left to right for grouping
    print("---------------------------------->>>>>>>>>>>>>>>>>")
    print("predictions : ", predictions)
    
    columns = []
    for pred in predictions:
        added_to_column = False
        for col in columns:
            avg_x = sum(p['x'] for p in col) / len(col)
            if abs(pred['x'] - avg_x) <= x_threshold:
                col.append(pred)
                added_to_column = True
                break
        if not added_to_column:
            columns.append([pred])

    # Sort each column top to bottom
    for col in columns:
        col.sort(key=lambda p: p['y'])

    # Merge detections within fixed_height vertically
    merged_columns = []
    for col in columns:
        current_group = []
        groups = []
        col.sort(key=lambda p: p['y'])  # Ensure sorted by Y

        for pred in col:
            if not current_group:
                current_group.append(pred)
            else:
                # Check if the current prediction is within fixed_height of the first prediction in the group
                first_y = current_group[0]['y']
                if abs(pred['y'] - first_y) <= fixed_height:
                    current_group.append(pred)
                else:
                    groups.append(current_group)
                    current_group = [pred]
        if current_group:
            groups.append(current_group)

        merged_columns.extend(groups)

    # Sort merged columns by average Y-coordinate (top to bottom)
    merged_columns.sort(key=lambda col: sum(p['y'] for p in col) / len(col))

    return merged_columns, fixed_width, fixed_height

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


def process_image(file, confidence, x_threshold=50):
    original_filename = secure_filename(file.filename)
    save_name = str(uuid.uuid4()) + "_" + original_filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
    file.save(save_path)

    custom_configuration = InferenceConfiguration(confidence_threshold=confidence, iou_threshold=0.5)

    # Detection with sku-110k model
    print("Sending image to Roboflow...")
    with DETECTOR_CLIENT.use_configuration(custom_configuration):
        crop_result = DETECTOR_CLIENT.infer(save_path, model_id="earring-crop-ugfno/1")
    print("Received response from Roboflow.")

    img = cv2.imread(save_path)
    height, width = img.shape[:2]

    predictions = []
    id_counter = 1
    min_conf_threshold = 0.50
    pipe_count = []
    if crop_result.get('predictions'):
        # sorted_detections = group_detections_by_rows(crop_result['predictions'], y_threshold=100)
        for pred in crop_result['predictions']:
            x, y = int(pred['x']), int(pred['y'])
            w, h = int(pred['width']), int(pred['height'])
            start_x, start_y = max(x - w // 2, 0), max(y - h // 2, 0)
            end_x, end_y = min(x + w // 2, width), min(y + h // 2, height)

            draw_label(img, f"Pipe-ID-{id_counter}", start_x, start_y)

            # save cropped images
            crop = img[start_y:end_y, start_x:end_x]
            crop_filename = f"crop_{original_filename}_{id_counter}.jpg"
            crop_path = os.path.join(app.config['CROPPED_FOLDER'], crop_filename)
            cv2.imwrite(crop_path, crop)
            
            if crop.shape[0] > 1000 or crop.shape[1] > 1000:
              crop = cv2.resize(crop, (800, 800), interpolation=cv2.INTER_AREA)

            print("Sending image to Roboflow...")
            with DETECTOR_CLIENT.use_configuration(custom_configuration):
                 crop_detection_result = DETECTOR_CLIENT.infer(crop_path, model_id="earring-box-count-0cf1h/1")
            print("Received response from Roboflow.")

            if crop_detection_result.get('predictions'):
                for i, pred in enumerate(crop_detection_result['predictions']):
                
                    x, y = int(pred['x']), int(pred['y'])
                    w, h = int(pred['width']), int(pred['height'])
                    start_x, start_y = max(x - w // 2, 0), max(y - h // 2, 0)
                    end_x, end_y = min(x + w // 2, width), min(y + h // 2, height)

                    # cropped_img = img[start_y:end_y, start_x:end_x]
                    # cropped_filename = f"pipe_{id_counter}_{save_name}"
                    # cropped_path = os.path.join(app.config['CROPPED_FOLDER'], cropped_filename)
                    # cv2.imwrite(cropped_path, cropped_img)

                    cv2.rectangle(crop, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    draw_label(crop, f"item", start_x, start_y, (0, 255, 0))

            pipe_count.append({
                            'pipe_id': f'Pipe-{id_counter}',
                            'item_count': len(crop_detection_result['predictions']),
                            'image': url_for('static', filename=f'cropped/{crop_filename}')
            })

            id_counter += 1

    result_img_name = "result_" + save_name
    result_img_path = os.path.join(app.config['RESULT_FOLDER'], result_img_name)
    cv2.imwrite(result_img_path, img)

    return {
        'original_filename': original_filename,
        'result_image': url_for('static', filename=f'results/{result_img_name}'),
        'predictions': pipe_count,
        'total_count' : len(pipe_count)
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
            confidence = float(request.form.get('confidence', 0.50))
        except ValueError:
            confidence = 0.10

        results = []
        for file in files:
            if allowed_file(file.filename):
                try:
                    print(file)
                    result = process_image(file, confidence)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
                    continue

        if not results:
            return render_template("index.html", error="No valid files processed")
        return render_template("index.html", results=results)

    return render_template("index.html")

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=8000)