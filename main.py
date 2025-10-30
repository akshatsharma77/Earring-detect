from flask import Flask, request, render_template, url_for
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import os
import cv2
import uuid
from werkzeug.utils import secure_filename
import pandas as pd
from flask import send_file


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['CROPPED_FOLDER'] = 'static/cropped'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Create folders if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER'], app.config['CROPPED_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Roboflow API clients
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="0lfsVUdCmiTvLOVlDejI"
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def draw_label(img, text, x, y, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    cv2.rectangle(img, (x, y - 20), (x + text_size[0], y), color, -1)
    cv2.putText(img, text, (x, y - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def group_predictions_by_columns(predictions, x_threshold=50, fixed_width=100, fixed_height=200):
    predictions = sorted(predictions, key=lambda p: p['x'])
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

    for col in columns:
        col.sort(key=lambda p: p['y'])

    merged_columns = []
    for col in columns:
        current_group = []
        groups = []
        col.sort(key=lambda p: p['y'])
        for pred in col:
            if not current_group:
                current_group.append(pred)
            else:
                first_y = current_group[0]['y']
                if abs(pred['y'] - first_y) <= fixed_height:
                    current_group.append(pred)
                else:
                    groups.append(current_group)
                    current_group = [pred]
        if current_group:
            groups.append(current_group)
        merged_columns.extend(groups)

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

def process_front_image(file, confidence=0.50):
    original_filename = secure_filename(file.filename)
    save_name = str(uuid.uuid4()) + "_" + original_filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
    file.save(save_path)

    custom_configuration = InferenceConfiguration(confidence_threshold=confidence, iou_threshold=0.5)

    # Detection with sku-110k model
    print("Sending image to sku-110K...")
    with CLIENT.use_configuration(custom_configuration):
        detection_result = CLIENT.infer(save_path, model_id="sku-110k/4")

    img = cv2.imread(save_path)
    height, width = img.shape[:2]

    predictions = []
    id_counter = 1
    min_conf_threshold = 0.50
    print("detection_result -> ", detection_result , "\n")
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
            crop_filename = f"crop_{uuid.uuid4()}.jpg"
            crop_path = os.path.join(app.config['CROPPED_FOLDER'], crop_filename)
            cv2.imwrite(crop_path, crop)

            classification_result = CLIENT.infer(crop_path, model_id="earrings-test-yolonas/1")

            if classification_result.get('predictions'):
                best_pred = max(classification_result['predictions'], key=lambda x: x['confidence'])
                if best_pred['confidence'] >= min_conf_threshold:
                    predictions.append({
                        'id': f'Earring-ID-{id_counter}',
                        'class': best_pred['class'],
                        'confidence': best_pred['confidence'],
                        'image': url_for('static', filename=f'cropped/{crop_filename}')
                    })
                else:
                    predictions.append({
                        'id': f'Earring-ID-{id_counter}',
                        'class': 'Unknown',
                        'confidence': best_pred['confidence'],
                        'image': url_for('static', filename=f'cropped/{crop_filename}')
                    })
            else:
                predictions.append({
                    'id': f'Earring-ID-{id_counter}',
                    'class': 'Unknown',
                    'confidence': 0.0,
                    'image': url_for('static', filename=f'cropped/{crop_filename}')
                })

            id_counter += 1

    result_img_name = "result_" + save_name
    result_img_path = os.path.join(app.config['RESULT_FOLDER'], result_img_name)
    cv2.imwrite(result_img_path, img)

    print("Front completed......")
    return {
        'original_filename': original_filename,
        'result_image': url_for('static', filename=f'results/{result_img_name}'),
        'predictions': predictions
    }

def process_top_image(file, confidence=0.50, x_threshold=50):
    original_filename = secure_filename(file.filename)
    save_name = str(uuid.uuid4()) + "_" + original_filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
    file.save(save_path)

    custom_configuration = InferenceConfiguration(confidence_threshold=confidence, iou_threshold=0.5)

    # Detection with sku-110k model
    print("Sending image to earring-crop-ugfno/1..")
    with CLIENT.use_configuration(custom_configuration):
        crop_result = CLIENT.infer(save_path, model_id="earring-crop-ugfno/1")

    img = cv2.imread(save_path)
    height, width = img.shape[:2]
  
    id_counter = 1
    min_conf_threshold = 0.50
    pipe_count = []
    if crop_result.get('predictions'):
        sorted_detections = group_detections_by_rows(crop_result['predictions'], y_threshold=100)
        for pred in sorted_detections:
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
            

            with CLIENT.use_configuration(custom_configuration):
                 crop_detection_result = CLIENT.infer(crop_path, model_id="earring-box-count-0cf1h/1")

            if crop_detection_result.get('predictions'):
                for i, pred in enumerate(crop_detection_result['predictions']):
                
                    x, y = int(pred['x']), int(pred['y'])
                    w, h = int(pred['width']), int(pred['height'])
                    start_x, start_y = max(x - w // 2, 0), max(y - h // 2, 0)
                    end_x, end_y = min(x + w // 2, width), min(y + h // 2, height)


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
        'pipe_groups': pipe_count,
        'total_count' : len(pipe_count)
    }


def map_ids(file_name, front_predictions, top_pipes):
    print("front predictions : ", front_predictions)
    print("top pipes : ", top_pipes)
    print("file name : ", file_name)
    
    mapped_results = []
    for i, front_pred in enumerate(front_predictions):
        if i < len(top_pipes):
            mapped_results.append({
                'mapped_id': front_pred['id'],  # Earring-ID-X
                'detected_class': front_pred['class'],
                'count': top_pipes[i]['item_count'],
                'stand_name': file_name
            })
        else:
            mapped_results.append({
                'mapped_id': front_pred['id'],
                'detected_class': front_pred['class'],
                'count': 0,
                'stand_name': file_name
            })
    return mapped_results

def generate_excel(mapped_results):
    df = pd.DataFrame(mapped_results)
    excel_path = os.path.join(app.config['RESULT_FOLDER'], 'earring_results.xlsx')
    df.to_excel(excel_path, index=False, engine='openpyxl')
    return excel_path

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    error = None

    if request.method == 'POST':
        if 'images' not in request.files:
            error = "⚠️ No file part detected."
            return render_template('index2.html', error=error, results=[])

        files = request.files.getlist('images')
        if not files or len(files) != 2:
            error = "⚠️ Please upload exactly two images: one with 'front' and one with 'top' in the filename."
            return render_template('index2.html', error=error, results=[])

        try:
            confidence = 0.50
        except ValueError:
            confidence = 0.50

        front_file = None
        top_file = None
        for file in files:
            if not file or not allowed_file(file.filename):
                error = f"⚠️ Invalid file: {file.filename}"
                return render_template('index2.html', error=error, results=[])
            if 'front' in file.filename.lower():
                front_file = file
            elif 'top' in file.filename.lower():
                top_file = file

        if not front_file or not top_file:
            error = "⚠️ Please ensure one image has 'front' and the other has 'top' in the filename."
            return render_template('index2.html', error=error, results=[])

        try:
            front_result = process_front_image(front_file, confidence=confidence)
            top_result = process_top_image(top_file, confidence=confidence)
            mapped_results = map_ids(front_file, front_result['predictions'], top_result['pipe_groups'])
            generate_excel(mapped_results)
            
            results.append({
                'front': front_result,
                'top': top_result,
                'mapped': mapped_results
            })
        except Exception as e:
            error = f"⚠️ Error processing images: {str(e)}"
            return render_template('index.html', error=error, results=[])

    return render_template('index.html', error=error, results=results)

@app.route('/download_excel')
def download_excel():
    excel_path = os.path.join(app.config['RESULT_FOLDER'], 'earring_results.xlsx')
    if os.path.exists(excel_path):
        return send_file(excel_path, as_attachment=True)
    return "Excel file not found", 404

if __name__ == '__main__':
    app.run(debug=True, port=8000)