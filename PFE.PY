from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
import numpy as np
import cv2
import os
import math
from ultralytics import YOLO
import cvzone
from rembg import remove
from PIL import Image, ImageOps
import io
import base64
from moviepy.editor import VideoFileClip


app = Flask(__name__, static_folder='static')
model = YOLO('yolov8l.pt')
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')
def convert_to_mp4(input_video, output_mp4):
    clip = VideoFileClip(input_video)
    clip.write_videofile(output_mp4)
def predict_on_image(image_stream):
    image_data = image_stream.read()
    if not image_data:
        raise ValueError("The image could not be read or is empty.")
    
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image decoding error.")

    results = model.predict(image, conf=0.5)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0] * 100
            cls = int(box.cls[0])
            label = f"{classNames[cls]}: {confidence:.2f}%"
            cv2.rectangle(image, (x1, y1 + 20), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.1)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(frame, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100))
                cls = int(box.cls[0])
                cvzone.putTextRect(frame, f'{classNames[cls]} {conf}%', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

        out.write(frame)

    cap.release()
    out.release()
    return output_path

@app.route('/detection_image', methods=['GET', 'POST'])
def detection_image():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return render_template('index.html', error='No file in request.')
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error='No file selected.')
            if file and allowed_file(file.filename):
                filename = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filename)
                file.stream.seek(0)  # Reset file pointer for further reading

                
                if filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}:
                    output_video_path = os.path.join(OUTPUT_FOLDER, 'output_video.mp4')
                    output = os.path.join(OUTPUT_FOLDER, 'resultat.mp4')
                    
                    output_video_path = process_video(filename, output_video_path)
                    convert_to_mp4(output_video_path, output)
                    
                    
                    return render_template('result.html', video_path=output)
                else:
                    predicted_image = predict_on_image(file.stream)
                    retval, buffer = cv2.imencode('.png', predicted_image)
                    detection_img_base64 = base64.b64encode(buffer).decode('utf-8')

                    file.stream.seek(0)  # Reset file pointer to read the original image
                    original_img_base64 = base64.b64encode(file.stream.read()).decode('utf-8')
                    return render_template('result.html', original_img_data=original_img_base64, detection_img_data=detection_img_base64)

        return render_template('index.html')
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/static/output/<filename>')
def send_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

cap2 = cv2.VideoCapture(0)

def predict_on_frame(frame):
    results = model.predict(frame, conf=0.5)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) 
            cls = int(box.cls[0])
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}%', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)
    
    return frame

def generate_frames():
    while True:
        success, frame = cap2.read()
        if not success:
            break
        else:
            frame = predict_on_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/detection_webcam')
def detection_webcam():
    return render_template('webcam.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/changer_background', methods=['GET', 'POST'])
def changer_background():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('back.html', error='No file in request.')

        file = request.files['image']
        if file.filename == '':
            return render_template('back.html', error='No file selected.')

        action = request.form.get('action')
        bg_color = request.form.get('bg_color', '#ffffff')

        input_image = Image.open(file.stream)
        output_image = remove(input_image)

        if action == "Change Background":
            bg_color = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
            background = Image.new("RGBA", output_image.size, bg_color)
            output_image = Image.alpha_composite(background, output_image.convert("RGBA"))

        byte_io = io.BytesIO()
        output_image.save(byte_io, 'PNG')
        byte_io.seek(0)

        output_path = os.path.join(OUTPUT_FOLDER, 'output.png')
        output_image.save(output_path)

        return render_template('back.html', image_url="/static/output/output.png")


    return render_template('back.html')

@app.route('/filtrage_image', methods=['GET', 'POST'])
def filtrage_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            selected_filter = request.form['filter']
            image = cv2.imread(file_path)

            if selected_filter == 'fc':
                filtered_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif selected_filter == 'bw':
                filtered_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif selected_filter == 'sobel':
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
                sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
                filtered_image = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)
            else:
                return "Filter not supported"

            filtered_image_filename = file.filename
            filtered_image_path = os.path.join(OUTPUT_FOLDER, filtered_image_filename)
            cv2.imwrite(filtered_image_path, filtered_image)

            return redirect(url_for('result2', filename=filtered_image_filename))

    return render_template('index2.html')

@app.route('/static/<filename>')
def result2(filename):
    return render_template('result2.html', filename=filename)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=False, port=5000, host='0.0.0.0')
