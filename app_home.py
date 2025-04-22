from flask import Flask, Response, render_template
import cv2
from model import process_frame

app = Flask(__name__)

# Initialize webcam
camera = cv2.VideoCapture(0)  # 0 for default webcam

@app.route('/')
def home():
    return render_template('index.html')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process the frame using YOLO
            frame = process_frame(frame)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as a response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)