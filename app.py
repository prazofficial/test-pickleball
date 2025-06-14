import cv2, numpy as np, mediapipe as mp
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Persistent state
homography = None
line_y = None  # kitchen line in court (e.g., 1524 cm from net)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calibrate', methods=['POST'])
def calibrate():
    global homography, line_y
    points = request.json['points']
    pts = np.array([[p['x'], p['y']] for p in points], dtype='float32')

    # Target destination: 5m wide, 0.25m deep rectangle
    dst = np.array([[0, 0], [500, 0], [500, 25], [0, 25]], dtype='float32')
    H, _ = cv2.findHomography(pts, dst)
    homography = H

    # Kitchen line: 7 ft = 213.36 cm from the net
    line_y = 213.36
    return jsonify(success=True)


@app.route('/detect', methods=['POST'])
def detect():
    global homography, line_y
    if homography is None:
        return jsonify({'error':'Calibrate first'}), 400

    data = request.json['image']
    img = _decode_img(data)
    h, w = img.shape[:2]
    res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return jsonify({'result':'NO_PERSON'})

    result = 'CLEAN'
    for side in [mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]:
        lm = res.pose_landmarks.landmark[side]
        pt = np.array([[lm.x * w, lm.y * h]], dtype='float32')
        top = cv2.perspectiveTransform(np.array([pt]), homography)[0][0]
        if top[1] < line_y:
            result = 'FOOT_FAULT'
            break

    return jsonify({'result': result})

def _decode_img(data):
    header, encoded = data.split(',',1)
    import base64
    arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
