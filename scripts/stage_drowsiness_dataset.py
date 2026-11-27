"""
Generate and stage test images and sample video sequence for Task 3 Drowsiness Detection.
"""

import os
import sys

# Ensure workspace root is in python path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np

from modules.drowsiness.detector import get_drowsiness_detector
from modules.drowsiness.state_machine import DrowsinessStateMachine

def main():
    img_dir = os.path.join("datasets", "drowsiness", "test_images")
    video_dir = os.path.join("datasets", "drowsiness", "test_videos")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    det = get_drowsiness_detector()

    # Load baseline face
    alice_path = os.path.join("datasets", "attendance", "test_faces", "alice_1.jpg")
    alice_img = cv2.imread(alice_path)
    if alice_img is None:
        raise FileNotFoundError(f"Cannot find {alice_path}")

    # 1. Driver Alert Image
    alert_path = os.path.join(img_dir, "driver_alert.jpg")
    cv2.imwrite(alert_path, alice_img)
    res_alert = det.process_frame(alice_img)
    print(f"Driver Alert -> EAR: {res_alert['ear']:.4f}, MAR: {res_alert['mar']:.4f}, Face: {res_alert['face_detected']}")

    # 2. Driver Closed Eyes Image
    # Extract eye anchor positions
    face_meta = det.detect_face_and_anchors(alice_img)
    fx, fy, fw, fh = [int(v) for v in face_meta["bbox"]]
    lex, ley = [int(round(v)) for v in face_meta["left_eye_center"]]
    rex, rey = [int(round(v)) for v in face_meta["right_eye_center"]]
    nlx, nly = [int(round(v)) for v in face_meta["left_mouth"]]
    nrx, nry = [int(round(v)) for v in face_meta["right_mouth"]]

    closed_img = alice_img.copy()
    skin_r = tuple(int(c) for c in alice_img[min(alice_img.shape[0]-1, rey + 25), rex])
    skin_l = tuple(int(c) for c in alice_img[min(alice_img.shape[0]-1, ley + 25), lex])

    # Fill eye aperture with eyelid skin tone and subtle eyelash line
    cv2.ellipse(closed_img, (rex, rey), (18, 12), 0, 0, 360, skin_r, -1)
    cv2.line(closed_img, (rex - 16, rey), (rex + 16, rey), (int(skin_r[0]*0.75), int(skin_r[1]*0.75), int(skin_r[2]*0.75)), 1)

    cv2.ellipse(closed_img, (lex, ley), (18, 12), 0, 0, 360, skin_l, -1)
    cv2.line(closed_img, (lex - 16, ley), (lex + 16, ley), (int(skin_l[0]*0.75), int(skin_l[1]*0.75), int(skin_l[2]*0.75)), 1)

    
    closed_path = os.path.join(img_dir, "driver_closed_eyes.jpg")
    cv2.imwrite(closed_path, closed_img)
    res_closed = det.process_frame(closed_img)
    print(f"Driver Closed Eyes -> EAR: {res_closed['ear']:.4f}, MAR: {res_closed['mar']:.4f}, Face: {res_closed['face_detected']}")

    # 3. Driver Yawn Image
    yawn_img = alice_img.copy()
    mc_x = int((nlx + nrx) / 2)
    mc_y = int((nly + nry) / 2) + 5
    mw = int(abs(nrx - nlx) * 0.45)
    mh = int(fh * 0.18) # wide open oral cavity
    # Draw dark oral cavity ellipse
    cv2.ellipse(yawn_img, (mc_x, mc_y), (mw, mh), 0, 0, 360, (20, 20, 40), -1)
    # Add lip contour
    cv2.ellipse(yawn_img, (mc_x, mc_y), (mw + 2, mh + 2), 0, 0, 360, (80, 80, 160), 2)
    
    yawn_path = os.path.join(img_dir, "driver_yawn.jpg")
    cv2.imwrite(yawn_path, yawn_img)
    res_yawn = det.process_frame(yawn_img)
    print(f"Driver Yawn -> EAR: {res_yawn['ear']:.4f}, MAR: {res_yawn['mar']:.4f}, Face: {res_yawn['face_detected']}")

    # 4. No Face Image
    no_face_src = os.path.join("datasets", "attendance", "test_faces", "no_face.jpg")
    no_face_dst = os.path.join(img_dir, "no_face.jpg")
    if os.path.exists(no_face_src):
        no_face_img = cv2.imread(no_face_src)
    else:
        no_face_img = np.full((300, 300, 3), 180, dtype=np.uint8)
        cv2.putText(no_face_img, "No Face Steering Wheel", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.imwrite(no_face_dst, no_face_img)
    res_noface = det.process_frame(no_face_img)
    print(f"No Face -> EAR: {res_noface['ear']:.4f}, MAR: {res_noface['mar']:.4f}, Face: {res_noface['face_detected']}")

    # 5. Video: sample_drowsy_sequence.mp4
    video_out_path = os.path.join(video_dir, "sample_drowsy_sequence.mp4")
    h, w = alice_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out_path, fourcc, 20.0, (w, h))

    # Frames 0-19: Alert (20 frames)
    for _ in range(20):
        out.write(alice_img)

    # Frames 20-44: Closed eyes (25 frames -> triggers alarm at frame 35, 15 consecutive frames)
    for _ in range(25):
        out.write(closed_img)

    # Frames 45-59: Wake up / Alert (15 frames -> triggers recovery)
    for _ in range(15):
        out.write(alice_img)

    out.release()
    print(f"Generated sample video at: {video_out_path} (60 frames)")

    # Verify video with state machine
    cap = cv2.VideoCapture(video_out_path)
    sm = DrowsinessStateMachine(ear_threshold=0.22, drowsy_min_frames=15)
    alarms = 0
    recoveries = 0
    f_num = 0
    while cap.isOpened():
        ret, frm = cap.read()
        if not ret:
            break
        f_num += 1
        p = det.process_frame(frm)
        tel = sm.update(p["ear"], p["mar"], p["face_detected"])
        if tel["drowsiness_detected"]:
            alarms += 1
            print(f"Frame {f_num}: Alarm triggered! Closed frames: {tel['consecutive_closed_frames']}")
    cap.release()

    for event in sm.event_log:
        if event["event_type"] == "RECOVERY":
            recoveries += 1

    print(f"Video Verification: Alarms triggered = {alarms}, Recoveries = {recoveries}, Total Logged Events = {len(sm.event_log)}")

if __name__ == "__main__":
    main()
