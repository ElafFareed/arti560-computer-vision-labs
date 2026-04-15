import cv2
import numpy as np


def dense_optical_flow(method, video_path, params=None, to_gray=False, output_path="output.mp4"):
    if params is None:
        params = []

    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()

    if not ret or old_frame is None:
        raise ValueError(f"Could not read video: {video_path}")

    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, new_frame = cap.read()
        if not ret or new_frame is None:
            break

        frame_copy = new_frame.copy()

        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        flow = method(old_frame, new_frame, None, *params)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)

        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

        old_frame = new_frame

    cap.release()
    cv2.destroyAllWindows()