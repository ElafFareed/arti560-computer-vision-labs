import cv2
import numpy as np


def lucas_kanade_method(video_path, output_path="LucasKanade.mp4"):
    cap = cv2.VideoCapture(video_path)

    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    color = np.random.randint(0, 255, (100, 3))

    ret, old_frame = cap.read()
    if not ret or old_frame is None:
        raise ValueError(f"Could not read video: {video_path}")

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    if p0 is None:
        raise ValueError("Could not detect initial features to track.")

    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is None or st is None:
            print("Tracking failed")
            break

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = np.round(new.ravel()).astype(int)
            c, d = np.round(old.ravel()).astype(int)
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)

        cv2.imshow("frame", img)
        k = cv2.waitKey(25) & 0xFF

        if k == 27:
            break
        if k == ord("c"):
            mask = np.zeros_like(old_frame)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()