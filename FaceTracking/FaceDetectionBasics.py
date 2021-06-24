import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection   # 미디어 파이프 얼굴 감지 모듈
mpDraw = mp.solution.drawing_utils   # 미디어 파이프 특징 그리는 모듈
faceDetection = mpFaceDetection.FaceDetection()   # 얼굴 감지 객체

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)  # 얼굴 감지 위치 출력

    if results.dectections:
        for id, detection in enumerate(results.detections):
            print(id, detection)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(1)