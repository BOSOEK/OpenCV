import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection   # 미디어 파이프 얼굴 감지 모듈
mpDraw = mp.solutions.drawing_utils   # 미디어 파이프 특징 그리는 모듈
faceDetection = mpFaceDetection.FaceDetection(0.75)   # 얼굴 감지 객체

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 이미지를 rgb로 변환
    results = faceDetection.process(imgRGB)   # 얼굴 감지 프로세스 결과
    print(results)  # 얼굴 감지 위치 출력

    if results.detections:   # 점감지
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)  # 경계 상자와 특징 점 출력
            #print(id, detection)
            #print(detection.score)  # 얼굴이라고 생각하는 퍼센트 출력(신뢰도 값)
            #print(detection.location_data.relative_bounding_box)  # 얼굴 감지 네모 경계 상자 출력
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (0, 255, 0), 2)  # 경계 상자만 출력
            cv2.putText(img, f'FPS: {int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)  # 신뢰도 출력

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(1)