import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils  # 랜드 마크 출력

mpPose = mp.solutions.pose
pose = mpPose.Pose()  # 인체 감지

cap = cv2.VideoCapture('./Videos/크리스 햄스워스 운동 모음.mp4')
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)   # 인체 포즈 감지
    print(results.pose_landmarks)  # 인체 감지 랜드마크 출력

    if results.pose_landmarks:  # 인체 감지시
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)  # 프레임 구하기
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3) # 프레임 좌측 상단에 출력

    cv2.imshow('Image', img)

    cv2.waitKey(1)