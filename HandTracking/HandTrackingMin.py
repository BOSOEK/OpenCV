import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # 비디오 캡쳐 사용

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # 손 인식 모델
mpDraw = mp.solutions.drawing_utils  # 점이나 선 그려주는 객체

pTime = 0 #이전시간
cTime = 0 #현재시간

while True:
    success, img = cap.read()  # 성공시 이미지 읽어옴
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:  # 손 인식 시
        for handLms in results.multi_hand_landmarks:  # 하나씩
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, lm)
                #if id == 0:  #손의 0번째 점에
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # 이미지 손 위에 점을 그리고 그 점들을 잇는다.
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # 프레임 출력

    cv2.imshow("Image", img)
    cv2.waitKey(1)