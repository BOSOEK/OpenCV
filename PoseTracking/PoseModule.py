import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detect=0.5, track=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detect = detect  # 감지 신뢰도
        self.track = track    # 추적 신뢰도

        self.mpDraw = mp.solutions.drawing_utils  # 랜드 마크 출력
        self.mpPose = mp.solutions.pose  # 자세 인식 객체
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detect, self.track)  # 인체 감지

    def findPose(self, img, draw=True):  # 자세 감지
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)   # 인체 포즈 감지
        if draw:
            if self.results.pose_landmarks:  # 인체 감지시
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 60), cv2.FILLED)
        return lmList



def main():
    cap = cv2.VideoCapture('./Videos/TheRack.mp4')
    pTime = 0
    detector = poseDetector()  # 동작 감지기
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # 프레임 구하기
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)  # 프레임 좌측 상단에 출력

        cv2.imshow('Image', img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()