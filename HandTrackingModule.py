import cv2
import mediapipe as mp
import time

class handDetector():

    def __init__(self, mode = False, maxHands = 2, complexity = 1, detectionConfidence = 0.5, trackingConfidence = 0.5):

        # Setting the parameters needed

        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    # Hand finder function

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)

        return img

    # Function that finds the coordinates of the hand in the screen

    def findPositions(self, img, handNumber = 0, draw = True):

        landmarksList = []

        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNumber]

            for id, landmark in enumerate(myHand.landmark):
                height, width, channel = img.shape
                # finding the positions
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                landmarksList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return landmarksList

def main():

    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landmarksList = detector.findPositions(img)

        if len(landmarksList) != 0:
            print(landmarksList[4])

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()