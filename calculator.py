import os
import cv2
import PIL
import numpy as np
import google.generativeai as genai
from mediapipe.python.solutions import hands, drawing_utils
from dotenv import load_dotenv
import streamlit as st
import sys # Need to remove

class Calculator:
    def __init__(self):
        # Load the env file for GenAI secret key
        load_dotenv()
        
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=950)
        self.cap.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=550)
        self.cap.set(propId=cv2.CAP_PROP_BRIGHTNESS, value=130)
        
        # Initialize the canvas image
        self.canvas = np.zeros(shape=(550,950,3), dtype=np.uint8)
        
        # Initialize mediapipe hand object
        self.mphands = hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
        
        # Set drawing origin to zero
        self.p1, self.p2 = 0, 0
        
        # Set previouse time to zero
        self.p_time = 0
        
        # Fingers open/close Positions
        self.fingers = []
        
    def process_frame(self):
        # Reading the frame from the webcam
        _, img = self.cap.read()
        
        # Resize the image
        img = cv2.resize(img, (950, 550))
        
        # Flip the image horizontally to have the selfie view
        self.img = cv2.flip(img, flipCode=1)
        
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        
    def process_hands(self):
        result = self.mphands.process(image=self.imgRGB)
        # Draw the landmarks and connects on the image
        self.landmark_list = []
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                # drawing_utils.draw_landmarks(self.img, hand_landmark, connections=hands.HAND_CONNECTIONS) # Not necessary unless need to show the landmark connections
                
                # Extract ID and Origin for Each Landmarks
                for idx, lm in enumerate(hand_landmark.landmark):
                    h, w, c = self.img.shape
                    x, y = lm.x, lm.y
                    cx, cy = int(x*w), int(y*h)
                    self.landmark_list.append([idx, cx, cy])
    
    
    def identify_fingers(self):
        # always initialize the list to None
        self.fingers = []
        
        if self.landmark_list:
            for id in [4,8,12,16,20]:
                # Checknig if tip of the finger height is higher than middle of the finger
                # other than thumb finger
                if id != 4 and self.landmark_list[id][2] < self.landmark_list[id-2][2]:
                    self.fingers.append(1)
                # Checking thumb finger
                elif id == 4 and self.landmark_list[id][1] < self.landmark_list[id-2][1]:
                    self.fingers.append(1)
                else:
                    self.fingers.append(0)
                    
            # Identify finger open position
            for i in range(5):
                if self.fingers[i] == 1:
                    cx, cy = self.landmark_list[(i+1)*4][1:]
                    cv2.circle(self.img, center=(cx, cy), radius=5, color=(255,0,255), thickness=1)
                    
    
    def handle_drawing_mode(self):
        # Thumb and index fingers are up --> Drawing mode
        if sum(self.fingers) == 1 and self.fingers[1] == 1:
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
            print(cx, cy)
            if self.p1 == self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(self.canvas, pt1=(self.p1, self.p2), pt2=(cx, cy), color=(0, 0, 255), thickness=5)
            self.p1, self.p2 = cx, cy
        
        # Thumb, index, and middle finger --> Disable drawing mode
        elif sum(self.fingers) == 3 and self.fingers[3] == self.fingers[1] == self.fingers[2] == 1:
            self.p1, self.p2 = 0, 0
        
        # Thumb and middle finger --> Erase mode
        elif sum(self.fingers) == 2 and self.fingers[1] == self.fingers[2] == 1:
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
            if self.p1 ==self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(self.canvas, pt1=(self.p1, self.p2), pt2=(cx, cy), color=(0,0,0), thickness=50)
            self.p1, self.p2 = cx, cy
        # Thumb and pinky --> Reset the canvas
        elif sum(self.fingers)==2 and self.fingers[0]==self.fingers[4] == 1:
            self.canvas=np.zeros(shape=(550,950,3), dtype=np.uint8)
    
    def blend_feed_with_canvas(self):
        # self.img = cv2.addWeighted(self.img, alpha=0.8, src2=self.canvas, beta=1, gamma=0)
        img = cv2.addWeighted(src1=self.img, alpha=0.7, src2=self.canvas, beta=1, gamma=0)
        imgGray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)  
        imgRev = cv2.cvtColor(thresh, cv2. COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgRev)
        self.img = cv2.bitwise_or(img, self.canvas)


    def release_cap(self):
        self.cap.release()
        

def main():
    c = Calculator()
    
    while True:
        c.process_frame()
        c.process_hands()
        c.identify_fingers()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        c.handle_drawing_mode()
        c.blend_feed_with_canvas()
        cv2.imshow('Frame', c.img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    c.release_cap()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    