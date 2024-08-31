import os
import cv2
import PIL
import numpy as np
import google.generativeai as genai
from mediapipe.python.solutions import hands, drawing_utils
from dotenv import load_dotenv
import streamlit as st


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
        
        return self.imgRGB
        
    def process_hands(self):
        result = self.mphands.process(image=self.imgRGB)
        print(result.multi_hand_landmarks)
        
        # Draw the landmarks and connects on the image
        self.landmark_list = []
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                drawing_utils.draw_landmarks(self.img, hand_landmark, connections=hands.HAND_CONNECTIONS)
                
                # Extract ID and Origin for Each Landmarks
                for idx, lm in enumerate(hand_landmark.landmark):
                    h, w, c = self.img.shape
                    x, y = lm.x, lm.y
                    cx, cy = int(x*w), int(y*h)
                    self.landmark_list.append([id, cx, cy])
    
    
    def identify_fingers(self):
        # always initialize the list to None
        self.fingers = []
        
        if self.landmark_list:
            for id in [4,8,12,16,20]:
                # other than thumb finger
                if id != 4 and self.landmark_list[id][2] < self.landmark_list[id-2][2]:
                    self.fingers.append(1)
                    
                elif self.landmark_list[id][1] < self.landmark_list[id-2][1]:
                    self.fingers.append(1)
                
                else:
                    self.fingers.append(0)
                    
            # Identify finger open position
            for i in range(5):
                if self.fingers[i] == 1:
                    cx, cy = self.landmark_list[(i+1)*4][1:]
                    cv2.circle(self.imgRGB, center=(cx, cy), radius=5, color=(255,0,255), thickness=1)
                    
    
        
    def release_cap(self):
        self.cap.release()
        

def main():
    c = Calculator()
    
    while True:
        frame = c.process_frame()
        c.process_hands()
        c.identify_fingers()
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    c.release_cap()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    