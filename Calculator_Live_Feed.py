import os
import PIL.Image
import cv2
import PIL
import numpy as np
import google.generativeai as genai
from streamlit_extras.add_vertical_space import add_vertical_space
from mediapipe.python.solutions import hands, drawing_utils
from dotenv import load_dotenv
import streamlit as st
import sys # Need to remove
import openai
import base64
import google.generativeai as genai

class Calculator:
    def streamlit_config(self):

        # page configuration
        # st.set_page_config(page_title='Calculator', layout="wide")

        # page header transparent color and Removes top padding 
        page_background_color = """
        <style>

        [data-testid="stHeader"] 
        {
        background: rgba(0,0,0,0);
        }

        .block-container {
            padding-top: 0rem;
        }

        </style>
        """
        st.markdown(page_background_color, unsafe_allow_html=True)

        # title and position
        st.markdown(f'<h1 style="text-align: center;">Virtual Calculator</h1>',
                    unsafe_allow_html=True)
        add_vertical_space(1)


    def __init__(self):
        # Load the env file for GenAI secret key
        load_dotenv()
        # openai.api_key = os.getenv("API_KEY")
        
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
            cv2.line(self.canvas, pt1=(self.p1, self.p2), pt2=(cx, cy), color=(0, 0, 255), thickness=2)
            self.p1, self.p2 = cx, cy
        
        # index, and middle finger --> Disable drawing mode
        elif sum(self.fingers) == 2 and self.fingers[1] == self.fingers[2] == 1:
            self.p1, self.p2 = 0, 0
        
        # Thumb, index and middle finger --> Erase mode
        elif sum(self.fingers) == 3 and self.fingers[3] == self.fingers[1] == self.fingers[2] == 1:
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
            if self.p1 ==self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(self.canvas, pt1=(self.p1, self.p2), pt2=(cx, cy), color=(0,0,0), thickness=50)
            self.p1, self.p2 = cx, cy
        # Thumb and pinky --> Reset the canvas
        elif sum(self.fingers)==2 and self.fingers[0]==self.fingers[4] == 1:
            self.canvas=np.zeros(shape=(550,950,3), dtype=np.uint8)
    
    def blend_feed_with_canvas(self):
        # Blend the live camera feed with the canvas image to add transparency
        img = cv2.addWeighted(src1=self.img, alpha=0.7, src2=self.canvas, beta=1, gamma=0)
        
        # Convert the canvas image to grayscale to create a mask for blending
        imgGray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)

        # Apply binary inverse thresholding to create a mask for separating drawing from the rest of the canvas
        _, thresh = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)  
        
        # Convert the thresholded mask back to BGR format to match the number of channels of the original image
        imgRev = cv2.cvtColor(thresh, cv2. COLOR_GRAY2BGR)
        
        # Use bitwise AND to remove drawn areas from the image where the mask is black
        img = cv2.bitwise_and(img, imgRev)

        # Use bitwise OR to overlay the drawing back onto the live feed
        self.img = cv2.bitwise_or(img, self.canvas)

    
    def model(self):
        print("=========================Starting the genai process===============================")
        imgCanvs = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
        imgCanvs = PIL.Image.fromarray(imgCanvs)

        genai.configure(api_key=os.getenv("API_KEY"))

        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        prompt = "Analyze the image and provide the following:\n" \
                 "* The mathematical equation represented in the image.\n" \
                 "* The solution to the equation.\n" \
                 "* A short and sweet explanation of the steps taken to arrive at the solution."
        response = model.generate_content([prompt, imgCanvs])
        print(response.text)
        return response.text

    """
    def model(self):
        retval, buffer = cv2.imencode('.jpg', self.img)
        jpg_as_text = base64.b64encode(buffer)
        print(jpg_as_text)
        
        client = openai.OpenAI(api_key=os.getenv("API_KEY"))
        MODEL="gpt-4o-mini"
        prompt = "Analyze the image and provide the following:\n" \
                 "* The mathematical equation represented in the image.\n" \
                 "* The solution to the equation.\n" \
                 "* A short and sweet explanation of the steps taken to arrive at the solution."
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{jpg_as_text}"
                }
                }
            ]
            }
        ]
        print("checking 1")
        
        try:
            response = client.chat.completions.create(
            messages=messages,
            model=MODEL
            )
        except Exception as e:
            # By this way we can know about the type of error occurring
            print("The error is: ",e)
        print("checking 2")
        # response = openai.completions.create(
        #     model=MODEL,
        #     prompt=prompt
        # )
        print("response:========= ", response.choices[0])
        print("checking 3")
        return response.choices[0]
        """

    def release_cap(self):
        self.cap.release()
        

    def main(self):
        c = Calculator()
        c.streamlit_config()
        col1, _, col3 = st.columns([0.8, 0.02, 0.18])

        with col1:
            # Stream the webcam video
            stframe = st.empty()
        
        with col3:
            # Placeholder for result output
            st.markdown(f'<h5 style="text-position:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        
        while True:
            if not c.cap.isOpened():
                add_vertical_space(5)
                st.markdown(body=f'<h4 style="text-position:center; color:orange;">Error: Could not open webcam. \
                                    Please ensure your webcam is connected and try again</h4>', 
                            unsafe_allow_html=True)
                break
            c.process_frame()
            c.process_hands()
            c.identify_fingers()
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            c.handle_drawing_mode()
            c.blend_feed_with_canvas()
            # cv2.imshow('Frame', c.img)
            # Display the Output Frame in the Streamlit App
            c.img = cv2.cvtColor(c.img, cv2.COLOR_BGR2RGB)
            # stframe.image(self.img, channels="RGB")
            stframe.image(c.img, channels="RGB")

            # After Done Process with AI
            if sum(c.fingers) == 5 and c.fingers[0]==c.fingers[1]==c.fingers[2]==c.fingers[3]==c.fingers[4]==1:
                result = c.model()
                result_placeholder.write(f"Result: {result}")
                # break
                # result_placeholder.write(f"Result: {result}")
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        c.release_cap()
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    