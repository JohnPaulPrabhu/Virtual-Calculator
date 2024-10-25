import os
import cv2
import numpy as np
import google.generativeai as genai
# import PIL.Image
from PIL import Image
from dotenv import load_dotenv
from mediapipe.python.solutions import hands
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_drawable_canvas import st_canvas

class Calculator:
    def __init__(self):
        """
        Initializes the Calculator class with webcam capture, canvas, and MediaPipe Hands model.
        """
        # Load environment variables
        load_dotenv()

    
    def streamlit_config(self):
        """
        Configures the Streamlit page appearance and layout.
        """
        st. set_page_config(layout="wide")
        st.title("Visual Calculator")
        ans=False
        col1, _, col3 = st.columns([0.7, 0.02, 0.28])

        with col1:
            stframe = st.empty()
            # Add buttons for erase and reset functionality
            drawing_mode = st.radio("Choose drawing mode:", ["freedraw", "Erase"], index=0)

            if drawing_mode == "Erase":
                stroke_color = "white"  # Set stroke color to white to simulate erasing
                # stroke_width=200
                # stroke_width = st.slider("Eraser Size", min_value=5, max_value=150, value=30)
                stroke_width = st.number_input("Enter Eraser Size", min_value=5, max_value=100, value=30)
            else:
                stroke_color = "red"  # Default stroke color for drawing
                stroke_width=5
 
            canvas_result = st_canvas(
            fill_color="white",  # Background color
            stroke_color=stroke_color,  # Drawing or erasing color
            stroke_width=stroke_width,
            background_color="white",
            height=500,
            width=900,
            display_toolbar=True,
            drawing_mode="freedraw" if drawing_mode != "Erase" else "freedraw",  # Drawing mode for freedraw and erase
            key="canvas",
            )
            # Save the drawn image
            if st.button("Get Answer"):
                if canvas_result.image_data is not None:
                    ans=True
                    # Convert to PIL Image
                    img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
                    img.save("rid.png")  # Save the image
                    st.success("Image saved as input_image.png")
                else:
                    st.warning("Please draw something before saving.")

        
        with col3:
            st.markdown(f'<h5 style="text-align:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        if ans:
            # img = cv2.imread("rid.png")
            # stframe.image(img, channels="RGB")
            result = self.model(img)
            result_placeholder.write(f"Result: {result}")

    def model(self, img):
        """
        Uses Google Generative AI to analyze the image and provide a solution.
        """
        # imgCanvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
        # imgCanvas = PIL.Image.fromarray(imgCanvas)
        genai.configure(api_key=os.getenv("API_KEY"))
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        prompt = """Analyze the image and provide the following:
        * The mathematical equation represented in the image.
        * The solution to the equation.
        * A short and sweet explanation of the steps taken to arrive at the solution."""
        response = model.generate_content([prompt, img])
        return response.text
    

def main():
    c = Calculator()
    c.streamlit_config()

if __name__ == "__main__":
    main()













# # import os
# # import cv2
# # import numpy as np
# # import google.generativeai as genai
# # import PIL.Image
# # from dotenv import load_dotenv
# # from mediapipe.python.solutions import hands
# # import streamlit as st
# # from streamlit_extras.add_vertical_space import add_vertical_space

# # class Calculator:
# #     def __init__(self):
# #         """
# #         Initializes the Calculator class with webcam capture, canvas, and MediaPipe Hands model.
# #         """
# #         # Load environment variables
# #         load_dotenv()
        
# #         self.master = master
# #         self.master.title("Canvas for Writing")
        
# #         self.canvas = tk.Canvas(master, width=400, height=400, bg="white")
# #         self.canvas.pack()

# #         self.canvas.bind("<B1-Motion>", self.paint)
# #         self.canvas.bind("<ButtonRelease-1>", self.reset)

# #         self.button = tk.Button(master, text="Submit", command=self.submit)
# #         self.button.pack()

# #         self.reset_button = tk.Button(master, text="Reset", command=self.reset_canvas)
# #         self.reset_button.pack()

# #         self.image = Image.new("L", (400, 400), 255)  # L mode for grayscale
# #         self.draw = ImageDraw.Draw(self.image)

# #         self.last_x, self.last_y = None, None

        
# #         # Initialize the canvas image
# #         self.canvas = np.zeros(shape=(550, 950, 3), dtype=np.uint8)
        
# #         # Initialize MediaPipe hand object
# #         self.mphands = hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
        
# #         # Set drawing origin to zero
# #         self.p1, self.p2 = 0, 0
        
# #         # Fingers open/close positions
# #         self.fingers = []
    
# #     def streamlit_config(self):
# #         """
# #         Configures the Streamlit page appearance and layout.
# #         """
# #         # Page configuration
# #         st.set_page_config(page_title='Virtual Calculator', layout="wide")

# #         # Set page header background to transparent and remove top padding
# #         page_background_color = """
# #         <style>
# #         [data-testid="stHeader"] { background: rgba(0,0,0,0); }
# #         .block-container { padding-top: 0rem; }
# #         </style>
# #         """
# #         st.markdown(page_background_color, unsafe_allow_html=True)

# #         # Page title
# #         st.markdown(f'<h1 style="text-align: center;">Virtual Calculator</h1>', unsafe_allow_html=True)
# #         add_vertical_space(1)
    
# #     def process_frame(self):
# #         """
# #         Captures and processes a frame from the webcam.
# #         """
# #         _, img = self.cap.read()
# #         img = cv2.resize(img, (950, 550))  # Resize the image
# #         self.img = cv2.flip(img, flipCode=1)  # Flip the image for selfie view
# #         self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
    
# #     def process_hands(self):
# #         """
# #         Processes the image to detect hands and extract landmarks.
# #         """
# #         result = self.mphands.process(image=self.imgRGB)
# #         self.landmark_list = []
# #         if result.multi_hand_landmarks:
# #             for hand_landmark in result.multi_hand_landmarks:
# #                 for idx, lm in enumerate(hand_landmark.landmark):
# #                     h, w, _ = self.img.shape
# #                     x, y = lm.x, lm.y
# #                     cx, cy = int(x * w), int(y * h)
# #                     self.landmark_list.append([idx, cx, cy])
    
# #     def identify_fingers(self):
# #         """
# #         Identifies which fingers are open.
# #         """
# #         self.fingers = []
# #         if self.landmark_list:
# #             for id in [4, 8, 12, 16, 20]:
# #                 # Check if the tip of the finger is higher than the middle joint (except thumb)
# #                 if id != 4 and self.landmark_list[id][2] < self.landmark_list[id - 2][2]:
# #                     self.fingers.append(1)
# #                 # Check thumb separately
# #                 elif id == 4 and self.landmark_list[id][1] < self.landmark_list[id - 2][1]:
# #                     self.fingers.append(1)
# #                 else:
# #                     self.fingers.append(0)
    
# #     def handle_drawing_mode(self):
# #         """
# #         Handles different drawing modes based on finger positions.
# #         """
# #         # Thumb and index fingers are up --> Drawing mode
# #         if sum(self.fingers) == 1 and self.fingers[1] == 1:
# #             cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
# #             if self.p1 == self.p2 == 0:
# #                 self.p1, self.p2 = cx, cy
# #             cv2.line(self.canvas, pt1=(self.p1, self.p2), pt2=(cx, cy), color=(0, 0, 255), thickness=2)
# #             self.p1, self.p2 = cx, cy
# #         # Index and middle finger --> Disable drawing mode
# #         elif sum(self.fingers) == 2 and self.fingers[1] == self.fingers[2] == 1:
# #             self.p1, self.p2 = 0, 0
# #         # Thumb, index, and middle finger --> Erase mode
# #         elif sum(self.fingers) == 3 and self.fingers[3] == self.fingers[1] == self.fingers[2] == 1:
# #             cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
# #             if self.p1 == self.p2 == 0:
# #                 self.p1, self.p2 = cx, cy
# #             cv2.line(self.canvas, pt1=(self.p1, self.p2), pt2=(cx, cy), color=(0, 0, 0), thickness=50)
# #             self.p1, self.p2 = cx, cy
# #         # Thumb and pinky --> Reset the canvas
# #         elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[4] == 1:
# #             self.canvas = np.zeros(shape=(550, 950, 3), dtype=np.uint8)
    
# #     def blend_feed_with_canvas(self):
# #         """
# #         Blends the live camera feed with the canvas to overlay the drawings.
# #         """
# #         img = cv2.addWeighted(src1=self.img, alpha=0.7, src2=self.canvas, beta=1, gamma=0)
# #         imgGray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)  # Convert canvas to grayscale
# #         _, thresh = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)  # Create binary inverse mask
# #         imgRev = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  # Convert mask to BGR
# #         img = cv2.bitwise_and(img, imgRev)  # Remove drawn areas using the mask
# #         self.img = cv2.bitwise_or(img, self.canvas)  # Overlay drawing back onto the live feed
    
# #     def model(self):
# #         """
# #         Uses Google Generative AI to analyze the image and provide a solution.
# #         """
# #         imgCanvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
# #         imgCanvas = PIL.Image.fromarray(imgCanvas)
# #         genai.configure(api_key=os.getenv("API_KEY"))
# #         model = genai.GenerativeModel(model_name='gemini-1.5-flash')
# #         prompt = """Analyze the image and provide the following:
# #         * The mathematical equation represented in the image.
# #         * The solution to the equation.
# #         * A short and sweet explanation of the steps taken to arrive at the solution."""
# #         response = model.generate_content([prompt, imgCanvas])
# #         return response.text
    
# #     def release_cap(self):
# #         """
# #         Releases the webcam.
# #         """
# #         self.cap.release()


# # def main():
# #     c = Calculator()
# #     c.streamlit_config()
# #     col1, _, col3 = st.columns([0.8, 0.02, 0.18])

# #     with col1:
# #         # Stream the webcam video
# #         stframe = st.empty()
    
# #     with col3:
# #         # Placeholder for result output
# #         st.markdown(f'<h5 style="text-align:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
# #         result_placeholder = st.empty()

# #     while True:
# #         if not c.cap.isOpened():
# #             add_vertical_space(5)
# #             st.markdown(body=f'<h4 style="text-align:center; color:orange;">Error: Could not open webcam. Please ensure your webcam is connected and try again</h4>', unsafe_allow_html=True)
# #             break
        
# #         # Process frame, hands, and identify finger positions
# #         c.process_frame()
# #         c.process_hands()
# #         c.identify_fingers()
# #         c.handle_drawing_mode()
# #         c.blend_feed_with_canvas()

# #         # Display the processed frame
# #         c.img = cv2.cvtColor(c.img, cv2.COLOR_BGR2RGB)
# #         stframe.image(c.img, channels="RGB")

# #         # If all five fingers are up, run the model
# #         if sum(c.fingers) == 5:
# #             result = c.model()
# #             result_placeholder.write(f"Result: {result}")
        
# #         if cv2.waitKey(25) & 0xFF == ord('q'):
# #             break
    
# #     c.release_cap()
# #     cv2.destroyAllWindows()

# # if __name__ == "__main__":
# #     main()
