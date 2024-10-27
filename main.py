import os
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from mediapipe.python.solutions import hands
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_drawable_canvas import st_canvas

from calculator_with_canvas import Calculator_Canvas
from Calculator_Live_Feed import Calculator

class Main:
    def __init__(self):
        """
        Initializes the Calculator class with webcam capture, canvas, and MediaPipe Hands model.
        """
        # Load environment variables
        load_dotenv()
        genai.configure(api_key=os.getenv("API_KEY"))

    def streamlit_config(self):
        st.set_page_config(page_title='Calculator', layout="wide")
        st.title("Visual Calculator")

        # Create a sidebar for navigation
        page = st.sidebar.selectbox("Select Page", ["Drawing Page", "Live Feed"])

        if page == "Drawing Page":
            c = Calculator_Canvas()
            c.drawing_page()
        elif page == "Live Feed":
            o = Calculator()
            o.main()
    
    # def model(self, img):
    #     """
    #     Uses Google Generative AI to analyze the image and provide a solution.
    #     """
    #     model = genai.GenerativeModel('gemini-1.5-flash')
    #     prompt = """Analyze the image and provide the following:
    #     * The mathematical equation represented in the image.
    #     * The solution to the equation.
    #     * A short and sweet explanation of the steps taken to arrive at the solution."""
    #     # img = Image.open("rid1.png")
    #     response = model.generate_content([prompt, img])
    #     return response.text


def main():
    # print(model_testing())
    m = Main()
    m.streamlit_config()
    

if __name__ == "__main__":
    main()