import os
import numpy as np
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import speech_recognition as sr

class Calculator_Canvas:
    def __init__(self):
        """
        Initializes the Calculator class with webcam capture, canvas, and MediaPipe Hands model.
        """
        # Load environment variables
        load_dotenv()
        genai.configure(api_key=os.getenv("API_KEY"))

    def drawing_page(self):
        # Initialize session state variables
        if "recognized_text" not in st.session_state:
            st.session_state["recognized_text"] = ""

        ans = False
        col1, _, col3 = st.columns([0.7, 0.02, 0.28])

        with col1:
            stframe = st.empty()
            drawing_mode = st.radio("Choose drawing mode:", ["freedraw", "Erase"], index=0)
            if drawing_mode == "Erase":
                stroke_color = "white"
                stroke_width = st.number_input("Enter Eraser Size", min_value=5, max_value=100, value=30)
            else:
                stroke_color = "red"
                stroke_width = 3

            Voice = st.checkbox("Listening for voice instructions...")
            if Voice:
                text_placeholder = st.empty()
                st.info("Listening for voice instructions...")
                t = self.voice()

            # Display the recognized text
            st.markdown("## Instructions:")
            st.text_area("Text from Voice Input", st.session_state["recognized_text"], height=100)

            canvas_result = st_canvas(
                fill_color="white",
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                background_color="white",
                height=500,
                width=900,
                display_toolbar=True,
                drawing_mode="freedraw" if drawing_mode != "Erase" else "freedraw",
                key="canvas",
            )

            if st.button("Get Answer"):
                if canvas_result.image_data is not None:
                    if 't' in locals():
                        print("printing the voice", t)
                    ans = True
                    img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
                    img.save("rid.png")
                    st.success("Image saved as input_image.png")
                else:
                    st.warning("Please draw something before saving.")

        with col3:
            st.markdown(f'<h5 style="text-align:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        if ans:
            print("Checking as: ", st.session_state["recognized_text"])
            result = self.model(img, st.session_state["recognized_text"])
            result_placeholder.write(f"Result: {result}")

    def voice(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Please say the instructions")
            audio = recognizer.listen(source=source)
            try:
                text = recognizer.recognize_google(audio)
                print("Vocie to text: ", text)
                st.session_state["recognized_text"] = text
            except sr.UnknownValueError:
                print("Could not understand voice")
            except sr.RequestError:
                print("Could not request results; {0}".format(sr.RequestError))



    def model(self, img, instructions=None):
        """
        Uses Google Generative AI to analyze the image and provide a solution.
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        if instructions:
            prompt = f"""Analyze the given image. The data for the image is as follows
            {instructions}
            Provide the following:
            * The mathematical equation represented in the image.
            * The solution to the equation.
            * A short and sweet explanation of the steps taken to arrive at the solution."""
        else:
            prompt = """Analyze the image and provide the following:
            * The mathematical equation represented in the image.
            * The solution to the equation.
            * A short and sweet explanation of the steps taken to arrive at the solution."""
        response = model.generate_content([prompt, img])
        return response.text