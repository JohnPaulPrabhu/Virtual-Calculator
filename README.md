# Virtual Calculator

**Virtual Calculator** is an interactive application developed using Python and OpenCV. This project allows users to perform mathematical operations by simulating button clicks using a virtual interface. It leverages computer vision techniques to detect user interactions with the virtual calculator through a webcam.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

The **Virtual Calculator** is designed to create a touchless experience for performing mathematical calculations. By utilizing a webcam or interactive canvas, the application captures user input, detects gestures or clicks, and interprets them as input for the calculator. 

This project is ideal for practicing skills in computer vision and integrating OpenCV for live video processing.

---

## Features

- Virtual calculator interface displayed on the screen.
- Real-time hand gesture or click detection for input.
- Supports from basic to advanced level mathematical operations.
- Uses OpenCV for live video feed and processing.
- Uses Streamlit for interactive canvas.
- Python-based implementation, ensuring compatibility and ease of use.

---

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/JohnPaulPrabhu/Virtual-Calculator.git
   cd Virtual-Calculator
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:
   ```bash
   streamlit run .\main.py
   ```

---

## Usage

1. Open the project by running the `main.py` file using streamlit.
2. Position yourself in front of the webcam if you want to use hand gestures or You can use a mouse to draw or write the mathematical equations.
3. Use gestures or simulate clicks on the virtual buttons displayed on the calculator interface.
4. Perform mathematical calculations in real-time.

---

## Technologies Used

- **Python**: Core programming language for the project.
- **OpenCV**: For real-time video feed and image processing.
- **MediaPipe**: For hand tracking or gesture recognition.
- **NumPy**: For handling numerical operations.
- **Streamlit**: For graphical user interface components.

---


## License

This project is licensed under the [MIT License](LICENSE).

---
