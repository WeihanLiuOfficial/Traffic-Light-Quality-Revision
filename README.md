# LED Board Tester

A Python desktop application (PyQt5 + OpenCV) for testing LED circuit boards.  
The program compares a "golden" reference board image against a test board image, detects the LED lights, and highlights differences visually.

---

## Features
- **Load images**: Select a golden (reference) and a test board image.
- **Automatic detection**: Uses OpenCV to detect LED lights.
- **Comparison**: Matches test board against the golden board layout.
- **Visual results**:
  - Displays the test board with detected LEDs marked.
  - Shows pass/fail status directly in the GUI.
- **Resizable GUI**: Image scales dynamically with the window size.
- **Immediate preview**: Test board image is displayed as soon as it is loaded.

---

## Requirements

- Python 3.8+ (tested on Python 3.13)
- The following Python packages:
  PyQt5
  opencv-python
  numpy
- bash pip install -r requirements.txt
