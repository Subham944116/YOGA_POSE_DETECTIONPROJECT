# Yoga Pose Detection (Django) - Camera Snapshot Demo

This is a minimal Django project that demonstrates a simple way to capture a camera snapshot
in the browser and send it to a Django backend that uses MediaPipe to detect pose landmarks.

IMPORTANT:
- MediaPipe requires a supported Python version (recommended: **Python 3.10 or 3.12** on Windows).
- If you are on Windows and pip cannot find `mediapipe`, use conda and install from conda-forge:
  ```
  conda create -n mp python=3.10
  conda activate mp
  conda install -c conda-forge mediapipe
  pip install opencv-python django numpy pillow
  ```
- Alternatively, use `mediapipe-nightly` via pip: `pip install mediapipe-nightly` (may be unstable).

Quick start:
1. Create a Python virtual environment (recommended Python 3.10/3.12).
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
   (If `mediapipe` fails on your platform, follow the conda instructions above.)
3. Run migrations and start server:
   ```
   python manage.py migrate
   python manage.py runserver
   ```
4. Open http://127.0.0.1:8000/ in your browser, allow camera access, and click "Capture".

 
 

 
