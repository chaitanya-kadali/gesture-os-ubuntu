# Gesture OS | AI & ML
- Mar 2026 | Final Year Project

Demo image of output

<img src="./screenshots/fig 1.png" alt="Preview"  height="250" />


- Developed a real-time hand gesture recognition system using Media Pipe and RFC,achieving high accuracy through landmark-based feature extraction.
- Implemented a SIFT-based gesture recognition model with OpenCV,  enabling feature matching.
- Captures live hand gesture and perform the OS task


# Process for Setup
   ## Step-1 : Environment Setup 
   * **Python Version:** Python 3.10 (Installed via `deadsnakes` PPA on Ubuntu 24.04).
   * **Virtual Environment:** `py_10_env` created and activated.
   * install packages from `requirements.txt` 
   ## Step-2 : Activate the environment
   * cmd: `source py_10_env/bin/activate`
   ## Step-3 : Run the Code
   1. `python 1_collect_imgs.py`
   2. `python 2_create_dataset.py`
   3. `python 3_train_classifier.py`
   4. `python 4_inference_classifier.py`

## Troubleshoot issue
   * the version of `mediapipe` can be `0.9.0.1` or use `0.10.5`