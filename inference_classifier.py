import pickle
import cv2
import mediapipe as mp
import numpy as np

from datetime import datetime



from utils.feature_extractor import extract_features
from utils.app_managing import AppProcessManager


def do_action(name, code, app_manager):
    if(code<=3):
        app_manager.open_app(name, name)
    elif(code<=7):
        app_manager.close_app(name)

def isModifyApp(last_time,app_name):
    if app_name not in last_time:
        return True
    last = last_time[app_name]
    cur = datetime.now()
    diff = int((cur - last).total_seconds())
    return (diff>=15)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
# this 0 for above... needs to be changed and depends brand of company
# each laptop brand have each code, but most of them have 0 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,      # IMPORTANT 
    max_num_hands=2,
    min_detection_confidence=0.3
)

labels_dict = {0: '0 cheese', 1: '1 calculator', 2: '2 vlc', 3:'3 clock', 4: '4 cheese', 5: '5 calculator', 6: '6 vlc', 7:'7 clock'}
related_app = {0:"cheese", 1:"gnome-calculator", 2:"vlc",3:"gnome-clocks", 4:"cheese", 5:"gnome-calculator", 6:"vlc",7:"gnome-clocks"}
last_time={}
app_manager = AppProcessManager()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    left_hand=[]
    right_hand = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):
            
            label = handedness.classification[0].label  # 'Left' or 'Right'

            x_ = []
            y_ = []
            temp = []

            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
             # Collect raw coords
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            # Normalize (same as create_dataset.py)
            for lm in hand_landmarks.landmark:
                temp.append(lm.x - min(x_))
                temp.append(lm.y - min(y_))

            if label == 'Left':
                left_hand = temp
            else:
                right_hand = temp

        # Zero-pad missing hand
        if len(left_hand) == 0:
            left_hand = [0.0] * 42
        if len(right_hand) == 0:
            right_hand = [0.0] * 42

        data_aux = left_hand + right_hand  # 84 features

        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw combined bounding box
            all_x = []
            all_y = []

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    all_x.append(lm.x)
                    all_y.append(lm.y)

            x1 = int(min(all_x) * W) - 10
            y1 = int(min(all_y) * H) - 10
            x2 = int(max(all_x) * W) + 10
            y2 = int(max(all_y) * H) + 10

            app_name = related_app[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            if(isModifyApp(last_time,app_name)):
                last_time[app_name] = datetime.now()
                do_action(app_name, int(prediction[0]), app_manager)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
