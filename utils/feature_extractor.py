def extract_features(results):
    hand_data = {
        "Left":  [0.0] * 42,
        "Right": [0.0] * 42
    }

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            label = handedness.classification[0].label
            coords = []

            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y])

            hand_data[label] = coords

    return hand_data["Left"] + hand_data["Right"]  # ALWAYS 84
