import os
import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import pyautogui

# Disable GPU for TensorFlow (works with Metal as well on macOS)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress TensorFlow logging (optional, reduces verbosity)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Default gesture to key mapping
gesture_to_key_mapping = {
    "thumbs_up": 'space',  # Example: Thumbs Up -> Space key
    "fist": 'p',           # Example: Fist -> Pause key
    "open_hand": 'esc',    # Example: Open Hand -> Escape key
    "pointing": 'enter'    # Example: Pointing -> Enter key
}
      

def set_gesture_mapping(gesture_name, key):
    """Allows reassigning a keystroke to a gesture."""
    gesture_to_key_mapping[gesture_name] = key
    print(f"Reassigned {gesture_name} to {key}.")

def is_fist(hand_landmarks):
    fingertip_ids = [4, 8, 12, 16, 20]
    for tip_id in fingertip_ids:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            return False
    return True

def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    return thumb_tip.y < index_tip.y

def is_open_hand(hand_landmarks):
    for tip_id in [8, 12, 16, 20]:
        if hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[tip_id - 2].y:
            return False
    return True

def is_pointing(hand_landmarks):
    index_tip = hand_landmarks.landmark[8]
    index_mcp = hand_landmarks.landmark[5]
    return index_tip.y < index_mcp.y and all(
        hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[tip_id - 2].y
        for tip_id in [12, 16, 20]
    )

def handle_gesture(gesture_name):
    """Handle the detected gesture by performing the assigned action."""
    if gesture_name in gesture_to_key_mapping:
        key = gesture_to_key_mapping[gesture_name]
        pyautogui.press(key)
        print(f"{gesture_name} - {key} Pressed")
    else:
        print(f"Gesture '{gesture_name}' not mapped to any key!")

# Function to start camera and gesture recognition
def start_camera():
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # Flip frame horizontally for a mirrored effect
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Detect gestures and handle them
                    if is_fist(hand_landmarks):
                        handle_gesture("fist")
                    elif is_thumbs_up(hand_landmarks):
                        handle_gesture("thumbs_up")
                    elif is_open_hand(hand_landmarks):
                        handle_gesture("open_hand")
                    elif is_pointing(hand_landmarks):
                        handle_gesture("pointing")

            cv2.imshow("Hand Gesture Tool", frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
                break

    cap.release()
    cv2.destroyAllWindows()

# GUI setup
def create_gui():
    root = tk.Tk()
    root.title("Gesture to Key Binding")

    # Creating a notebook (tab container)
    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, expand=True)

    # Create first tab (for gesture-to-key functionality)
    gesture_tab = ttk.Frame(notebook)
    notebook.add(gesture_tab, text="Gesture to Key")

    # Create second tab (for ASL recognition)
    asl_tab = ttk.Frame(notebook)
    notebook.add(asl_tab, text="ASL Recognition")
    asl_functionality

    # Add content to the Gesture-to-Key tab (same as before)
    # Create a frame to contain the gesture-to-key dropdowns
    frame = ttk.Frame(gesture_tab, padding="10")
    frame.grid(row=0, column=0, padx=10, pady=10)

    # Label for gesture bindings
    ttk.Label(frame, text="Gesture").grid(row=0, column=0, padx=10)
    ttk.Label(frame, text="Key Binding").grid(row=0, column=1, padx=10)

    # Dropdown values for gestures and key bindings
    gestures = ["thumbs_up", "fist", "open_hand", "pointing"]
    keys = ['space', 'enter', 'esc', 'p', 'm', 'a', 's', 'd', 'w', 'q', 'z']  # Add more keys as needed

    # Function to handle gesture to key selection
    def on_gesture_select(gesture, key):
        set_gesture_mapping(gesture, key)

    # Create dropdowns for gestures and key bindings
    row = 1
    dropdowns = {}
    for gesture in gestures:
        # Create gesture dropdown
        gesture_label = ttk.Label(frame, text=gesture)
        gesture_label.grid(row=row, column=0, padx=10, pady=5, sticky="w")

        # Create key selection dropdown
        key_combobox = ttk.Combobox(frame, values=keys, state="readonly")
        key_combobox.set(gesture_to_key_mapping[gesture])  # Set the default key
        key_combobox.grid(row=row, column=1, padx=10, pady=5)

        # Update mapping when the user selects a key
        key_combobox.bind("<<ComboboxSelected>>", lambda e, g=gesture, k=key_combobox: on_gesture_select(g, k.get()))

        row += 1

    # Start the camera when the button is pressed
    def start_button_pressed():
        root.withdraw()  # Hide the GUI
        start_camera()   # Start the camera for gesture detection

    start_button = ttk.Button(gesture_tab, text="Start Camera", command=start_button_pressed)
    start_button.grid(row=row, columnspan=2, pady=10)

    # ASL tab is blank for now, but you can add content here later
    ttk.Label(asl_tab, text="ASL Recognition Tab").pack(pady=20)

    root.mainloop()

# Run the GUI
create_gui()
