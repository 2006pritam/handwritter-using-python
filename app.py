import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Page config
st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")

# Title
st.title("🖐️ Hand Gesture Recognition System")

# Sidebar
st.sidebar.header("Settings")
app_mode = st.sidebar.selectbox("Choose Mode", 
                                 ["Air Writing", "Gesture Recognition"])

# Initialize session state
if 'drawing' not in st.session_state:
    st.session_state.drawing = False
if 'canvas' not in st.session_state:
    st.session_state.canvas = None

# Finger landmark indices
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_BASES = [3, 6, 10, 14, 18]

def get_finger_status(hand_landmarks):
    finger_status = []
    landmarks = hand_landmarks.landmark
    
    # Thumb
    if landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_BASES[0]].x:
        finger_status.append(1)
    else:
        finger_status.append(0)
    
    # Other four fingers
    for i in range(1, 5):
        if landmarks[FINGER_TIPS[i]].y < landmarks[FINGER_BASES[i]].y:
            finger_status.append(1)
        else:
            finger_status.append(0)
    
    return finger_status

def recognize_gesture(fingers):
    if fingers == [0, 1, 0, 0, 0]:
        return "☝️ Pointing Up"
    elif fingers == [0, 1, 1, 0, 0]:
        return "✌️ Victory"
    elif fingers == [1, 0, 0, 0, 0]:
        return "👍 Thumbs Up"
    elif fingers == [0, 0, 0, 0, 0]:
        return "✊ Fist"
    elif fingers == [1, 1, 1, 1, 1]:
        return "🖐️ Open Hand"
    elif fingers == [0, 0, 0, 0, 1]:
        return "🤙 Call Me"
    elif fingers == [1, 0, 0, 0, 1]:
        return "🤘 Rock Sign"
    elif fingers == [0, 1, 1, 1, 1]:
        return "✋ Stop"
    else:
        return ""

# Air Writing Mode
if app_mode == "Air Writing":
    st.header("✍️ Air Writing Mode")
    st.markdown("""
    **Instructions:**
    - Click 'Start Writing' to enable drawing mode
    - Use your index finger to write in the air
    - Click 'Stop Writing' to pause
    - Click 'Clear Canvas' to erase
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🟢 Start Writing"):
            st.session_state.drawing = True
    with col2:
        if st.button("🔴 Stop Writing"):
            st.session_state.drawing = False
    with col3:
        if st.button("🗑️ Clear Canvas"):
            st.session_state.canvas = None
    
    status_text = st.empty()
    frame_placeholder = st.empty()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        
        prev_x, prev_y = None, None
        
        stop_button = st.button("Stop Camera")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Camera not detected!")
                break
            
            frame = cv2.flip(frame, 1)
            
            if st.session_state.canvas is None:
                st.session_state.canvas = np.zeros_like(frame)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            h, w, _ = frame.shape
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get index fingertip
                    index_tip = hand_landmarks.landmark[8]
                    x, y = int(index_tip.x * w), int(index_tip.y * h)
                    
                    if st.session_state.drawing:
                        if prev_x is not None and prev_y is not None:
                            cv2.line(st.session_state.canvas, 
                                   (prev_x, prev_y), (x, y), 
                                   (0, 0, 255), 5)
                        prev_x, prev_y = x, y
                    else:
                        prev_x, prev_y = None, None
            
            # Merge canvas with frame
            frame = cv2.addWeighted(frame, 0.6, st.session_state.canvas, 0.8, 0)
            
            # Add status text
            status_color = (0, 255, 0) if st.session_state.drawing else (0, 0, 255)
            cv2.putText(frame, 
                       f"Writing: {'ON' if st.session_state.drawing else 'OFF'}", 
                       (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                       status_color, 3)
            
            # Display frame
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            status_text.info(f"📝 Writing Mode: {'**ACTIVE**' if st.session_state.drawing else '**PAUSED**'}")
    
    cap.release()

# Gesture Recognition Mode
elif app_mode == "Gesture Recognition":
    st.header("👋 Advanced Gesture Recognition")
    st.markdown("""
    **Supported Gestures:**
    - ☝️ Pointing Up
    - ✌️ Victory
    - 👍 Thumbs Up
    - ✊ Fist
    - 🖐️ Open Hand
    - 🤙 Call Me
    - 🤘 Rock Sign
    - ✋ Stop
    """)
    
    gesture_text = st.empty()
    frame_placeholder = st.empty()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        
        stop_button = st.button("Stop Camera")
        
        while cap.isOpened() and not stop_button:
            success, frame = cap.read()
            if not success:
                st.error("❌ Camera not detected!")
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            detected_gesture = ""
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    
                    fingers = get_finger_status(hand_landmarks)
                    gesture = recognize_gesture(fingers)
                    
                    if gesture:
                        detected_gesture = gesture
                        cv2.putText(frame, gesture, (50, 80), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  1.5, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Display frame
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            if detected_gesture:
                gesture_text.success(f"🎯 Detected: **{detected_gesture}**")
            else:
                gesture_text.info("👋 Show a gesture to the camera")
    
    cap.release()

st.sidebar.markdown("---")
st.sidebar.info("💡 Make sure your camera is enabled and properly positioned")
