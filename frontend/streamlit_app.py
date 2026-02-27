"""
Streamlit Frontend Application
Vision-based Model Authentication with Face Recognition and Token Login
"""

import streamlit as st
import requests
import cv2
import numpy as np
import base64
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# API endpoints
API_URL = "http://127.0.0.1:8000"
AUTH_ENDPOINT = f"{API_URL}/api/auth"

# Session state keys
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = None
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = None
if 'auth_mode' not in st.session_state:
    st.session_state.auth_mode = "login"  # login, register


class FaceCaptureTransformer(VideoTransformerBase):
    """Video transformer for face capture"""
    
    def __init__(self):
        self.captured_frame = None
        self.capture_triggered = False
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Draw a rectangle to show capture area
        height, width = img.shape[:2]
        cv2.rectangle(img, (width//4, height//4), (3*width//4, 3*height//4), (0, 255, 0), 2)
        
        # Add text instruction
        cv2.putText(img, "Look at the camera", (width//4 - 20, height//4 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.capture_triggered:
            self.captured_frame = img.copy()
            cv2.putText(img, "CAPTURED!", (width//2 - 60, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return img


def login_with_face():
    """Face login section with webcam"""
    st.header("🔐 Face Login")
    
    st.info("Please look at the camera for face authentication")
    
    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}
    )
    
    # Create webrtc streamer
    webrtc_ctx = webrtc_streamer(
        key="face-login",
        video_transformer_factory=FaceCaptureTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔍 Authenticate with Face", type="primary", use_container_width=True):
            if webrtc_ctx.video_transformer:
                # Get the captured frame
                frame = webrtc_ctx.video_transformer.captured_frame
                
                if frame is not None:
                    # Convert to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send to API
                    try:
                        response = requests.post(
                            f"{AUTH_ENDPOINT}/face-login-image",
                            json={"image_data": image_base64}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.authenticated = True
                            st.session_state.patient_id = data.get("patient_id")
                            st.session_state.patient_name = data.get("patient_name")
                            st.success(f"✅ {data.get('message')}")
                            st.rerun()
                        else:
                            st.error(f"Face not recognized. Please try again or use token login.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please wait for the camera to initialize and capture a frame")
            else:
                st.warning("Please start the camera first")
    
    with col2:
        if st.button("🔑 Use Token Instead", use_container_width=True):
            st.session_state.auth_mode = "token_login"
            st.rerun()


def login_with_token():
    """Token login section"""
    st.header("🔑 Token Login")
    
    st.info("Enter your token to login. Don't have a token? Register below!")
    
    token = st.text_input("Enter Your Token", type="password")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Login with Token", type="primary", use_container_width=True):
            if token:
                try:
                    response = requests.post(
                        f"{AUTH_ENDPOINT}/token-login",
                        json={"token": token}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.authenticated = True
                        st.session_state.patient_id = data.get("patient_id")
                        st.session_state.auth_mode = "login"
                        st.success(f"✅ {data.get('message')}")
                        st.rerun()
                    else:
                        st.error("Invalid token. Please check and try again.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter your token")
    
    with col2:
        if st.button("⬅️ Back to Face Login", use_container_width=True):
            st.session_state.auth_mode = "login"
            st.rerun()
    
    # Registration section
    st.divider()
    st.subheader("New User? Register Here")
    
    with st.expander("Register New Account", expanded=False):
        register_form()


def register_form():
    """Registration form"""
    st.write("Create a new account to get your token")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name *")
        email = st.text_input("Email")
    
    with col2:
        phone = st.text_input("Phone")
        age = st.number_input("Age", min_value=1, max_value=150, step=1)
    
    gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
    
    if st.button("Register", type="primary"):
        if name:
            try:
                response = requests.post(
                    f"{AUTH_ENDPOINT}/register",
                    params={
                        "name": name,
                        "email": email if email else None,
                        "phone": phone if phone else None,
                        "age": int(age) if age else None,
                        "gender": gender if gender else None,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Registration Successful!")
                    st.info(f"Your Token: `{data.get('token')}`")
                    st.warning("⚠️ Please save this token! You will need it for login.")
                    
                    # Offer face registration
                    if st.button("Register Face for Future Logins"):
                        st.session_state.auth_mode = "face_register"
                        st.session_state.new_patient_id = data.get("patient_id")
                        st.rerun()
                else:
                    st.error(f"Registration failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter your name")


def register_face():
    """Face registration section with webcam"""
    st.header("📸 Face Registration")
    
    st.info("Please look at the camera to register your face")
    
    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}
    )
    
    # Create webrtc streamer
    webrtc_ctx = webrtc_streamer(
        key="face-register",
        video_transformer_factory=FaceCaptureTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📸 Capture & Register Face", type="primary", use_container_width=True):
            if webrtc_ctx.video_transformer:
                # Get the captured frame
                frame = webrtc_ctx.video_transformer.captured_frame
                
                if frame is not None:
                    # Convert to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send to API
                    patient_id = st.session_state.get('new_patient_id', 1)
                    try:
                        response = requests.post(
                            f"{AUTH_ENDPOINT}/face-register-image",
                            params={"patient_id": patient_id},
                            json={"image_data": image_base64}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.success(f"✅ {data.get('message')}")
                            st.balloons()
                            st.info("You can now login with your face!")
                            
                            # Go back to login
                            if st.button("Go to Login"):
                                st.session_state.auth_mode = "login"
                                st.rerun()
                        else:
                            st.error(f"Face registration failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please wait for the camera to initialize and capture a frame")
            else:
                st.warning("Please start the camera first")
    
    with col2:
        if st.button("Skip for Now", use_container_width=True):
            st.session_state.auth_mode = "login"
            st.rerun()


# ============================================================================
# MAIN APPLICATION - After Authentication
# ============================================================================

def show_pre_consultation_form():
    """Pre-Consultation Form Section"""
    st.header("📋 Pre-Consultation Form")
    
    # Initialize form session state
    if 'chief_complaint' not in st.session_state:
        st.session_state.chief_complaint = ""
    if 'symptoms' not in st.session_state:
        st.session_state.symptoms = []
    if 'symptom_duration' not in st.session_state:
        st.session_state.symptom_duration = ""
    if 'severity' not in st.session_state:
        st.session_state.severity = "mild"
    if 'medical_history' not in st.session_state:
        st.session_state.medical_history = ""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.chief_complaint = st.text_input(
            "Chief Complaint *", 
            value=st.session_state.chief_complaint,
            placeholder="e.g., Headache, Fever, Cough"
        )
        
        symptom_input = st.text_input(
            "Symptoms (comma separated)",
            value=",".join(st.session_state.symptoms) if st.session_state.symptoms else "",
            placeholder="e.g., headache, nausea, fatigue"
        )
        if symptom_input:
            st.session_state.symptoms = [s.strip() for s in symptom_input.split(",") if s.strip()]
        
        st.session_state.symptom_duration = st.selectbox(
            "How long have you had these symptoms?",
            ["", "Less than a day", "1-3 days", "4-7 days", "1-2 weeks", "More than 2 weeks"],
            index=0 if not st.session_state.symptom_duration else ["", "Less than a day", "1-3 days", "4-7 days", "1-2 weeks", "More than 2 weeks"].index(st.session_state.symptom_duration) if st.session_state.symptom_duration in ["", "Less than a day", "1-3 days", "4-7 days", "1-2 weeks", "More than 2 weeks"] else 0
        )
    
    with col2:
        st.session_state.severity = st.selectbox(
            "Severity Level",
            ["mild", "moderate", "severe"],
            index=["mild", "moderate", "severe"].index(st.session_state.severity) if st.session_state.severity in ["mild", "moderate", "severe"] else 0
        )
        
        st.session_state.medical_history = st.text_area(
            "Medical History (optional)",
            value=st.session_state.medical_history,
            placeholder="Any existing conditions, allergies, etc."
        )
    
    # Submit form button
    if st.button("Submit Pre-Consultation Form", type="primary", use_container_width=True):
        if st.session_state.chief_complaint and st.session_state.symptoms and st.session_state.symptom_duration:
            try:
                # Submit form to API - send as JSON body
                response = requests.post(
                    f"{API_URL}/api/patient/form",
                    json={
                        "patient_id": st.session_state.patient_id,
                        "chief_complaint": st.session_state.chief_complaint,
                        "symptoms": st.session_state.symptoms,
                        "symptom_duration": st.session_state.symptom_duration,
                        "severity": st.session_state.severity,
                        "medical_history": st.session_state.medical_history if st.session_state.medical_history else None
                    }
                )
                
                if response.status_code == 200:
                    st.success("✅ Pre-consultation form submitted successfully!")
                    # Start session - send as JSON body
                    session_response = requests.post(
                        f"{API_URL}/api/session/start",
                        json={
                            "patient_id": st.session_state.patient_id,
                            "chief_complaint": st.session_state.chief_complaint,
                            "symptoms": st.session_state.symptoms,
                            "symptom_duration": st.session_state.symptom_duration,
                            "severity": st.session_state.severity
                        }
                    )
                    
                    if session_response.status_code == 200:
                        session_data = session_response.json()
                        st.session_state.session_id = session_data.get("session_id")
                        st.session_state.current_step = "vision"
                        st.rerun()
                    else:
                        st.error(f"Failed to start session: {session_response.text}")
                else:
                    st.error(f"Failed to submit form: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please fill in all required fields (*)")
    
    return False


def show_vision_analysis():
    """Vision Analysis Section with Webcam"""
    st.header("📹 Vision Analysis")
    
    st.info("Please look at the camera for real-time vision analysis")
    
    # Initialize vision data in session state
    if 'vision_data' not in st.session_state:
        st.session_state.vision_data = None
    
    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}
    )
    
    # Create webrtc streamer
    webrtc_ctx = webrtc_streamer(
        key="vision-analysis",
        video_transformer_factory=FaceCaptureTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📸 Capture & Analyze", type="primary", use_container_width=True):
            if webrtc_ctx.video_transformer:
                frame = webrtc_ctx.video_transformer.captured_frame
                
                if frame is not None:
                    # Convert to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send to vision API
                    try:
                        response = requests.post(
                            f"{API_URL}/api/vision/analyze-base64",
                            json={"image": image_base64}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.vision_data = {
                                "emotion": data.get("emotion", {}).get("primary", "neutral"),
                                "emotion_confidence": data.get("emotion", {}).get("confidence", 0.0),
                                "blink_rate": data.get("eyes", {}).get("blink_rate", 0.0),
                                "eye_strain_score": data.get("eyes", {}).get("strain_score", 0.0),
                                "lip_tension": data.get("lips", {}).get("tension_score", 0.0),
                                "is_speaking": data.get("lips", {}).get("is_speaking", False)
                            }
                            st.success("✅ Vision analysis complete!")
                        else:
                            st.error("Vision analysis failed. Please try again.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please wait for the camera to initialize")
            else:
                st.warning("Please start the camera first")
    
    with col2:
        if st.button("Skip Vision Analysis", use_container_width=True):
            st.session_state.vision_data = {
                "emotion": "neutral",
                "emotion_confidence": 0.5,
                "blink_rate": 15.0,
                "eye_strain_score": 0.2,
                "lip_tension": 0.3,
                "is_speaking": False
            }
            st.rerun()
    
    # Show captured vision data
    if st.session_state.vision_data:
        st.markdown("### 📊 Captured Vision Data")
        v_data = st.session_state.vision_data
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Emotion", v_data.get("emotion", "N/A"))
        with col2:
            st.metric("Blink Rate", f"{v_data.get('blink_rate', 0):.1f} /min")
        with col3:
            st.metric("Eye Strain", f"{v_data.get('eye_strain_score', 0):.2f}")
        
        if st.button("Continue to Speech Analysis ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = "speech"
            st.rerun()
    
    return False


def show_speech_analysis():
    """Speech Analysis Section"""
    st.header("🎤 Speech Analysis")
    
    st.info("Upload an audio file describing your symptoms for analysis")
    
    # Initialize speech data in session state
    if 'speech_data' not in st.session_state:
        st.session_state.speech_data = None
    
    # Audio file uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg'],
        help="Supported formats: MP3, WAV, M4A, FLAC, OGG"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file is not None:
            st.audio(uploaded_file)
            st.write(f"File: {uploaded_file.name}")
            
            if st.button("🎤 Process Audio", type="primary", use_container_width=True):
                try:
                    # Save to temp file and send to API
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    response = requests.post(
                        f"{API_URL}/api/speech/process-audio",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success"):
                            st.session_state.speech_data = {
                                "transcript": data.get("transcript", ""),
                                "sentiment": data.get("sentiment", "neutral"),
                                "sentiment_score": data.get("confidence", 0.5),
                                "language": data.get("language", "en"),
                                "duration": data.get("duration", 0)
                            }
                            st.success("✅ Speech analysis complete!")
                        else:
                            st.error(f"Processing failed: {data.get('errors', 'Unknown error')}")
                    else:
                        st.error("Failed to process audio")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("Skip Speech Analysis", use_container_width=True):
            st.session_state.speech_data = {
                "transcript": "",
                "sentiment": "neutral",
                "sentiment_score": 0.5,
                "language": "en",
                "duration": 0
            }
            st.rerun()
    
    # Show captured speech data
    if st.session_state.speech_data:
        st.markdown("### 📊 Captured Speech Data")
        s_data = st.session_state.speech_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", s_data.get("sentiment", "N/A").upper())
            st.metric("Language", s_data.get("language", "N/A").upper())
        with col2:
            st.metric("Confidence", f"{s_data.get('sentiment_score', 0):.2f}")
            st.metric("Duration", f"{s_data.get('duration', 0):.1f}s")
        
        if s_data.get("transcript"):
            st.markdown("### 📝 Transcript")
            st.info(s_data["transcript"])
        
        if st.button("Run AI Analysis 🤖", type="primary", use_container_width=True):
            st.session_state.current_step = "analysis"
            st.rerun()
    
    return False


def show_analysis_results():
    """Show AI Analysis Results"""
    st.header("🧠 AI Analysis Results")
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Run analysis button
    if not st.session_state.analysis_results:
        if st.button("🔍 Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("Running AI analysis... This may take a minute."):
                try:
                    vision_data = st.session_state.get("vision_data", {})
                    speech_data = st.session_state.get("speech_data", {})
                    
                    response = requests.post(
                        f"{API_URL}/api/session/analyze",
                        params={
                            "session_id": st.session_state.session_id
                        },
                        json={
                            "vision_data": vision_data,
                            "speech_data": speech_data
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.analysis_results = data
                        st.session_state.current_step = "results"
                        st.rerun()
                    else:
                        st.error(f"Analysis failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Display results if available
    if st.session_state.analysis_results:
        data = st.session_state.analysis_results
        analysis = data.get("analysis", {})
        
        st.success("✅ Analysis Complete!")
        
        # Condition Prediction
        st.markdown("## 🔬 Condition Prediction")
        condition = analysis.get("condition", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Condition", condition.get("condition", "N/A"))
        with col2:
            st.metric("Confidence", f"{condition.get('confidence', 0)*100:.1f}%")
        
        if condition.get("explanation"):
            st.markdown("### Explanation")
            st.info(condition["explanation"])
        
        # Medication Suggestions
        st.markdown("## 💊 Medication Suggestions")
        medication = analysis.get("medication", {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Medication", medication.get("medication", "N/A"))
        with col2:
            st.metric("Dosage", medication.get("dosage", "N/A"))
        with col3:
            if medication.get("warnings"):
                st.warning("⚠️ Warnings Available")
            else:
                st.success("✅ No Warnings")
        
        if medication.get("warnings"):
            st.markdown("### ⚠️ Warnings")
            for warning in medication.get("warnings", []):
                st.warning(warning)
        
        # Safety Check
        st.markdown("## 🛡️ Safety Check")
        safety = analysis.get("safety", {})
        
        if safety.get("is_safe"):
            st.success("✅ Safety Check Passed")
        else:
            st.error("❌ Safety Check Failed")
        
        if safety.get("warnings"):
            for warn in safety.get("warnings", []):
                st.warning(warn)
        
        # Vision Analysis Summary
        if analysis.get("vision"):
            st.markdown("## 👁️ Vision Analysis Summary")
            vision = analysis.get("vision", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Emotion", vision.get("emotion", "N/A"))
            with col2:
                st.metric("Blink Rate", f"{vision.get('blink_rate', 0):.1f} /min")
            with col3:
                st.metric("Eye Strain", vision.get("eye_strain_level", "N/A"))
        
        # Speech Analysis Summary
        if analysis.get("speech"):
            st.markdown("## 🗣️ Speech Analysis Summary")
            speech = analysis.get("speech", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", speech.get("sentiment", "N/A").upper())
            with col2:
                st.metric("Confidence", f"{speech.get('sentiment_score', 0)*100:.1f}%")
        
        # Similar Cases
        if analysis.get("similar_cases"):
            st.markdown("## 📚 Similar Past Cases")
            similar = analysis.get("similar_cases", [])
            for i, case in enumerate(similar[:5], 1):
                with st.expander(f"Case {i}: {case.get('condition', 'Unknown')}"):
                    st.write(f"**Symptoms:** {case.get('symptoms', [])}")
                    st.write(f"Condition: {case.get('condition', 'N/A')}")
                    st.write(f"**Medication:** {case.get('medication', 'N/A')}")
        
        # Final Recommendation
        st.markdown("## 📌 Final Recommendation")
        recommendation = data.get("recommendation", "No recommendation available")
        st.info(recommendation)
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Generate Report", type="primary", use_container_width=True):
                try:
                    report_response = requests.get(
                        f"{API_URL}/api/report/generate/{st.session_state.session_id}"
                    )
                    if report_response.status_code == 200:
                        report_data = report_response.json()
                        st.session_state.report_path = report_data.get("report_path")
                        st.success(f"✅ Report generated: {st.session_state.report_path}")
                    else:
                        st.error("Failed to generate report")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("📧 Email Report", use_container_width=True):
                email = st.text_input("Enter your email address")
                if email and st.button("Send", type="primary"):
                    try:
                        email_response = requests.post(
                            f"{API_URL}/api/report/email/{st.session_state.session_id}",
                            json={"recipient_email": email}
                        )
                        if email_response.status_code == 200:
                            st.success("✅ Report sent to your email!")
                        else:
                            st.error("Failed to send email")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    return False


def show_session_history():
    """Show session history"""
    st.header("📜 Session History")
    
    # Button to start new consultation
    if st.button("➕ Start New Consultation", type="primary", use_container_width=True):
        # Reset session state
        st.session_state.current_step = "form"
        st.session_state.session_id = None
        st.session_state.chief_complaint = ""
        st.session_state.symptoms = []
        st.session_state.symptom_duration = ""
        st.session_state.severity = "mild"
        st.session_state.medical_history = ""
        st.session_state.vision_data = None
        st.session_state.speech_data = None
        st.session_state.analysis_results = None
        st.rerun()
    
    st.info("Session history will be displayed here")
    return False


def show_main_application():
    """Main Application after authentication"""
    
    # Initialize step tracking
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "form"
    
    # Progress indicator
    steps = ["form", "vision", "speech", "analysis", "results"]
    step_labels = ["📋 Form", "📹 Vision", "🎤 Speech", "🧠 Analysis", "📊 Results"]
    current_index = steps.index(st.session_state.current_step) if st.session_state.current_step in steps else 0
    
    # Show progress
    st.progress((current_index + 1) / len(steps))
    
    # Step indicator
    cols = st.columns(len(step_labels))
    for i, (step, label) in enumerate(zip(steps, step_labels)):
        with cols[i]:
            if i < current_index:
                st.success(label.split()[0] + " ✅")
            elif i == current_index:
                st.info(label.split()[0] + " 🔄")
            else:
                st.write(label.split()[0])
    
    st.markdown("---")
    
    # Show appropriate section based on current step
    if st.session_state.current_step == "form":
        show_pre_consultation_form()
    elif st.session_state.current_step == "vision":
        show_vision_analysis()
    elif st.session_state.current_step == "speech":
        show_speech_analysis()
    elif st.session_state.current_step in ["analysis", "results"]:
        show_analysis_results()


def logout():
    """Logout section"""
    st.header("🚪 Logout")
    
    if st.button("Logout", type="primary"):
        # Clear all session state
        st.session_state.authenticated = False
        st.session_state.patient_id = None
        st.session_state.patient_name = None
        st.session_state.auth_mode = "login"
        st.session_state.current_step = "form"
        st.session_state.session_id = None
        st.session_state.chief_complaint = ""
        st.session_state.symptoms = []
        st.session_state.symptom_duration = ""
        st.session_state.severity = "mild"
        st.session_state.medical_history = ""
        st.session_state.vision_data = None
        st.session_state.speech_data = None
        st.session_state.analysis_results = None
        st.rerun()


def main():
    """Main application"""
    st.set_page_config(
        page_title="Vision Based Model - Authentication",
        page_icon="🏥",
        layout="wide"
    )
    
    # Sidebar
    st.sidebar.title("🏥 Vision Based Model")
    st.sidebar.markdown("---")
    
    if st.session_state.authenticated:
        st.sidebar.success(f"Logged in as: {st.session_state.get('patient_name', 'User')}")
        st.sidebar.markdown("---")
        
        # Navigation
        st.sidebar.markdown("### Navigation")
        if st.sidebar.button("🏠 Home / Start"):
            if 'current_step' in st.session_state:
                st.session_state.current_step = "form"
            st.rerun()
        
        if st.sidebar.button("📜 Session History"):
            if 'current_step' in st.session_state:
                st.session_state.current_step = "history"
            st.rerun()
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout"):
            logout()
    else:
        st.sidebar.info("Please login to continue")
    
    # Main content
    st.title("🏥 Vision Based Model Authentication")
    
    if st.session_state.authenticated:
        # Show main app after authentication
        st.success(f"Welcome, {st.session_state.get('patient_name', 'User')}!")
        
        # Check if we should show main app or specific step
        if st.session_state.get('current_step') == "history":
            show_session_history()
        else:
            show_main_application()
        
    else:
        # Show authentication options
        if st.session_state.auth_mode == "login":
            # Tabs for login methods
            tab1, tab2 = st.tabs(["🔐 Face Login", "🔑 Token Login"])
            
            with tab1:
                login_with_face()
            
            with tab2:
                login_with_token()
                
        elif st.session_state.auth_mode == "token_login":
            login_with_token()
            
        elif st.session_state.auth_mode == "register":
            register_form()
            
        elif st.session_state.auth_mode == "face_register":
            register_face()


if __name__ == "__main__":
    main()
