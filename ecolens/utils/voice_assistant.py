"""
Voice assistant utilities.

IMPORTANT CAVEAT:
sr.Microphone() captures audio from the microphone of the machine RUNNING
`streamlit run app.py`. That's fine for local use (your own PC), but it will
NOT work if this app is deployed to a remote/cloud server, because the server
has no access to your browser's microphone. For a cloud deployment you'd need
a browser-side mic component instead (e.g. streamlit-webrtc or
streamlit-mic-recorder), which streams audio from the browser to the server.

Requires: pip install SpeechRecognition pyaudio pyttsx3
  - pyaudio can be finicky on Windows; if `pip install pyaudio` fails, try:
    pip install pipwin && pipwin install pyaudio
"""

import streamlit as st


def listen_from_microphone(timeout=5, phrase_time_limit=8):
    """
    Records from the local microphone and returns transcribed text.
    Returns (text, error) — one of them will be None.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        return None, "SpeechRecognition isn't installed. Run: pip install SpeechRecognition pyaudio"

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except OSError:
        return None, "No microphone found. Voice input only works when running this app locally."
    except Exception as e:
        return None, f"Couldn't access the microphone: {e}"

    try:
        text = recognizer.recognize_google(audio)
        return text, None
    except sr.UnknownValueError:
        return None, "Couldn't understand the audio — try again a bit slower."
    except sr.RequestError as e:
        return None, f"Speech recognition service error: {e}"


def speak_text(text):
    """
    Speaks text aloud using offline TTS (pyttsx3). Silently no-ops if
    pyttsx3 isn't installed, so this never blocks the rest of the app.
    """
    try:
        import pyttsx3
    except ImportError:
        st.info("Install pyttsx3 (`pip install pyttsx3`) to enable spoken responses.")
        return

    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"Text-to-speech failed: {e}")


def route_voice_command(text, search_fn):
    """
    Very small intent router: interprets a transcribed voice command and
    returns a text response. Extend this with more intents as needed
    (e.g. 'analyze this image', 'switch to camera mode').
    """
    text_lower = text.lower().strip()

    if text_lower.startswith(("search for", "look up", "tell me about", "what is")):
        for prefix in ["search for", "look up", "tell me about", "what is"]:
            if text_lower.startswith(prefix):
                query = text_lower[len(prefix):].strip()
                break
        result = search_fn(query)
        if result["results"]:
            top = result["results"][0]
            return top.get("summary", "I found a match but no summary is available.")
        return f"I couldn't find anything about '{query}'."

    return "I heard you, but I don't have a command for that yet. Try: 'tell me about the Bengal tiger'."
