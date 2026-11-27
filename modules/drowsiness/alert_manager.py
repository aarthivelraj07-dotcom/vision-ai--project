"""
Alert & Reporting Management for Task 3: Drowsiness Detection.
Provides:
- In-browser synthesized audio alert via HTML5 Web Audio API.
- Non-blocking Windows winsound hardware buzzer trigger.
- Visual warning banner HTML components.
- Downloadable CSV session report generation.
"""

import threading
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# HTML5 Web Audio Siren component that plays directly in the user's browser
BROWSER_AUDIO_SIREN_HTML = """
<script>
    (function() {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            const ctx = new AudioContext();
            
            // Oscillator for siren beep
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            
            osc.type = 'sawtooth';
            // Alternating two-tone pitch (880Hz to 1200Hz)
            osc.frequency.setValueAtTime(880, ctx.currentTime);
            osc.frequency.setValueAtTime(1200, ctx.currentTime + 0.15);
            osc.frequency.setValueAtTime(880, ctx.currentTime + 0.30);
            
            gain.gain.setValueAtTime(0.3, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.45);
            
            osc.connect(gain);
            gain.connect(ctx.destination);
            
            osc.start();
            osc.stop(ctx.currentTime + 0.45);
        } catch(e) {
            console.log("Audio alert playback error: ", e);
        }
    })();
</script>
"""


def trigger_host_buzzer(freq: int = 2200, duration_ms: int = 400):
    """Trigger Windows system beep in background non-blocking thread."""
    if not HAS_WINSOUND:
        return

    def _beep():
        try:
            winsound.Beep(freq, duration_ms)
        except Exception:
            pass

    t = threading.Thread(target=_beep, daemon=True)
    t.start()


def render_browser_audio_alert():
    """Inject client-side HTML5 Web Audio buzzer into Streamlit."""
    st.components.v1.html(BROWSER_AUDIO_SIREN_HTML, height=0, width=0)


def generate_drowsiness_csv_report(event_log: List[Dict[str, Any]]) -> str:
    """
    Format and export session event log into a standard downloadable CSV string.
    """
    if not event_log:
        df = pd.DataFrame(columns=[
            "timestamp", "event_type", "severity", "frames_duration", "ear", "mar", "details"
        ])
        return df.to_csv(index=False)

    df = pd.DataFrame(event_log)
    return df.to_csv(index=False)
