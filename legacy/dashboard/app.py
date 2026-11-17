"""
Border AI — AI-Driven Border Intelligence & Multi-Sector Surveillance System
(Bharat Shield // Western, Northern, Eastern, and Southern Frontier Sectors)
"""

from __future__ import annotations

import os
import sys
import glob
import json
import time
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import cv2

# Append workspace root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

st.set_page_config(
    page_title="Border AI | Bharat Shield Command Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# HIGH-FIDELITY DARK COMMAND-CENTER CSS (HIGH CONTRAST & LEGIBILITY)
# -------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
  --bg-primary: #07090e;
  --bg-secondary: #0d121c;
  --bg-card: #111827;
  --border-card: #334155;
  --accent-orange: #ff8c00;
  --accent-amber: #ffa600;
  --accent-cyan: #00e5ff;
  --accent-mint: #10b981;
  --accent-red: #ef4444;
  --text-primary: #ffffff;
  --text-secondary: #e2e8f0;
  --text-muted: #cbd5e1;
}

html, body, [class*="css"], .stMarkdown, p, span, div, label, li {
  font-family: 'Inter', sans-serif !important;
  color: #f1f5f9 !important;
}

h1, h2, h3, h4, .tactical-title {
  font-family: 'Chakra Petch', sans-serif !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #ffffff !important;
}

code, .mono-font {
  font-family: 'JetBrains Mono', monospace !important;
  color: #00e5ff !important;
}

/* Base card styling */
.tactical-card {
  background: #111827;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 1.15rem;
  margin-bottom: 0.95rem;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.55);
}

.tactical-card-active {
  background: #1e293b;
  border: 1.5px solid #ff8c00;
  border-radius: 8px;
  padding: 1.15rem;
  margin-bottom: 0.95rem;
  box-shadow: 0 0 18px rgba(255, 140, 0, 0.3);
}

/* Header Banner */
.header-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.4rem 0 1rem 0;
  border-bottom: 1px solid #334155;
  margin-bottom: 1.2rem;
}

/* Badges */
.badge-demo {
  background: rgba(255, 140, 0, 0.2);
  border: 1.5px solid #ff8c00;
  color: #ffaa33 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.74rem;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 4px;
  letter-spacing: 0.08em;
}

.badge-simulated {
  background: rgba(239, 68, 68, 0.22);
  border: 1px solid #ef4444;
  color: #fca5a5 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 4px;
  letter-spacing: 0.06em;
}

.badge-stable {
  color: #10b981 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem;
  font-weight: 600;
}

.badge-degraded {
  color: #f59e0b !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem;
  font-weight: 600;
}

.badge-critical {
  color: #ef4444 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem;
  font-weight: 700;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
  background-color: #090d16 !important;
  border-right: 1px solid #1e293b;
}

[data-testid="stSidebar"] hr {
  border-color: #1e293b;
  margin: 0.8rem 0;
}

[data-testid="stSidebar"] .stRadio label {
  color: #e2e8f0 !important;
  font-weight: 500;
  font-size: 0.93rem;
  padding: 4px 8px;
  border-radius: 6px;
  transition: all 0.2s ease;
}

[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(255, 140, 0, 0.12);
  color: #ffaa33 !important;
}

/* Top Tricolour Line */
.top-tricolour-stripe {
  height: 3px;
  width: 100%;
  background: linear-gradient(90deg, #FF9933 0%, #FF9933 33.3%, #FFFFFF 33.3%, #FFFFFF 66.6%, #138808 66.6%, #138808 100%);
  border-radius: 2px;
  margin-bottom: 1.2rem;
}

/* Pulse animation */
@keyframes pulseDot {
  0% { transform: scale(0.95); opacity: 0.8; }
  50% { transform: scale(1.25); opacity: 1; filter: drop-shadow(0 0 6px #10b981); }
  100% { transform: scale(0.95); opacity: 0.8; }
}
.pulse-green {
  display: inline-block;
  width: 9px;
  height: 9px;
  background-color: #10b981;
  border-radius: 50%;
  margin-right: 8px;
  animation: pulseDot 2s infinite ease-in-out;
}
.pulse-red {
  display: inline-block;
  width: 9px;
  height: 9px;
  background-color: #ef4444;
  border-radius: 50%;
  margin-right: 8px;
  animation: pulseDot 1.4s infinite ease-in-out;
}

/* Camera Card Frame */
.cctv-card {
  background: #0d121c;
  border: 1px solid #334155;
  border-radius: 7px;
  padding: 10px;
  margin-bottom: 12px;
  position: relative;
}
.cctv-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: #f1f5f9;
}
.cctv-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.76rem;
  color: #cbd5e1;
}

/* Table High Contrast */
table {
  border-collapse: collapse;
  width: 100%;
}
th {
  background: #1e293b;
  color: #ffffff !important;
  font-weight: 700;
  padding: 8px 10px;
  border: 1px solid #334155;
}
td {
  padding: 8px 10px;
  border: 1px solid #334155;
  color: #f1f5f9 !important;
}

/* Buttons */
.stButton>button {
  background: #1e293b !important;
  color: #ffffff !important;
  border: 1px solid #475569 !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
}
.stButton>button:hover {
  background: #ff8c00 !important;
  color: #000000 !important;
  border-color: #ff8c00 !important;
  font-weight: 700 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------
# SECTOR METADATA & ASSETS (AUTHENTIC BORDER STREAMS)
# -------------------------------------------------------------
SECTORS = {
    "Northern Sector": {
        "cam_id": "CAM-001",
        "region": "JAMMU & KASHMIR / PUNJAB FRONTIER",
        "status": "LINK STABLE",
        "res": "1080p",
        "quality_class": "badge-stable",
        "annotated_video": "data/detections/annotated_cam_001_jk.mp4",
        "raw_video": "data/sample_videos/cam_001_jk.mp4",
        "snapshot": "dashboard/assets/cams/cam_001_jk.jpg",
        "threat_level": "ELEVATED",
        "threat_badge": "badge-stable",
        "alerts_count": 23,
        "detections_active": ["PERSON (96.4%)", "PERSON (91.2%)", "PATROL FLANK"],
        "coords": [33.7782, 76.5762],
        "terrain": "Alpine Ridge / Perimeter Fence",
        "threat_desc": "High-altitude security corridor. Perimeter sensors detected 23 boundary events."
    },
    "Western Sector": {
        "cam_id": "CAM-002",
        "region": "RAJASTHAN BORDER",
        "status": "LINK STABLE",
        "res": "1080p",
        "quality_class": "badge-stable",
        "annotated_video": "data/detections/annotated_cam_002_rajasthan.mp4",
        "raw_video": "data/sample_videos/cam_002_rajasthan.mp4",
        "snapshot": "dashboard/assets/cams/cam_002_rajasthan.jpg",
        "threat_level": "CRITICAL (ACTIVE ALERT)",
        "threat_badge": "badge-critical",
        "alerts_count": 2,
        "detections_active": ["HUMAN (88.7%)", "PROXIMITY BREACH", "RESTRICTED ZONE"],
        "coords": [28.0229, 73.3119],
        "terrain": "Desert Dunes / Barbed Fence",
        "threat_desc": "ACTIVE INTRUSION ALERT: Movement within 15 meters of desert barbed-wire boundary."
    },
    "Eastern Sector": {
        "cam_id": "CAM-003",
        "region": "NORTHEAST BORDER",
        "status": "LINK STABLE",
        "res": "1080p",
        "quality_class": "badge-stable",
        "annotated_video": "data/detections/annotated_cam_003_northeast.mp4",
        "raw_video": "data/sample_videos/cam_003_northeast.mp4",
        "snapshot": "dashboard/assets/cams/cam_003_northeast.jpg",
        "threat_level": "CRITICAL (ACTIVE ALERT)",
        "threat_badge": "badge-critical",
        "alerts_count": 0,
        "detections_active": ["VEHICLE (94.1%)", "RIVER TRANSIT"],
        "coords": [26.2006, 92.9376],
        "terrain": "Riverine / Forest Corridor",
        "threat_desc": "ACTIVE RADAR ALERT: Motorized transport transiting river perimeter access route."
    },
    "Southern Sector": {
        "cam_id": "CAM-004",
        "region": "GUJARAT BORDER",
        "status": "DEGRADED",
        "res": "480p",
        "quality_class": "badge-degraded",
        "annotated_video": "data/detections/annotated_cam_004_gujarat.mp4",
        "raw_video": "data/sample_videos/cam_004_gujarat.mp4",
        "snapshot": "dashboard/assets/cams/cam_004_gujarat.jpg",
        "threat_level": "NORMAL",
        "threat_badge": "badge-stable",
        "alerts_count": 17,
        "detections_active": ["PERIMETER SECURE", "OPTICAL DEGRADED"],
        "coords": [23.7337, 69.8597],
        "terrain": "Rann of Kutch Salt Marsh",
        "threat_desc": "Boundary fence physical link intact. Optical stream degraded by coastal salt mist (480p)."
    }
}

# -------------------------------------------------------------
# SIDEBAR NAVIGATION (MATCHING ATTACHED IMAGE 2)
# -------------------------------------------------------------
with st.sidebar:
    # Header Branding
    st.markdown("""
    <div style='display:flex; align-items:center; gap:12px; margin-bottom:1.2rem; padding: 6px 0;'>
      <div style='width:42px; height:42px; border-radius:10px; background:rgba(255,140,0,0.14); border:2px solid #ff8c00; display:flex; align-items:center; justify-content:center;'>
        <span style='font-size:22px;'>🛡️</span>
      </div>
      <div>
        <div style='font-family:"Chakra Petch", sans-serif; font-size:1.2rem; font-weight:700; color:#ffffff; letter-spacing:0.06em;'>BHARAT SHIELD</div>
        <div style='font-family:"JetBrains Mono", monospace; font-size:0.68rem; color:#00e5ff; letter-spacing:0.08em; font-weight:700;'>AI BORDER INTELLIGENCE</div>
      </div>
    </div>
    <div class='top-tricolour-stripe'></div>
    """, unsafe_allow_html=True)

    # Navigation menu
    nav_options = [
        "Command Overview",
        "Live Surveillance",
        "AI Threat Detection",
        "Border Map",
        "Intrusion Analytics",
        "Camera Management",
        "Detection History",
        "Reports & Insights",
        "System Settings",
    ]

    # Pre-select based on session state
    default_idx = 0
    if "nav_page" in st.session_state and st.session_state["nav_page"] in nav_options:
        default_idx = nav_options.index(st.session_state["nav_page"])

    page = st.radio(
        "NAVIGATION",
        nav_options,
        index=default_idx,
        label_visibility="collapsed",
    )
    st.session_state["nav_page"] = page

    st.markdown("<hr>", unsafe_allow_html=True)

    # Bottom status matching image 2
    now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%H:%M:%S IST")
    st.markdown(f"""
    <div style='padding: 6px 0; font-family:"Inter", sans-serif; font-size:0.86rem;'>
      <div style='display:flex; align-items:center; margin-bottom:7px; color:#10b981; font-weight:600;'>
        <span class='pulse-green'></span> System Operational
      </div>
      <div style='display:flex; align-items:center; margin-bottom:14px; color:#10b981; font-weight:600;'>
        <span class='pulse-green'></span> AI Engine Active
      </div>
      <div style='font-family:"JetBrains Mono", monospace; font-size:0.72rem; color:#cbd5e1; letter-spacing:0.08em;'>
        LAST SYNC &middot; <span style='color:#ffffff; font-weight:700;'>{now_ist}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# HELPER: TOP BAR FOR PAGES
# -------------------------------------------------------------
def render_page_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class='header-container'>
      <div>
        <h2 style='margin:0; font-size:1.65rem; color:#ffffff;'>{title}</h2>
        <div style='font-size:0.88rem; color:#cbd5e1; margin-top:3px;'>{subtitle}</div>
      </div>
      <div style='display:flex; gap:10px; align-items:center;'>
        <span class='badge-demo'>DEMO DATA</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# INTERACTIVE LEAFLET MAP OF INDIA GENERATOR
# -------------------------------------------------------------
def render_interactive_india_map(height: int = 460):
    """Renders a fully interactive Leaflet dark-mode map of India with sector markers and popups."""
    leaflet_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <style>
        body {{ margin:0; padding:0; background:#07090e; }}
        #map {{ width:100%; height:{height}px; background:#07090e; border-radius:8px; border:1px solid #334155; }}
        .leaflet-popup-content-wrapper {{
          background:#111827; color:#f1f5f9; border:1px solid #ff8c00; border-radius:6px;
          box-shadow: 0 4px 16px rgba(0,0,0,0.7);
        }}
        .leaflet-popup-tip {{ background:#111827; border:1px solid #ff8c00; }}
        .popup-title {{ font-family:sans-serif; font-weight:700; font-size:14px; color:#ffffff; margin-bottom:4px; }}
        .popup-desc {{ font-size:12px; color:#cbd5e1; margin-bottom:6px; }}
        .popup-status {{ font-size:11px; font-weight:700; color:#ff8c00; }}
        .marker-alert {{
          background:#ef4444; width:16px; height:16px; border-radius:50%; border:2px solid #ffffff;
          box-shadow:0 0 12px #ef4444; animation: alertPulse 1.2s infinite;
        }}
        .marker-cam {{
          background:#00e5ff; width:14px; height:14px; border-radius:50%; border:2px solid #ffffff;
          box-shadow:0 0 8px #00e5ff;
        }}
        .marker-station {{
          background:#10b981; width:14px; height:14px; border-radius:3px; border:2px solid #ffffff;
          box-shadow:0 0 8px #10b981;
        }}
        @keyframes alertPulse {{
          0% {{ transform: scale(0.9); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }}
          70% {{ transform: scale(1.3); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }}
          100% {{ transform: scale(0.9); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
        }}
        .map-legend {{
          position:absolute; bottom:12px; left:12px; z-index:1000;
          background:rgba(17, 24, 39, 0.92); border:1px solid #334155; border-radius:6px;
          padding:6px 12px; font-family:sans-serif; font-size:11px; color:#cbd5e1; display:flex; gap:14px; align-items:center;
        }}
      </style>
    </head>
    <body>
      <div id="map"></div>
      <div class="map-legend">
        <span><span style="color:#00e5ff; font-size:14px;">●</span> Camera</span>
        <span><span style="color:#10b981; font-size:14px;">■</span> Surveillance station</span>
        <span><span style="color:#ef4444; font-size:14px;">●</span> Active alert</span>
        <span style="color:#ffaa33; font-weight:700; letter-spacing:0.05em;">ILLUSTRATIVE POSITIONS — TACTICAL RADAR GRID</span>
      </div>
      <script>
        var map = L.map('map', {{
          center: [27.0, 79.0],
          zoom: 5,
          minZoom: 4,
          maxZoom: 9
        }});

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
          attribution: '&copy; OpenStreetMap &copy; CARTO',
          subdomains: 'abcd',
          maxZoom: 19
        }}).addTo(map);

        var sectors = [
          {{
            name: "PUNJAB BORDER",
            lat: 31.63, lon: 74.87,
            type: "normal",
            id: "CAM-001",
            desc: "Northern Sector Frontier & Perimeter Corridor"
          }},
          {{
            name: "RAJASTHAN BORDER",
            lat: 28.02, lon: 73.31,
            type: "alert",
            id: "CAM-002",
            desc: "Western Sector Barbed Fence — CRITICAL ALERT: Intrusion Breach"
          }},
          {{
            name: "GUJARAT BORDER",
            lat: 23.73, lon: 69.86,
            type: "normal",
            id: "CAM-004",
            desc: "Southern Sector Rann of Kutch Salt Perimeter"
          }},
          {{
            name: "NORTHEAST BORDER",
            lat: 26.20, lon: 92.94,
            type: "alert",
            id: "CAM-003",
            desc: "Eastern Sector River Corridor — ACTIVE ALERT: Motorized Movement"
          }}
        ];

        sectors.forEach(function(s) {{
          var iconHtml = s.type === "alert" 
            ? '<div class="marker-alert"></div>' 
            : '<div class="marker-cam"></div>';
            
          var customIcon = L.divIcon({{
            className: 'custom-pin',
            html: iconHtml,
            iconSize: [16, 16],
            iconAnchor: [8, 8]
          }});

          var stationIcon = L.divIcon({{
            className: 'custom-station',
            html: '<div class="marker-station"></div>',
            iconSize: [14, 14],
            iconAnchor: [7, 7]
          }});

          // Marker & Station
          L.marker([s.lat, s.lon], {{icon: customIcon}}).addTo(map)
            .bindPopup('<div class="popup-title">' + s.name + ' (' + s.id + ')</div><div class="popup-desc">' + s.desc + '</div><div class="popup-status">COORDINATES: ' + s.lat + '° N, ' + s.lon + '° E</div>');

          L.marker([s.lat + 0.3, s.lon + 0.3], {{icon: stationIcon}}).addTo(map)
            .bindPopup('<div class="popup-title">SURVEILLANCE STATION: ' + s.name + '</div><div class="popup-desc">Command Post & Radar Outpost</div>');
        }});
      </script>
    </body>
    </html>
    """
    components.html(leaflet_html, height=height + 10)


# =============================================================
# 1. COMMAND OVERVIEW
# =============================================================
if page == "Command Overview":
    render_page_header("Command Overview", "Pan-India Automated Border Defense & Sector Surveillance Matrix")

    # Top KPI Metrics Row (High Contrast)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown("""
        <div class='tactical-card'>
          <div style='font-family:"JetBrains Mono"; font-size:0.74rem; color:#cbd5e1; font-weight:600;'>ACTIVE CAMERAS</div>
          <div style='font-family:"Chakra Petch"; font-size:1.8rem; font-weight:700; color:#00e5ff;'>4 / 4</div>
          <div style='font-size:0.76rem; color:#10b981; font-weight:600;'>● All Streams Live</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class='tactical-card' style='border-color: rgba(239, 68, 68, 0.6);'>
          <div style='font-family:"JetBrains Mono"; font-size:0.74rem; color:#fca5a5; font-weight:600;'>ACTIVE ALERTS</div>
          <div style='font-family:"Chakra Petch"; font-size:1.8rem; font-weight:700; color:#ef4444;'>2 CRITICAL</div>
          <div style='font-size:0.76rem; color:#ffaa33; font-weight:600;'>Rajasthan &amp; Northeast</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class='tactical-card'>
          <div style='font-family:"JetBrains Mono"; font-size:0.74rem; color:#cbd5e1; font-weight:600;'>HUMAN DETECTIONS (24H)</div>
          <div style='font-family:"Chakra Petch"; font-size:1.8rem; font-weight:700; color:#ffffff;'>28</div>
          <div style='font-size:0.76rem; color:#10b981; font-weight:600;'>↑ +14% Night Trend</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class='tactical-card'>
          <div style='font-family:"JetBrains Mono"; font-size:0.74rem; color:#cbd5e1; font-weight:600;'>VEHICLE DETECTIONS (24H)</div>
          <div style='font-family:"Chakra Petch"; font-size:1.8rem; font-weight:700; color:#ffffff;'>14</div>
          <div style='font-size:0.76rem; color:#cbd5e1; font-weight:600;'>Logistics &amp; Armored</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown("""
        <div class='tactical-card'>
          <div style='font-family:"JetBrains Mono"; font-size:0.74rem; color:#cbd5e1; font-weight:600;'>AI INFERENCE LATENCY</div>
          <div style='font-family:"Chakra Petch"; font-size:1.8rem; font-weight:700; color:#10b981;'>52 ms</div>
          <div style='font-size:0.76rem; color:#10b981; font-weight:600;'>~19 FPS Realtime</div>
        </div>
        """, unsafe_allow_html=True)

    # 4-Camera Tactical Grid (Matching Image 3)
    st.markdown("<h4 style='color:#ff8c00; margin:1.2rem 0 0.8rem 0;'>📹 LIVE TACTICAL SECTOR FEEDS &middot; 4-WAY SURVEILLANCE MATRIX</h4>", unsafe_allow_html=True)
    c_row1_col1, c_row1_col2 = st.columns(2)
    c_row2_col1, c_row2_col2 = st.columns(2)

    col_map = [c_row1_col1, c_row1_col2, c_row2_col1, c_row2_col2]
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    for idx, (s_name, s_data) in enumerate(SECTORS.items()):
        with col_map[idx]:
            st.markdown(f"""
            <div class='cctv-card'>
              <div class='cctv-header'>
                <span><b style='color:#ffffff;'>{s_data['cam_id']}</b> | {s_name} &nbsp;<span style='color:#cbd5e1;'>// {s_data['region']}</span></span>
                <span class='badge-simulated'>&bull; SIMULATED</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Display real annotated video if exists, else fallback to real snapshot
            annotated_v = s_data.get("annotated_video")
            if annotated_v and os.path.exists(annotated_v):
                st.video(annotated_v)
            elif os.path.exists(s_data["snapshot"]):
                st.image(s_data["snapshot"], use_container_width=True)

            st.markdown(f"""
            <div class='cctv-footer'>
              <span class='{s_data['quality_class']}'>&bull; {s_data['status']} &middot; {s_data['res']}</span>
              <span style='color:#e2e8f0;'>{now_utc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#334155; margin:1.6rem 0;'>", unsafe_allow_html=True)

    # Interactive Working India Border Map Section
    st.markdown("<h4 style='color:#ff8c00;'>🗺️ BORDER GRID &middot; PAN-INDIA FRONTIER RADAR (CLICK SECTOR TO OPEN)</h4>", unsafe_allow_html=True)
    render_interactive_india_map(height=420)

    # Interactive Sector Open Buttons (Direct Jump to Live Camera Stream)
    st.markdown("<div style='margin-top:10px;'><b>QUICK OPEN SECTOR SURVEILLANCE &amp; TELEMETRY:</b></div>", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("📍 Open Northern Sector (Punjab/J&K)", use_container_width=True):
            st.session_state["nav_page"] = "Live Surveillance"
            st.session_state["selected_sector"] = "Northern Sector"
            st.rerun()
    with b2:
        if st.button("🚨 Open Western Sector (Rajasthan - ALERT)", use_container_width=True):
            st.session_state["nav_page"] = "Live Surveillance"
            st.session_state["selected_sector"] = "Western Sector"
            st.rerun()
    with b3:
        if st.button("🚨 Open Eastern Sector (Northeast - ALERT)", use_container_width=True):
            st.session_state["nav_page"] = "Live Surveillance"
            st.session_state["selected_sector"] = "Eastern Sector"
            st.rerun()
    with b4:
        if st.button("📍 Open Southern Sector (Gujarat)", use_container_width=True):
            st.session_state["nav_page"] = "Live Surveillance"
            st.session_state["selected_sector"] = "Southern Sector"
            st.rerun()


# =============================================================
# 2. LIVE SURVEILLANCE
# =============================================================
elif page == "Live Surveillance":
    render_page_header("Live Surveillance", "Real-Time Neural Detection & Video Stream Analytics across Frontier Corridors")

    # Sector Selector Dropdown
    curr_selected = st.session_state.get("selected_sector", "Northern Sector")
    sector_names = list(SECTORS.keys())
    s_idx = sector_names.index(curr_selected) if curr_selected in sector_names else 0

    c_sel, c_ctrl = st.columns([2.0, 1.0])
    with c_sel:
        active_sector_name = st.selectbox(
            "SELECT ACTIVE FRONTIER CAMERA SECTOR:",
            sector_names,
            index=s_idx,
        )
        st.session_state["selected_sector"] = active_sector_name
    with c_ctrl:
        view_mode = st.radio("STREAM MODE", ["AI Neural Overlay (YOLOv8)", "Raw Optical Feed"], horizontal=True)

    sec_info = SECTORS[active_sector_name]

    # Primary Video / Feed Display
    c_video, c_meta = st.columns([2.1, 1.0])
    with c_video:
        st.markdown(f"""
        <div class='cctv-card'>
          <div class='cctv-header'>
            <span><b style='color:#ffffff;'>{sec_info['cam_id']}</b> | {active_sector_name} &nbsp;<span style='color:#cbd5e1;'>// {sec_info['region']}</span></span>
            <span class='badge-simulated'>&bull; LIVE FEED</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Video stream playback
        if "Raw" in view_mode:
            v_path = sec_info.get("raw_video")
        else:
            v_path = sec_info.get("annotated_video") or sec_info.get("raw_video")

        if v_path and os.path.exists(v_path):
            st.video(v_path)
            st.caption(f"Broadcasting live 1080p feed from {sec_info['region']} sensor outpost ({sec_info['terrain']}).")
        elif os.path.exists(sec_info["snapshot"]):
            st.image(sec_info["snapshot"], use_container_width=True)

        # Tactical action buttons
        b1, b2 = st.columns(2)
        with b1:
            run_live_infer = st.button("⚡ Run Full Neural Inference on this Stream", use_container_width=True)
        with b2:
            snap_path = sec_info["snapshot"]
            if os.path.exists(snap_path):
                st.download_button(
                    "📥 Download Current Keyframe Snapshot",
                    data=open(snap_path, "rb").read(),
                    file_name=f"{sec_info['cam_id']}_snapshot.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

        if run_live_infer:
            with st.spinner(f"Executing YOLOv8 neural detection on {active_sector_name}..."):
                from src.pipeline import SurveillancePipeline
                pipe = SurveillancePipeline(model_path=os.path.join("models", "border_yolo.pt"), conf_thresh=0.20)
                out_sample_mp4 = os.path.join("data", "detections", f"{sec_info['cam_id']}_live_run.mp4")
                summary = pipe.run_video(sec_info["raw_video"], output_path=out_sample_mp4, max_frames=75)
                st.success(f"Inference Complete! Processed {summary['frames_processed']} frames @ {summary['average_fps']} FPS.")
                if os.path.exists(out_sample_mp4):
                    st.video(out_sample_mp4)

    with c_meta:
        st.markdown(f"""
        <div class='tactical-card'>
          <h4 style='color:#ff8c00; margin-top:0;'>STREAM TELEMETRY</h4>
          <table style='width:100%; font-size:0.86rem;'>
            <tr><td style='color:#cbd5e1; font-weight:600;'>Sector</td><td><b>{active_sector_name}</b></td></tr>
            <tr><td style='color:#cbd5e1; font-weight:600;'>Camera ID</td><td><b>{sec_info['cam_id']}</b></td></tr>
            <tr><td style='color:#cbd5e1; font-weight:600;'>Region</td><td><b>{sec_info['region']}</b></td></tr>
            <tr><td style='color:#cbd5e1; font-weight:600;'>Coordinates</td><td><code>{sec_info['coords'][0]:.4f} N, {sec_info['coords'][1]:.4f} E</code></td></tr>
            <tr><td style='color:#cbd5e1; font-weight:600;'>Stream Status</td><td><b class='{sec_info['quality_class']}'>{sec_info['status']} ({sec_info['res']})</b></td></tr>
            <tr><td style='color:#cbd5e1; font-weight:600;'>Threat Level</td><td><b style='color:#ef4444;'>{sec_info['threat_level']}</b></td></tr>
            <tr><td style='color:#cbd5e1; font-weight:600;'>Total Alerts</td><td><b style='color:#ffaa33;'>{sec_info['alerts_count']} Events</b></td></tr>
          </table>
          <hr style='border-color:#334155; margin:12px 0;'>
          <div style='font-size:0.80rem; color:#cbd5e1; font-weight:700;'>ACTIVE AI TARGETS:</div>
          <div style='margin-top:6px;'>
        """, unsafe_allow_html=True)
        for t in sec_info["detections_active"]:
            st.markdown(f"<span class='badge-demo' style='display:inline-block; margin-bottom:5px; margin-right:4px;'>{t}</span>", unsafe_allow_html=True)
        st.markdown(f"""
          </div>
          <hr style='border-color:#334155; margin:12px 0;'>
          <div style='font-size:0.84rem; color:#f1f5f9; line-height:1.4;'>{sec_info['threat_desc']}</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================
# 3. AI THREAT DETECTION
# =============================================================
elif page == "AI Threat Detection":
    render_page_header("AI Threat Detection", "Autonomous Object Detection, Armed Entity Linking & 3-Band Risk Stratification")

    c_cfg, c_res = st.columns([1.0, 1.8])
    with c_cfg:
        st.markdown("""
        <div class='tactical-card'>
          <h4 style='color:#ff8c00; margin-top:0;'>NEURAL CONTROLS</h4>
        """, unsafe_allow_html=True)
        conf = st.slider("Detection Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
        iou = st.slider("NMS IoU Threshold", 0.20, 0.80, 0.45, 0.05)
        enable_armed = st.checkbox("Enable Armed-Person Spatial Linking", value=True)
        enable_zones = st.checkbox("Enable 3-Band Spatial Risk Zones", value=True)
        st.markdown("""
          <hr style='border-color:#334155;'>
          <div style='font-size:0.80rem; color:#ffffff; font-weight:700;'>SPATIAL RISK CORRIDORS:</div>
          <div style='font-size:0.82rem; color:#ef4444; font-weight:600; margin-top:4px;'>&bull; Restricted Fence Zone (Y: 65% - 100%)</div>
          <div style='font-size:0.82rem; color:#f59e0b; font-weight:600;'>&bull; Tactical Buffer Zone (Y: 35% - 65%)</div>
          <div style='font-size:0.82rem; color:#10b981; font-weight:600;'>&bull; Observation Zone (Y: 0% - 35%)</div>
        </div>
        """, unsafe_allow_html=True)

    with c_res:
        st.markdown("<h4 style='color:#ff8c00;'>DETECTED ENTITIES &amp; SPATIAL RISK CLASSIFICATION</h4>", unsafe_allow_html=True)
        sample_img_path = "dashboard/assets/cams/cam_001_jk.jpg"
        if os.path.exists(sample_img_path):
            st.image(sample_img_path, caption="Live Detected Frame with YOLOv8 Bounding Boxes & Tactical Zones", use_container_width=True)

        st.markdown("""
        <div class='tactical-card'>
          <div style='display:flex; justify-content:space-between; align-items:center;'>
            <span style='color:#ef4444; font-weight:700; font-size:1.05rem;'>🚨 CRITICAL ALERT: HUMAN DETECTED IN TACTICAL BUFFER CORRIDOR</span>
            <span class='badge-demo'>PRIORITY 1</span>
          </div>
          <div style='font-size:0.84rem; color:#cbd5e1; margin-top:6px;'>
            Proximity heuristic and spatial boundary coordinates registered active target in high-risk frontier sector. Incident logged to persistent alerts log.
          </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================
# 4. BORDER MAP (MATCHING ATTACHED IMAGE 1)
# =============================================================
elif page == "Border Map":
    render_page_header("Border Map", "Interactive Pan-India Border Grid, Sensor Deployments & Active Alerts")

    st.markdown("""
    <div class='tactical-card' style='padding:14px; margin-bottom:14px;'>
      <div style='font-family:"JetBrains Mono"; font-size:0.84rem; color:#ffffff; letter-spacing:0.08em; font-weight:700; margin-bottom:10px;'>
        BORDER GRID &middot; PAN-INDIA FRONTIER (INTERACTIVE SENSOR MAP)
      </div>
    """, unsafe_allow_html=True)

    render_interactive_india_map(height=480)

    st.markdown("</div>", unsafe_allow_html=True)

    # Clickable Sector Stations (as requested: on clicking that zone that part should open)
    st.markdown("<h4 style='color:#ff8c00; margin-top:1.2rem;'>CLICK SECTOR TO OPEN LIVE SURVEILLANCE &amp; TELEMETRY:</h4>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown("""
        <div style='background:#111827; padding:12px; border-radius:7px; border:1px solid #334155;'>
          <div style='color:#10b981; font-size:0.78rem; font-weight:700;'>● CAMERA + STATION</div>
          <div style='font-family:"Chakra Petch"; font-size:1.15rem; color:#ffffff; font-weight:700;'>PUNJAB BORDER</div>
          <div style='font-size:0.76rem; color:#cbd5e1;'>Northern Frontier &middot; CAM-001</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect Punjab Sector", use_container_width=True):
            st.session_state["nav_page"] = "Live Surveillance"
            st.session_state["selected_sector"] = "Northern Sector"
            st.rerun()

    with m2:
        st.markdown("""
        <div style='background:#1f1318; padding:12px; border-radius:7px; border:1.5px solid #ef4444;'>
          <div style='color:#ef4444; font-size:0.78rem; font-weight:700;'>🚨 ACTIVE ALERT</div>
          <div style='font-family:"Chakra Petch"; font-size:1.15rem; color:#ffffff; font-weight:700;'>RAJASTHAN BORDER</div>
          <div style='font-size:0.76rem; color:#ffaa33;'>Western Frontier &middot; CAM-002</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect Rajasthan Sector", use_container_width=True):
            st.session_state["nav_page"] = "Live Surveillance"
            st.session_state["selected_sector"] = "Western Sector"
            st.rerun()

    with m3:
        st.markdown("""
        <div style='background:#111827; padding:12px; border-radius:7px; border:1px solid #334155;'>
          <div style='color:#10b981; font-size:0.78rem; font-weight:700;'>● CAMERA + STATION</div>
          <div style='font-family:"Chakra Petch"; font-size:1.15rem; color:#ffffff; font-weight:700;'>GUJARAT BORDER</div>
          <div style='font-size:0.76rem; color:#cbd5e1;'>Southern Frontier &middot; CAM-004</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect Gujarat Sector", use_container_width=True):
            st.session_state["nav_page"] = "Live Surveillance"
            st.session_state["selected_sector"] = "Southern Sector"
            st.rerun()

    with m4:
        st.markdown("""
        <div style='background:#1f1318; padding:12px; border-radius:7px; border:1.5px solid #ef4444;'>
          <div style='color:#ef4444; font-size:0.78rem; font-weight:700;'>🚨 ACTIVE ALERT</div>
          <div style='font-family:"Chakra Petch"; font-size:1.15rem; color:#ffffff; font-weight:700;'>NORTHEAST BORDER</div>
          <div style='font-size:0.76rem; color:#ffaa33;'>Eastern Frontier &middot; CAM-003</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect Northeast Sector", use_container_width=True):
            st.session_state["nav_page"] = "Live Surveillance"
            st.session_state["selected_sector"] = "Eastern Sector"
            st.rerun()


# =============================================================
# 5. INTRUSION ANALYTICS (MATCHING ATTACHED IMAGE 4)
# =============================================================
elif page == "Intrusion Analytics":
    # Top Status Bar matching Image 4
    st.markdown("""
    <div style='display:flex; justify-content:space-between; align-items:center; background:#111827; padding:10px 16px; border-radius:8px; border:1px solid #334155; margin-bottom:1.2rem;'>
      <div style='display:flex; align-items:center; gap:12px;'>
        <span style='font-size:1.1rem; color:#ffffff;'>☰</span>
        <span style='font-weight:700; color:#ffffff; font-size:0.95rem;'>Indian Border Surveillance System</span>
      </div>
      <div style='background:#1e293b; padding:6px 14px; border-radius:6px; color:#cbd5e1; font-size:0.82rem; border:1px solid #334155; width:280px;'>
        🔍 Search cameras, detections...
      </div>
      <div style='display:flex; align-items:center; gap:14px;'>
        <span style='color:#10b981;'>📶</span>
        <span style='color:#ffaa33;'>🔔</span>
        <span style='background:#ff8c00; color:#000000; font-weight:700; font-size:0.75rem; padding:3px 7px; border-radius:4px;'>CO</span>
        <span style='font-family:"JetBrains Mono"; font-size:0.78rem; color:#ffffff; font-weight:700;'>CLEARANCE L5</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    render_page_header("Intrusion Analytics", "Statistical Telemetry, Daily Detection Trends & Weekly AI Detection Breakdown")

    # CARD 1: DAILY DETECTION TREND (Smooth Glowing Line Chart)
    st.markdown("""
    <div class='tactical-card'>
      <div style='font-family:"JetBrains Mono"; font-size:0.78rem; color:#ffffff; letter-spacing:0.08em; font-weight:700; margin-bottom:10px;'>DAILY DETECTION TREND</div>
    """, unsafe_allow_html=True)

    hours = ["00:00", "02:00", "04:00", "06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00", "24:00"]
    orange_trend = [7.0, 9.5, 12.2, 8.5, 7.2, 5.0, 2.8, 6.0, 7.5, 7.5, 6.0, 8.5, 10.0]
    cyan_trend = [4.0, 4.0, 2.5, 1.2, 2.2, 3.0, 4.0, 4.0, 2.8, 2.0, 1.5, 2.5, 4.0]

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=hours, y=orange_trend, mode='lines', name='Human & Armed Incursions',
        line=dict(color='#ff8c00', width=3.5, shape='spline'),
        fill='tozeroy', fillcolor='rgba(255, 140, 0, 0.10)'
    ))
    fig_daily.add_trace(go.Scatter(
        x=hours, y=cyan_trend, mode='lines', name='Vehicular Transits',
        line=dict(color='#00e5ff', width=3.0, shape='spline'),
        fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.08)'
    ))
    fig_daily.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(family="JetBrains Mono", color="#cbd5e1", size=11),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False, range=[0, 15]),
        margin=dict(l=20, r=20, t=20, b=20),
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#ffffff"))
    )
    st.plotly_chart(fig_daily, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # CARD 2: WEEKLY AI DETECTION STATISTICS (Stacked Bar Chart)
    st.markdown("""
    <div class='tactical-card'>
      <div style='font-family:"JetBrains Mono"; font-size:0.78rem; color:#ffffff; letter-spacing:0.08em; font-weight:700; margin-bottom:10px;'>WEEKLY AI DETECTION STATISTICS</div>
    """, unsafe_allow_html=True)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    human_counts = [150, 160, 130, 170, 155, 180, 165]
    vehicle_counts = [45, 50, 40, 60, 48, 55, 52]
    armed_counts = [25, 20, 22, 28, 24, 26, 25]
    unverified_counts = [12, 10, 8, 14, 11, 13, 10]

    fig_weekly = go.Figure()
    fig_weekly.add_trace(go.Bar(x=days, y=human_counts, name='Human', marker_color='#ff8c00'))
    fig_weekly.add_trace(go.Bar(x=days, y=vehicle_counts, name='Vehicle', marker_color='#00e5ff'))
    fig_weekly.add_trace(go.Bar(x=days, y=armed_counts, name='Armed Alert', marker_color='#10b981'))
    fig_weekly.add_trace(go.Bar(x=days, y=unverified_counts, name='Unverified', marker_color='#fde047'))

    fig_weekly.update_layout(
        barmode='stack',
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(family="JetBrains Mono", color="#cbd5e1", size=11),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", range=[0, 300], tickvals=[0, 70, 140, 210, 280]),
        margin=dict(l=20, r=20, t=20, b=20),
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#ffffff"))
    )
    st.plotly_chart(fig_weekly, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# 6. CAMERA MANAGEMENT
# =============================================================
elif page == "Camera Management":
    render_page_header("Camera Management", "Sensor Inventory, RTSP Streams, Resolution & Optical Health Status")

    c_cards = st.columns(4)
    for idx, (s_name, s_data) in enumerate(SECTORS.items()):
        with c_cards[idx]:
            st.markdown(f"""
            <div class='tactical-card'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <b style='color:#ffffff; font-size:1.0rem;'>{s_data['cam_id']}</b>
                <span class='{s_data['quality_class']}'>&bull; {s_data['status']}</span>
              </div>
              <div style='font-family:"Chakra Petch"; font-size:1.15rem; color:#ffaa33; font-weight:700; margin:6px 0;'>{s_name}</div>
              <div style='font-size:0.80rem; color:#cbd5e1;'>Region: <b style='color:#ffffff;'>{s_data['region']}</b></div>
              <div style='font-size:0.80rem; color:#cbd5e1;'>Resolution: <b style='color:#ffffff;'>{s_data['res']}</b></div>
              <div style='font-size:0.80rem; color:#cbd5e1;'>Bitrate: <b style='color:#ffffff;'>4.2 Mbps</b></div>
              <div style='font-size:0.80rem; color:#cbd5e1;'>FPS: <b style='color:#ffffff;'>25 FPS</b></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<h4 style='color:#ff8c00; margin-top:1.2rem;'>FRONTIER SENSOR SPECIFICATIONS</h4>", unsafe_allow_html=True)
    df_cams = pd.DataFrame([
        {"Camera ID": "CAM-001", "Sector": "Northern Sector", "Terrain": "Alpine Ridge / Snow", "Sensor Type": "Long-Range Thermal + NVG", "FPS": 25, "Status": "ONLINE", "Resolution": "1080p"},
        {"Camera ID": "CAM-002", "Sector": "Western Sector", "Terrain": "Desert Dunes / Fence", "Sensor Type": "FLIR Cooled MWIR", "FPS": 25, "Status": "ONLINE", "Resolution": "1080p"},
        {"Camera ID": "CAM-003", "Sector": "Eastern Sector", "Terrain": "Riverine / Forest Corridor", "Sensor Type": "EO/IR Pan-Tilt-Zoom", "FPS": 25, "Status": "ONLINE", "Resolution": "1080p"},
        {"Camera ID": "CAM-004", "Sector": "Southern Sector", "Terrain": "Salt Marsh / Creek", "Sensor Type": "Coastal Low-Light SWIR", "FPS": 15, "Status": "DEGRADED (480p)", "Resolution": "480p"},
    ])
    st.dataframe(df_cams, use_container_width=True)


# =============================================================
# 7. DETECTION HISTORY
# =============================================================
elif page == "Detection History":
    render_page_header("Detection History", "Real-Time Log of Neural Detections, Spatial Zones & Historical Alerts")

    alerts_csv = os.path.join("data", "results", "alerts.csv")
    if os.path.exists(alerts_csv) and os.path.getsize(alerts_csv) > 20:
        df_alerts = pd.read_csv(alerts_csv)
    else:
        df_alerts = pd.DataFrame([
            {"timestamp": "2026-11-02 01:05:12", "sector": "Western Sector", "target_type": "ARMED PERSON", "zone": "Restricted Fence Zone", "priority": "CRITICAL", "confidence": 0.94},
            {"timestamp": "2026-11-02 00:52:40", "sector": "Northern Sector", "target_type": "PERSON", "zone": "Tactical Buffer Zone", "priority": "HIGH", "confidence": 0.91},
            {"timestamp": "2026-11-01 23:40:15", "sector": "Eastern Sector", "target_type": "VEHICLE", "zone": "Observation Zone", "priority": "LOW", "confidence": 0.88},
            {"timestamp": "2026-11-01 22:15:30", "sector": "Western Sector", "target_type": "ARMED PERSON", "zone": "Restricted Fence Zone", "priority": "CRITICAL", "confidence": 0.96},
            {"timestamp": "2026-11-01 20:10:05", "sector": "Southern Sector", "target_type": "PERSON", "zone": "Observation Zone", "priority": "LOW", "confidence": 0.82},
        ])

    st.dataframe(df_alerts, use_container_width=True)


# =============================================================
# 8. REPORTS & INSIGHTS (MATCHING ATTACHED IMAGE 5)
# =============================================================
elif page == "Reports & Insights":
    st.markdown("""
    <div style='margin-bottom:0.9rem;'>
      <div style='font-family:"JetBrains Mono"; font-size:0.76rem; color:#ff8c00; letter-spacing:0.1em; font-weight:700;'>INTELLIGENCE</div>
      <h2 style='margin:0; font-size:1.65rem; color:#ffffff;'>Reports &amp; Insights &nbsp;<span class='badge-demo'>DEMO DATA</span></h2>
    </div>
    """, unsafe_allow_html=True)

    # 3 SUMMARY INSIGHT CARDS (Matching Image 5)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div class='tactical-card'>
          <div style='color:#ff8c00; font-size:1.4rem; margin-bottom:6px;'>📈</div>
          <div style='font-family:"Inter"; font-weight:700; font-size:1.05rem; color:#ffffff; margin-bottom:6px;'>Night-time human detections up 14%</div>
          <div style='font-size:0.86rem; color:#cbd5e1;'>Western sector activity peaks between 01:00&ndash;04:00.</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown("""
        <div class='tactical-card'>
          <div style='color:#f59e0b; font-size:1.4rem; margin-bottom:6px;'>⚠️</div>
          <div style='font-family:"Inter"; font-weight:700; font-size:1.05rem; color:#ffffff; margin-bottom:6px;'>CAM-004 link degraded</div>
          <div style='font-size:0.86rem; color:#cbd5e1;'>Southern sector feed dropped to 480p; maintenance advised.</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown("""
        <div class='tactical-card'>
          <div style='color:#10b981; font-size:1.4rem; margin-bottom:6px;'>✅</div>
          <div style='font-family:"Inter"; font-weight:700; font-size:1.05rem; color:#ffffff; margin-bottom:6px;'>False-positive rate at 3.2%</div>
          <div style='font-size:0.86rem; color:#cbd5e1;'>Animal reclassification improved after model update.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # GENERATED REPORTS SECTION (Matching Image 5)
    st.markdown("""
    <div class='tactical-card'>
      <div style='font-family:"JetBrains Mono"; font-size:0.78rem; color:#ffffff; letter-spacing:0.08em; font-weight:700; margin-bottom:14px;'>GENERATED REPORTS</div>
    """, unsafe_allow_html=True)

    reports = [
        {"title": "Weekly Sector Summary — W44", "type": "PDF &middot; DEMO"},
        {"title": "Monthly Intrusion Report — October", "type": "PDF &middot; DEMO"},
        {"title": "Camera Health Audit — Q4", "type": "PDF &middot; DEMO"},
        {"title": "AI Model Performance Review", "type": "PDF &middot; DEMO"},
    ]

    for rep in reports:
        col_t, col_b = st.columns([3.0, 1.0])
        with col_t:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid #1e293b;'>
              <span style='color:#00e5ff; font-size:1.2rem;'>📄</span>
              <span style='font-size:0.95rem; color:#ffffff; font-weight:600;'>{rep['title']}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div style='text-align:right; padding:10px 0;'>
              <span class='badge-demo' style='color:#cbd5e1 !important; border-color:#475569;'>{rep['type']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# 9. SYSTEM SETTINGS
# =============================================================
elif page == "System Settings":
    render_page_header("System Settings", "Surveillance Network Configuration, Alert Rules & Operational Overrides")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("""
        <div class='tactical-card'>
          <h4 style='color:#ff8c00; margin-top:0;'>DETECTION SENSITIVITY &amp; RULES</h4>
        """, unsafe_allow_html=True)
        st.slider("Proximity Linking Radius (Pixels)", 20, 200, 65, 5)
        st.slider("Alert Cooldown Window (Seconds)", 1.0, 10.0, 3.0, 0.5)
        st.checkbox("Trigger Audio Siren on CRITICAL Threat", value=True)
        st.checkbox("Forward Incident Logs to Central Command via Webhook", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div class='tactical-card'>
          <h4 style='color:#ff8c00; margin-top:0;'>DEFCON LEVEL OVERRIDE</h4>
        """, unsafe_allow_html=True)
        defcon = st.select_slider(
            "Security Posture",
            options=["DEFCON 5 (Normal)", "DEFCON 4 (Guarded)", "DEFCON 3 (Elevated)", "DEFCON 2 (High)", "DEFCON 1 (Maximum Alert)"],
            value="DEFCON 2 (High)"
        )
        st.info(f"Active Posture: {defcon}. Automated QRT dispatch enabled for Restricted Zone intrusions.")
        st.markdown("</div>", unsafe_allow_html=True)
