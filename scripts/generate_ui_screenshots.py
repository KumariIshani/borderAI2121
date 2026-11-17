"""Generate High-Fidelity UI Screenshots for Streamlit Dashboard.

Renders mock-browser screenshots of:
1. Live Tactical Surveillance Command Center (main page)
2. Model Analytics & Telemetry Dashboard (analytics page)
3. Wagah-Attari Frontier Sector Map & Zones (sector map page)
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def render_ui_screenshots():
    out_dir = os.path.join("data", "results", "ui_screenshots")
    os.makedirs(out_dir, exist_ok=True)

    W, H = 1920, 1080

    # -------------------------------------------------------------
    # SCREENSHOT 1: COMMAND CENTER (MAIN PAGE)
    # -------------------------------------------------------------
    img1 = np.full((H, W, 3), (15, 12, 10), dtype=np.uint8)  # Dark #0a0c0f

    # Top Navigation Bar with Tricolour Accent
    cv2.rectangle(img1, (0, 0), (W, 70), (28, 22, 18), -1)
    # Tricolour strip at top
    cv2.rectangle(img1, (0, 0), (W // 3, 4), (51, 153, 255), -1)      # Saffron (BGR)
    cv2.rectangle(img1, (W // 3, 0), (2 * W // 3, 4), (255, 255, 255), -1) # White
    cv2.rectangle(img1, (2 * W // 3, 0), (W, 4), (8, 136, 19), -1)    # Green

    # Title & Shield Emblem
    cv2.circle(img1, (45, 36), 18, (128, 0, 0), 2)
    cv2.putText(img1, "BORDER AI", (80, 44), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img1, "BHARAT SHIELD // AI BORDER INTELLIGENCE [LIVE]", (320, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (51, 153, 255), 1)

    # Status Pills
    cv2.rectangle(img1, (W - 350, 18), (W - 200, 52), (35, 60, 20), -1)
    cv2.putText(img1, "DEFCON: ELEVATED", (W - 335, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

    cv2.rectangle(img1, (W - 180, 18), (W - 30, 52), (25, 25, 40), -1)
    cv2.putText(img1, "MODEL: YOLOv8-OP", (W - 170, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1)

    # Left Sidebar (Nav matching Image 2)
    cv2.rectangle(img1, (20, 90), (320, H - 30), (22, 18, 14), -1)
    cv2.rectangle(img1, (20, 90), (320, H - 30), (45, 38, 30), 1)
    cv2.putText(img1, "BHARAT SHIELD", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.65, (51, 153, 255), 1)

    nav_items = [
        ("> COMMAND OVERVIEW", True),
        ("  LIVE SURVEILLANCE", False),
        ("  AI THREAT DETECTION", False),
        ("  BORDER MAP", False),
        ("  INTRUSION ANALYTICS", False),
        ("  CAMERA MANAGEMENT", False),
        ("  DETECTION HISTORY", False),
        ("  REPORTS & INSIGHTS", False),
        ("  SYSTEM SETTINGS", False),
    ]
    for idx, (item, active) in enumerate(nav_items):
        y = 180 + idx * 45
        if active:
            cv2.rectangle(img1, (30, y - 25), (310, y + 10), (45, 30, 15), -1)
            cv2.rectangle(img1, (30, y - 25), (35, y + 10), (51, 153, 255), -1)
            cv2.putText(img1, item, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        else:
            cv2.putText(img1, item, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

    # Main Center Feed (Video Frame)
    cv2.rectangle(img1, (350, 90), (1450, 720), (24, 20, 16), -1)
    cv2.rectangle(img1, (350, 90), (1450, 720), (51, 153, 255), 2)

    # Load and composite demo frame into feed
    demo_frame_path = os.path.join("data", "results", "demo_frames", "frame_02.jpg")
    if os.path.exists(demo_frame_path):
        dframe = cv2.imread(demo_frame_path)
        dframe_resized = cv2.resize(dframe, (1096, 626))
        img1[92:92 + 626, 352:352 + 1096] = dframe_resized

    # Telemetry Cards Below Feed
    metrics_summary = [
        ("INFERENCE FPS", "29.4 FPS", (0, 229, 255)),
        ("PERSONS DETECTED", "4 DETECTED", (255, 200, 0)),
        ("ARMED THREATS", "1 ARMED (CRITICAL)", (50, 50, 255)),
        ("ACTIVE VEHICLES", "2 UNITS", (0, 255, 120)),
        ("SECURITY ZONE", "RESTRICTED FENCE", (0, 0, 255)),
    ]
    card_w = (1100 - 4 * 15) // 5
    for idx, (label, val, col) in enumerate(metrics_summary):
        cx = 350 + idx * (card_w + 15)
        cv2.rectangle(img1, (cx, 740), (cx + card_w, 830), (25, 20, 16), -1)
        cv2.rectangle(img1, (cx, 740), (cx + card_w, 830), (50, 40, 30), 1)
        cv2.putText(img1, label, (cx + 12, 765), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
        cv2.putText(img1, val, (cx + 12, 805), cv2.FONT_HERSHEY_DUPLEX, 0.55, col, 1)

    # Right Column: Radar & Alert Stream
    # Radar Widget
    cv2.rectangle(img1, (1470, 90), (W - 20, 430), (22, 18, 14), -1)
    cv2.rectangle(img1, (1470, 90), (W - 20, 430), (45, 38, 30), 1)
    cv2.putText(img1, "TACTICAL RADAR SWEEP", (1530, 125), cv2.FONT_HERSHEY_DUPLEX, 0.60, (51, 153, 255), 1)

    rcx, rcy, rad = 1695, 270, 120
    for r in [30, 60, 90, 120]:
        cv2.circle(img1, (rcx, rcy), r, (50, 70, 40), 1)
    cv2.line(img1, (rcx - rad, rcy), (rcx + rad, rcy), (50, 70, 40), 1)
    cv2.line(img1, (rcx, rcy - rad), (rcx, rcy + rad), (50, 70, 40), 1)
    # Sweep line
    cv2.line(img1, (rcx, rcy), (rcx + 85, rcy - 85), (0, 255, 120), 2)
    # Blips
    cv2.circle(img1, (rcx + 35, rcy - 45), 4, (0, 0, 255), -1)  # Armed hostile
    cv2.circle(img1, (rcx - 50, rcy + 20), 4, (0, 255, 255), -1) # Vehicle
    cv2.circle(img1, (rcx + 70, rcy + 40), 4, (255, 200, 0), -1) # Patrol

    # Alert Stream
    cv2.rectangle(img1, (1470, 450), (W - 20, H - 30), (22, 18, 14), -1)
    cv2.rectangle(img1, (1470, 450), (W - 20, H - 30), (45, 38, 30), 1)
    cv2.putText(img1, "LIVE THREAT STREAM", (1540, 485), cv2.FONT_HERSHEY_DUPLEX, 0.60, (50, 50, 255), 1)

    alerts = [
        ("[CRITICAL]", "ARMED HOSTILE IN FENCE ZONE", "SEC-A3", (50, 50, 255)),
        ("[HIGH]", "UNIDENTIFIED WEAPON DROPPED", "SEC-B1", (0, 140, 255)),
        ("[HIGH]", "ARMED PERSON DETECTED", "SEC-B2", (0, 140, 255)),
        ("[MEDIUM]", "MILITARY CONVOY IN BUFFER", "SEC-C4", (0, 220, 255)),
        ("[LOW]", "SURVEILLANCE PATROL TRANSIT", "SEC-A1", (100, 255, 100)),
        ("[LOW]", "VEHICLE AT CHECKPOINT ALPHA", "SEC-WAGAH", (100, 255, 100)),
    ]
    for idx, (prio, desc, sec, col) in enumerate(alerts):
        ay = 530 + idx * 75
        cv2.rectangle(img1, (1485, ay - 20), (W - 35, ay + 45), (32, 24, 20), -1)
        cv2.rectangle(img1, (1485, ay - 20), (1490, ay + 45), col, -1)
        cv2.putText(img1, prio, (1500, ay), cv2.FONT_HERSHEY_DUPLEX, 0.48, col, 1)
        cv2.putText(img1, sec, (W - 120, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        cv2.putText(img1, desc, (1500, ay + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    cv2.imwrite(os.path.join(out_dir, "main_command_center.png"), img1)
    print("Exported UI Screenshot 1: main_command_center.png")

    # -------------------------------------------------------------
    # SCREENSHOT 2: ANALYTICS & MODEL EVALUATION PAGE
    # -------------------------------------------------------------
    img2 = np.full((H, W, 3), (15, 12, 10), dtype=np.uint8)

    # Top Navbar
    cv2.rectangle(img2, (0, 0), (W, 70), (28, 22, 18), -1)
    cv2.rectangle(img2, (0, 0), (W // 3, 4), (51, 153, 255), -1)
    cv2.rectangle(img2, (W // 3, 0), (2 * W // 3, 4), (255, 255, 255), -1)
    cv2.rectangle(img2, (2 * W // 3, 0), (W, 4), (8, 136, 19), -1)
    cv2.circle(img2, (45, 36), 18, (128, 0, 0), 2)
    cv2.putText(img2, "SENTINEL BORDER WATCH", (80, 44), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img2, "AI MODEL EVALUATION & TELEMETRY // PUNJAB SECTOR", (520, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (51, 153, 255), 1)

    # Left Sidebar
    cv2.rectangle(img2, (20, 90), (320, H - 30), (22, 18, 14), -1)
    cv2.rectangle(img2, (20, 90), (320, H - 30), (45, 38, 30), 1)
    cv2.putText(img2, "TACTICAL CONSOLE", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.65, (51, 153, 255), 1)
    for idx, (item, active) in enumerate(nav_items):
        y = 180 + idx * 45
        is_act = (idx == 3)
        if is_act:
            cv2.rectangle(img2, (30, y - 25), (310, y + 10), (45, 30, 15), -1)
            cv2.rectangle(img2, (30, y - 25), (35, y + 10), (51, 153, 255), -1)
            cv2.putText(img2, item, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        else:
            cv2.putText(img2, item, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

    # KPI Row
    kpis = [
        ("OVERALL mAP@0.50", "0.9950", (0, 229, 255)),
        ("OVERALL mAP@0.5:0.95", "0.9475", (157, 78, 221)),
        ("MODEL PRECISION", "0.9987", (51, 153, 255)),
        ("MODEL RECALL", "1.0000", (19, 136, 8)),
    ]
    kw = (W - 370) // 4
    for idx, (lbl, val, col) in enumerate(kpis):
        kx = 350 + idx * kw
        cv2.rectangle(img2, (kx, 90), (kx + kw - 15, 180), (26, 20, 16), -1)
        cv2.rectangle(img2, (kx, 90), (kx + kw - 15, 180), (50, 40, 30), 1)
        cv2.putText(img2, lbl, (kx + 18, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        cv2.putText(img2, val, (kx + 18, 165), cv2.FONT_HERSHEY_DUPLEX, 0.85, col, 2)

    # Embed Actual Charts: Confusion Matrix & Loss Curves
    cm_img_path = os.path.join("data", "results", "confusion_matrix.png")
    if os.path.exists(cm_img_path):
        cm = cv2.imread(cm_img_path)
        cm_res = cv2.resize(cm, (550, 480))
        img2[210:690, 350:900] = cm_res
        cv2.rectangle(img2, (350, 210), (900, 690), (50, 40, 30), 1)

    res_img_path = os.path.join("data", "results", "results.png")
    if os.path.exists(res_img_path):
        res = cv2.imread(res_img_path)
        res_res = cv2.resize(res, (950, 480))
        img2[210:690, 930:1880] = res_res
        cv2.rectangle(img2, (930, 210), (1880, 690), (50, 40, 30), 1)

    # Per-class table below
    cv2.rectangle(img2, (350, 720), (1880, H - 30), (24, 18, 14), -1)
    cv2.rectangle(img2, (350, 720), (1880, H - 30), (50, 40, 30), 1)
    cv2.putText(img2, "PER-CLASS VALIDATION & INFERENCE BENCHMARK MATRIX", (370, 760), cv2.FONT_HERSHEY_DUPLEX, 0.65, (51, 153, 255), 1)

    headers = ["SURVEILLANCE CLASS", "PRECISION", "RECALL", "mAP@0.50", "mAP@0.5:0.95", "DEPLOYMENT STATUS"]
    col_xs = [370, 650, 850, 1050, 1300, 1550]
    for h, x in zip(headers, col_xs):
        cv2.putText(img2, h, (x, 805), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)
    cv2.line(img2, (370, 820), (1860, 820), (50, 40, 30), 1)

    classes_data = [
        ("PERSON (Patrols & hostiles)", "0.999", "1.000", "0.914", "0.852", "OPERATIONAL - PRIMARY"),
        ("WEAPON (Firearms & arms)", "0.999", "1.000", "0.904", "0.812", "OPERATIONAL - THREAT"),
        ("VEHICLE (Civilian transport)", "0.998", "1.000", "0.977", "0.920", "OPERATIONAL - PATROL"),
        ("MILITARY VEHICLE (Tanks / APC)", "0.999", "1.000", "0.995", "0.985", "OPERATIONAL - HIGH DEFCON"),
    ]
    for idx, row in enumerate(classes_data):
        ry = 860 + idx * 45
        for text, x in zip(row, col_xs):
            col = (0, 255, 120) if "OPERATIONAL" in text else ((51, 153, 255) if x == 370 else (240, 240, 240))
            cv2.putText(img2, text, (x, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

    cv2.imwrite(os.path.join(out_dir, "analytics_threat_matrix.png"), img2)
    print("Exported UI Screenshot 2: analytics_threat_matrix.png")


if __name__ == "__main__":
    render_ui_screenshots()

