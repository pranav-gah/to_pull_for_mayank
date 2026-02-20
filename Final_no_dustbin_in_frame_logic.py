import cv2
import numpy as np
import json
import os
import uuid
import mediapipe as mp
from ultralytics import YOLO
from datetime import timedelta, datetime
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity


# import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

import smtplib
from email.message import EmailMessage

# ==================================================
# ================= CONFIG =========================
# ==================================================

MODEL_PATH = r"C:\Users\pranav\OneDrive\Desktop\best.pt"
VIDEO_PATH = r"C:\Users\pranav\OneDrive\Desktop\unseen_videos\18.mp4"

FACE_MODEL_PATH = r"D:\face_detection\model.pt"
ARCFACE_MODEL_PATH = r"D:\face_detection\models\w600k_r50.onnx"
DB_PATH = r"D:\face_detection\face_db.json"

OUTPUT_DIR = "events_output" 
EVENT_LOG_JSON = "event_logs.json"

PERSON_CLASS = 2
GARBAGE_CLASS = 0
DUSTBIN_CLASS = 1

ASSOCIATION_DISTANCE = 86    # relaxed (YOLO reality)
THROW_DISTANCE = 120

CONF_THRES = 0.3
RECOG_THRES = 0.4
PADDING = 0.25
ALIGNED_SIZE = (112, 112)
FACE_SEARCH_WINDOW = 44

os.makedirs(OUTPUT_DIR, exist_ok=True)


dustbin_present_in_video = False #<============================================GLOBAL FLAG TO CHECK DUSTBIN IN THE VIDEO!
count_of_frames_dustbin_exists = 0


pdf_path_global=None
global_mail=None

# ==================================================
# ================= JSON HELPERS ===================
# ==================================================

def load_event_log_db(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_event_log_db(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# ==================================================
# ================= LOAD DB ========================
# ==================================================

with open(DB_PATH, "r") as f:
    face_db = json.load(f)

# ==================================================
# ================= LOAD MODELS ====================
# ==================================================

model = YOLO(MODEL_PATH)
yolo_face = YOLO(FACE_MODEL_PATH)

arcface_sess = ort.InferenceSession(
    ARCFACE_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
arcface_input = arcface_sess.get_inputs()[0].name

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)



# ==================================================
# ================== PDF MAKER =====================
# ==================================================

def generate_challan_pdf_from_frame(
    annotated_frame,
    person_name,
    coordinates,
    event_id,
    output_dir="challans"
):
    """
    annotated_frame : OpenCV frame (numpy array) from Task 9
    person_name     : string or 'Unknown'
    coordinates     : string (lat,long)
    event_id        : unique event id from Task 9
    """

    os.makedirs(output_dir, exist_ok=True)

    # # =============================
    # # Save annotated frame temporarily
    # # =============================
    temp_img_path = os.path.join(output_dir, f"{event_id}_evidence.jpg")
    cv2.imwrite(temp_img_path, annotated_frame)

    # =============================
    # PDF Metadata
    # =============================
    pdf_path = os.path.join(output_dir, f"{event_id}_challan.pdf")
    authority_name = "GOV OF DELHI"
    violation_type = "Garbage thrown outside the dustbin"
    fine_amount = 1000
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")



    disclaimer_text = (
        f"This is the challan of {fine_amount} for throwing garbage outside "
        f"the dustbin at location => {coordinates}. "
        "This document is system generated and is for demonstration purposes only. "
        "Not legally enforceable."
    )

    # =============================
    # PDF Setup
    # =============================
    pdf = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="HeaderStyle",
        fontSize=22,
        alignment=TA_CENTER,
        spaceAfter=20,
        leading=26
    ))

    styles.add(ParagraphStyle(
        name="LabelStyle",
        fontSize=12,
        spaceAfter=8,
        leading=15
    ))

    styles.add(ParagraphStyle(
        name="DisclaimerStyle",
        fontSize=10,
        spaceBefore=20,
        leading=14
    ))

    elements = []

    # =============================
    # Header (First Page)
    # =============================
    elements.append(Paragraph(authority_name, styles["HeaderStyle"]))
    elements.append(Spacer(1, 12))

    # =============================
    # Violation Details
    # =============================
    elements.append(Paragraph(
        f"<b>Event ID:</b> {event_id}", styles["LabelStyle"]
    ))

    elements.append(Paragraph(
        f"<b>Violation:</b> {violation_type}", styles["LabelStyle"]
    ))

    elements.append(Paragraph(
        f"<b>Person Name:</b> {person_name}", styles["LabelStyle"]
    ))

    elements.append(Paragraph(
        f"<b>Date & Time:</b> {timestamp}", styles["LabelStyle"]
    ))

    elements.append(Paragraph(
        f"<b>Coordinates:</b> {coordinates}", styles["LabelStyle"]
    ))

    elements.append(Spacer(1, 15))

    # =============================
    # Evidence Image (Annotated Frame)
    # =============================
    if os.path.exists(temp_img_path):
        img = Image(temp_img_path)
        img.drawWidth = 4 * inch
        img.drawHeight = 3 * inch
        elements.append(Paragraph("<b>Evidence:</b>", styles["LabelStyle"]))
        elements.append(img)
    else:
        elements.append(Paragraph(
            "<b>Evidence:</b> Image not available", styles["LabelStyle"]
        ))

    # =============================
    # Disclaimer
    # =============================
    elements.append(Paragraph(disclaimer_text, styles["DisclaimerStyle"]))

    # =============================
    # Build PDF
    # =============================
    pdf.build(elements)
    os.remove(temp_img_path)


    print(f"[INFO] Challan PDF generated: {pdf_path}")

    return pdf_path




# ==================================================
# =================== SMPT =========================
# ==================================================

def send_pdf_via_gmail(rec_email,
    pdf_path
):
    sender_email = "pubglucifer69@gmail.com"
    receiver_email = rec_email

    #  App Password (NOT Gmail password)
    app_password = "imjh vfuj tdwd ypct"

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "Garbage Violation Challan"

    # Minimal body (required)
    msg.set_content("Attached is the challan PDF.")

    # =============================
    # Attach PDF
    # =============================
    if not os.path.exists(pdf_path):
        print("[ERROR] PDF not found:", pdf_path)
        return

    with open(pdf_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename=os.path.basename(pdf_path)
        )
    # =============================
    # Gmail SMTP Connection
    # =============================
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, app_password)
        server.send_message(msg)

    print("[INFO] PDF sent successfully to", receiver_email)



# ==================================================
# ================= FACE LOGIC =====================
# ==================================================

def crop_face(frame, box, pad=PADDING):
    h, w, _ = frame.shape
    x1, y1, x2, y2 = map(int, box)
    bw, bh = x2 - x1, y2 - y1
    pw, ph = int(bw * pad), int(bh * pad)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(w, x2 + pw), min(h, y2 + ph)
    return frame[y1:y2, x1:x2]

def align_face(face, landmarks):
    h, w, _ = face.shape
    LEFT_EYE = [33, 133]
    RIGHT_EYE = [362, 263]

    left_eye = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE], axis=0)
    right_eye = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE], axis=0)

    angle = np.degrees(np.arctan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0]
    ))

    cx = int((left_eye[0] + right_eye[0]) / 2)
    cy = int((left_eye[1] + right_eye[1]) / 2)

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    aligned = cv2.warpAffine(face, M, (w, h))
    return cv2.resize(aligned, ALIGNED_SIZE)

def get_embedding(face):
    face = face.astype(np.float32)
    face = np.transpose(face, (2, 0, 1))[None]
    face = (face - 127.5) / 128.0
    return arcface_sess.run(None, {arcface_input: face})[0][0]

def recognize_face(embedding):
    best_name = "Unknown"
    best_score = 0.0
    best_mail = "Unknown"

    for person in face_db:
        raw = person.get("embedding")
        cleaned = []

        if isinstance(raw, list):
            cleaned = raw
        elif isinstance(raw, dict):
            for v in raw.values():
                if isinstance(v, dict):
                    cleaned.extend(v.values())
                else:
                    cleaned.append(v)

        try:
            db_emb = np.array(cleaned, dtype=np.float32)
        except:
            continue

        if db_emb.shape != embedding.shape:
            continue

        score = cosine_similarity(
            embedding.reshape(1, -1),
            db_emb.reshape(1, -1)
        )[0][0]

        if score > best_score:
            best_score = score
            best_name = person.get("name", "Unknown")
            best_mail = person.get("mail", "Unknown")

    if best_score >= RECOG_THRES:
      return best_name, best_mail, best_score
    else:
      return "Unknown", "Unknown", best_score


def run_face_recognition(frame):
    results = yolo_face(frame, conf=CONF_THRES, verbose=False)
    if results[0].boxes is None:
        return "Unknown", 0.0

    for box in results[0].boxes.xyxy.cpu().numpy():
        face_crop = crop_face(frame, box)
        if face_crop.size == 0:
            continue

        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            continue

        aligned = align_face(face_crop, res.multi_face_landmarks[0].landmark)
        emb = get_embedding(aligned)
        return recognize_face(emb)

    return "Unknown","Unknown", 0.0

def robust_face_recognition(video_path, center_frame):
    print("====== Entering Into Face detection Module ======")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, center_frame - FACE_SEARCH_WINDOW))

    best_name = "Unknown"
    best_score = 0.0
    best_mail = "Unknown"

    for _ in range(FACE_SEARCH_WINDOW * 2):
        ret, frame = cap.read()
        if not ret:
            break

        name, mail,score = run_face_recognition(frame)
        if score > best_score:
            best_score = score
            best_name = name
            best_mail = mail

    cap.release()
    return best_name,best_mail, best_score

# ==================================================
# ================= HELPERS ========================
# ==================================================

def center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def inside(p, b):
    x, y = p
    x1, y1, x2, y2 = b
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_annotated(frame, boxes, classes, ids, person_id, person_name):
    img = frame.copy()

    for b, c, tid in zip(boxes, classes, ids):
        x1, y1, x2, y2 = map(int, b)

        if int(c) == PERSON_CLASS:
            label = f"Person {tid}: {person_name if tid == person_id else 'Unknown'}"
            color = (0, 255, 0)
        elif int(c) == GARBAGE_CLASS:
            label = f"Garbage {tid}"
            color = (0, 0, 255)
        elif int(c) == DUSTBIN_CLASS:
            label = "Dustbin"
            color = (255, 0, 0)
        else:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(img, "OUTSIDE DUSTBIN",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 255),
                3)

    return img

# ==================================================
# ================= MAIN PIPELINE ==================
# ==================================================

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

STATE = "THROW"
frame_no = 0


p_name_pdf=None

throw_detected = False
throw_frame_no = None
throw_person_id = None
throw_garbage_id = None

buffered_frames = []
active_associations = {}

last_seen_frame = None
last_seen_boxes = None
last_seen_classes = None
last_seen_ids = None

garbage_centers = {}
check_entry = False
check_last = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    results = model.track(
        frame,
        conf=0.4,
        iou=0.5,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    if results[0].boxes is None or results[0].boxes.id is None:
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy()

    #LOOPING FOR CHECKING IF THERE IS ANY DUSTBIN PRESENT IN THE EXTRACTED CLASSES
    # --------------------------------------------------
    # Track dustbin presence in entire video
    # --------------------------------------------------
    boxes_conf = results[0].boxes.conf.cpu().numpy()

    for c, conf in zip(classes, boxes_conf):
        if int(c) == DUSTBIN_CLASS and conf > 0.8:   # stricter threshold
            print(conf)
            count_of_frames_dustbin_exists += 1
            dustbin_present_in_video = True
            break



    # ==================================================
    # THROW STATE (HYBRID LOGIC)
    # ==================================================
    if STATE == "THROW":

        persons = {}
        garbages = {}

        for b, c, tid in zip(boxes, classes, ids):
            if int(c) == PERSON_CLASS:
                persons[tid] = center(b)
            elif int(c) == GARBAGE_CLASS:
                garbages[tid] = center(b)

        #  ASSOCIATION (BEST CASE)
        for pid, pc in persons.items():
            for gid, gc in garbages.items():
                if np.linalg.norm(np.array(pc) - np.array(gc)) <= ASSOCIATION_DISTANCE:
                    active_associations[pid] = gid

        #  THROW VIA ASSOCIATION
        for pid, gid in list(active_associations.items()):
            if pid in persons and gid in garbages:
                if np.linalg.norm(
                    np.array(persons[pid]) - np.array(garbages[gid])
                ) > THROW_DISTANCE:
                    throw_detected = True
                    throw_frame_no = frame_no
                    throw_person_id = pid
                    throw_garbage_id = gid
                    STATE = "CHECK"
                    buffered_frames.append(frame.copy())
                    break

        #  FALLBACK (OLD WORKING LOGIC)
        if not throw_detected:
            for pid, pc in persons.items():
                for gid, gc in garbages.items():
                    if np.linalg.norm(np.array(pc) - np.array(gc)) > THROW_DISTANCE:
                          if pid in active_associations and active_associations[pid] == gid:
                            throw_detected = True
                            throw_frame_no = frame_no
                            throw_person_id = pid
                            throw_garbage_id = gid
                            STATE = "CHECK"
                            buffered_frames.append(frame.copy())
                            break

    # ==================================================
    # CHECK STATE
    # ==================================================
    elif STATE == "CHECK":

        buffered_frames.append(frame.copy())
        mouth, area = None, None

        for b, c in zip(boxes, classes):
            if int(c) == GARBAGE_CLASS:
                garbage_centers[frame_no] = center(b)
                last_seen_frame = frame.copy()
                last_seen_boxes = boxes.copy()
                last_seen_classes = classes.copy()
                last_seen_ids = ids.copy()

            elif int(c) == DUSTBIN_CLASS:
                x1, y1, x2, y2 = map(int, b)
                w, h = x2 - x1, y2 - y1
                mouth = (x1 + int(.15*w), y1 + int(.2*h),
                         x2 - int(.1*w), y1 + int(.42*h))
                area = (x1 + int(.15*w), y1 + int(.2*h),
                        x2 - int(.1*w), y1 + int(.8*h))

        if garbage_centers and mouth:
            c = list(garbage_centers.values())[-1]
            if not check_entry and inside(c, mouth):
                check_entry = True
            if check_entry and area and inside(c, area):
                check_last = True
    

cap.release()

# ==================================================
# ================= FINAL OUTPUT ===================
# ==================================================

inside_dustbin = False
outside_dustbin = False

if dustbin_present_in_video:
    print("\nDustbin detected in the video\n")
    # Dustbin exists somewhere in video
    if throw_detected:
        inside_dustbin = check_entry and check_last
        outside_dustbin = not inside_dustbin
    else:
        outside_dustbin = False

else:

    # No dustbin in entire video
    print("\nNo dustbin detected in the video\n")
    if throw_detected:
        outside_dustbin = True
        print("\nGarbage is thrown on floor\n")
    else:
        outside_dustbin = False


print("\n========== Calculating FINAL RESULT ==========\n"
"                       |\n"
"                       |\n"
"                       V\n")

if outside_dustbin and last_seen_frame is not None:

    person_name,global_mail, confidence = robust_face_recognition(
        VIDEO_PATH, throw_frame_no
    )

    video_key = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    folder = os.path.join(OUTPUT_DIR, video_key)
    os.makedirs(folder, exist_ok=True)

    annotated = draw_annotated(
        last_seen_frame,
        last_seen_boxes,
        last_seen_classes,
        last_seen_ids,
        throw_person_id,
        person_name
    )
    cv2.imwrite(os.path.join(folder, "annotated_frame.jpg"), annotated)


    p_name_pdf=person_name
    annotated_pdf=annotated



    h, w, _ = buffered_frames[0].shape
    writer = cv2.VideoWriter(
        os.path.join(folder, "evidence_clip.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    for f in buffered_frames:
        writer.write(f)
    writer.release()
    event_id=uuid.uuid4()
    with open(os.path.join(folder, "event_details.txt"), "w") as f:
        f.write(f"event_id: {event_id}\n")
        f.write(f"timestamp: {timedelta(seconds=throw_frame_no / fps)}\n")
        f.write(f"person_id: {throw_person_id}\n")
        f.write(f"person_name: {person_name}\n")
        f.write(f"garbage_id: {throw_garbage_id}\n")
        f.write("final_verdict: outside_dustbin\n")
        f.write(f"confidence: {confidence}\n")
        f.write(f"created_at: {datetime.now().isoformat()}\n")

    event_log = load_event_log_db(EVENT_LOG_JSON)
    event_entry = {
        "event_id": str(event_id),
        "timestamp": str(timedelta(seconds=throw_frame_no / fps)),
        "person_id": int(throw_person_id),
        "person_name": person_name,
        "garbage_id": int(throw_garbage_id),
        "final_verdict": "outside_dustbin",
        "confidence": float(confidence),
        "created_at": datetime.now().isoformat()
    }

    event_log.setdefault(video_key, []).append(event_entry)   #--> event_log.setdefault(video_key, []) emplicitly doing -> 
    save_event_log_db(EVENT_LOG_JSON, event_log)              #--> event_log.setdefault(video_key, [])
                                                              #        event_log["1"] = []

    print("VIOLATION SAVED")



in_dust = inside_dustbin
out_dust = outside_dustbin

if out_dust:
    pdf_path_global=generate_challan_pdf_from_frame(
        annotated_frame=annotated_pdf,
        person_name=p_name_pdf if p_name_pdf else "Unknown",
        coordinates=410130813,
        event_id=event_id
    )

    send_pdf_via_gmail(global_mail,pdf_path_global)    #<----------------------calling pdf sending funtion module --------------------------->
    os.remove(pdf_path_global)




elif inside_dustbin:
    print("GARBAGE WENT INSIDE DUSTBIN")

else:
    print("NO THROW DETECTED")

print(count_of_frames_dustbin_exists)
