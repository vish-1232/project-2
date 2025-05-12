import cv2
import datetime
import os
import csv

# ---------------------- LOGIN SYSTEM ----------------------
users = {"admin": "1234"}  # Add more users as needed

def login():
    print("Enter Username:")
    username = input()
    print("Enter Password:")
    password = input()

    if users.get(username) == password:
        print("✅ Login successful!\n")
        return username
    else:
        print("❌ Invalid username or password. Try again.\n")
        return None

# Loop until correct login
username = None
while username is None:
    username = login()

# ---------------------- LOG SETUP ----------------------
log_folder = "logs"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

log_file_path = os.path.join(log_folder, "person_log.txt")
csv_log_path = os.path.join(log_folder, "event_log.csv")

# Create CSV file with headers if it doesn't exist
if not os.path.exists(csv_log_path):
    with open(csv_log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Username", "Event"])

log_file = open(log_file_path, "a", encoding="utf-8")

# ---------------------- CAMERA CHECK ----------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame = cap.read()

if not ret:
    status = "❌ Camera failed to initialize!"
elif cv2.countNonZero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) == 0:
    status = "⚠️ Black screen detected!"
else:
    status = "✅ Camera is working normally."

timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
log_file.write(f"[{timestamp}] {username}: {status}\n")
log_file.flush()

# ---------------------- GENERATE STATUS PAGE ----------------------
with open("status.html", "w", encoding="utf-8") as html:
    html.write(f"""
<html>
<head>
    <title>Webcam Status</title>
</head>
<body style="font-family:sans-serif; text-align:center;">
    <h1>📷 Webcam Monitoring System</h1>
    <p style="font-size:20px;">👤 User: <b>{username}</b></p>
    <p style="font-size:22px; color:{'red' if 'Black' in status or 'failed' in status else 'green'};">{status}</p>
    <p>🕒 Last Checked: {timestamp}</p>
    <a href="status.html"><button style="padding:10px;font-size:16px;">🔁 Refresh</button></a>
</body>
</html>
""")
print("✅ Status page created: 'status.html'")

# ---------------------- MOTION + FACE DETECTION ----------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame1_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

print("🔍 Starting detection. Press 'q' to quit.")
while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)

    diff = cv2.absdiff(frame1_gray, frame2_gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False

    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    # Face Detection
    faces = face_cascade.detectMultiScale(frame2_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 0, 0), 2)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f"[{timestamp}] {username}: 👤 Face Detected at {faces[0]}\n")
        log_file.flush()

    if motion_detected:
        cv2.putText(frame2, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(csv_log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, username, "Motion Detected"])

    if len(faces) > 0:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(csv_log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, username, "Face Detected"])

    cv2.imshow("Surveillance", frame2)
    frame1_gray = frame2_gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
print("📁 Surveillance ended. Log saved.")

# ---------------------- GRAPH GENERATION ----------------------
def generate_graph():
    # Read data from the CSV file
    motion_count = 0
    face_count = 0

    with open(csv_log_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) < 3:  # Skip malformed rows
                continue
            event = row[2]
            if "Motion Detected" in event:
                motion_count += 1
            if "Face" in event:
                face_count += 1

    # Display ASCII bar chart in console
    print("\nDetection Events Bar Chart")
    print("============================")
    print(f"Motion Detected: {'#' * motion_count} ({motion_count})")
    print(f"Face Detected:   {'#' * face_count} ({face_count})")

    print("\n✅ Graph generated in console.")

# Call the function to generate the graph
generate_graph()
