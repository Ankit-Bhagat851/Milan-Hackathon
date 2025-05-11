from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Set up Edge driver
driver_path = r"D:\Hackathon\Milan-Hackathon2\msedgedriver.exe"
options = Options()
service = EdgeService(executable_path=driver_path)

driver = webdriver.Edge(service=service, options=options)
# Maximize the browser window
driver.maximize_window()

driver.get("https://www.linkedin.com/jobs/")
time.sleep(2)

# Accept LinkedIn cookies (if any)
try:
    accept_button = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="artdeco-global-alert-container"]/div/section/div/div[2]/button[1]'))
    )
    accept_button.click()
    print("Accepted LinkedIn cookie policy.")
except Exception:
    print("No LinkedIn cookie popup found.")

# Wait for the login form to load
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, '//*[@id="session_key"]'))
)

# Enter email and password
email_element = driver.find_element(By.XPATH, '//*[@id="session_key"]')
email_element.send_keys("ankitbhagatrock392000@gmail.com")  # Replace with your email

password_element = driver.find_element(By.XPATH, '//*[@id="session_password"]')
password_element.send_keys("Arya@007")  # Replace with your password

# Click on the sign-in button
sign_in_button = driver.find_element(By.XPATH, '//*[@id="main-content"]/section[1]/div/div/form/div[2]/button')
sign_in_button.click()

# Wait for login to complete (adjust time if needed)
time.sleep(5)

# Open Rinascente job application form in a new tab
driver.execute_script("window.open('https://rinascente.intervieweb.it/jobs/stage-customer-insight-analyst-313/it/', '_blank');")
time.sleep(2)

# Switch to the new tab
driver.switch_to.window(driver.window_handles[1])
print("Switched to the second tab.")

# Accept cookies on Rinascente site using provided XPath
try:
    cookie_accept = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="qc-cmp2-ui"]/div[2]/div/button[3]'))
    )
    cookie_accept.click()
    print("Accepted Rinascente cookie policy.")
except Exception as e:
    print("No Rinascente cookie popup found or error:", e)

# Click the "Apply" or "Candidati" button
try:
    apply_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="description__buttons"]/div/div[1]/button'))
    )
    apply_button.click()
    print("Clicked on the 'Apply' button.")
except Exception as e:
    print("Could not click the apply button:", e)


try:
    nome_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="nome"]'))
    )
    
    location = nome_input.location
    size = nome_input.size
    
    print(f"Textbox Location: x={location['x']}, y={location['y']}")
    print(f"Textbox Size: width={size['width']}, height={size['height']}")
    x1 = int(location['x'])
    y1 = int(location['y'])
    x2 = x1 + int(size['width'])
    y2 = y1 + int(size['height'])
except Exception as e:
    print("Could not get textbox location/size:", e)


import cv2
import mediapipe as mp
import time
import csv
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Setup MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing = mp.solutions.drawing_utils

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Output paths
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)
csv_file = os.path.join(output_dir, 'gaze_feedback_data.csv')
model_file = os.path.join(output_dir, 'gaze_model.pkl')

# Load or train model
def load_or_train_model():
    if os.path.exists(model_file):
        return joblib.load(model_file)
    elif os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = df[df['feedback'] == 'yes']
        if df.empty:
            return None
        X = df[['eye_x', 'eye_y']]
        y = df['box_x1', 'box_y1', 'box_x2', 'box_y2']
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)
        joblib.dump(model, model_file)
        return model
    return None

# Save to CSV
def save_to_csv(data):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'eye_x', 'eye_y', 'box_x1', 'box_y1', 'box_x2', 'box_y2', 'x1','y1','x2','y2','feedback']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# Initialize video
cap = cv2.VideoCapture(0)
start_time = time.time()
model = load_or_train_model()

# Define screen size (fake screen resolution for grid simulation)
screen_width, screen_height = 1600, 1080

eye_x, eye_y = -1, -1
box_coords = None

tracking_duration = 10
draw_duration = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    frame_height, frame_width = frame.shape[:2]

    current_time = time.time()
    elapsed = current_time - start_time

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get coordinates from the left eye landmarks
            left_eye_coords = [face_landmarks.landmark[i] for i in LEFT_EYE]
            eye_x = int(sum([p.x for p in left_eye_coords]) / len(left_eye_coords) * frame_width)
            # Get coordinates from the left eye landmarks
            right_eye_coords = [face_landmarks.landmark[i] for i in RIGHT_EYE]
            eye_y = int(sum([p.y for p in left_eye_coords]) / len(left_eye_coords) * frame_height)
            # Draw eye points
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * frame_width)
                y = int(face_landmarks.landmark[idx].y * frame_height)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            break

    if elapsed > tracking_duration and box_coords is None:
        box_w, box_h = 150, 50
        x1 = max(eye_x - box_w // 2, 0)
        y1 = max(eye_y - box_h // 2, 0)
        x2 = min(x1 + box_w, frame_width)
        y2 = min(y1 + box_h, frame_height)
        box_coords = (x1, y1, x2, y2)

    if box_coords:
        cv2.rectangle(frame, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (255, 0, 255), 2)
        cv2.putText(frame, "Textbox", (box_coords[0], box_coords[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

    # Resize the frame
    resized_frame = cv2.resize(frame, (screen_width, screen_height))
    cv2.imshow("Pupil Tracker", resized_frame)
    if cv2.waitKey(1) & 0xFF == 27 or elapsed > (tracking_duration + draw_duration):
        break

cap.release()
cv2.destroyAllWindows()

if box_coords:
    print(f"\nRectangle Box Coordinates: Top-Left ({box_coords[0]}, {box_coords[1]}), "
          f"Bottom-Right ({box_coords[2]}, {box_coords[3]})")
    feedback = input("Are the box coordinates with the gaze coordinates correct? (yes/no): ").strip().lower()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_to_csv({
        'timestamp': timestamp,
        'eye_x': eye_x,
        'eye_y': eye_y,
        'box_x1': box_coords[0],
        'box_y1': box_coords[1],
        'box_x2': box_coords[2],
        'box_y2': box_coords[3],
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'feedback': feedback
    })

    if feedback == 'yes':
        df = pd.read_csv(csv_file)
        df = df[df['feedback'] == 'yes']
        if not df.empty:
            X = df[['eye_x', 'eye_y']]
            y = df[['box_x1', 'box_y1', 'box_x2', 'box_y2']]
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(X, y)
            joblib.dump(model, model_file)
            print("Model updated.")
else:
    print("No box was drawn.")


# Enter name in the textbox
name_element = driver.find_element(By.XPATH, '//*[@id="nome"]')
name_element.send_keys("Anson")

try:
    nome_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="cognome"]'))
    )
    
    location = nome_input.location
    size = nome_input.size
    
    print(f"Textbox Location: x={location['x']}, y={location['y']}")
    print(f"Textbox Size: width={size['width']}, height={size['height']}")
    x1 = int(location['x'])
    y1 = int(location['y'])
    x2 = x1 + int(size['width'])
    y2 = y1 + int(size['height'])
except Exception as e:
    print("Could not get textbox location/size:", e)

# Setup MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing = mp.solutions.drawing_utils

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Output paths
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)
csv_file = os.path.join(output_dir, 'gaze_feedback_data.csv')
model_file = os.path.join(output_dir, 'gaze_model.pkl')

# Load or train model
def load_or_train_model():
    if os.path.exists(model_file):
        return joblib.load(model_file)
    elif os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = df[df['feedback'] == 'yes']
        if df.empty:
            return None
        X = df[['eye_x', 'eye_y']]
        y = df['box_x1', 'box_y1', 'box_x2', 'box_y2']
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)
        joblib.dump(model, model_file)
        return model
    return None

# Save to CSV
def save_to_csv(data):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'eye_x', 'eye_y', 'box_x1', 'box_y1', 'box_x2', 'box_y2', 'x1','y1','x2','y2','feedback']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# Initialize video
cap = cv2.VideoCapture(0)
start_time = time.time()
model = load_or_train_model()

# Define screen size (fake screen resolution for grid simulation)
screen_width, screen_height = 1600, 1080

eye_x, eye_y = -1, -1
box_coords = None

tracking_duration = 10
draw_duration = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    frame_height, frame_width = frame.shape[:2]

    current_time = time.time()
    elapsed = current_time - start_time

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get coordinates from the left eye landmarks
            left_eye_coords = [face_landmarks.landmark[i] for i in LEFT_EYE]
            eye_x = int(sum([p.x for p in left_eye_coords]) / len(left_eye_coords) * frame_width)
            # Get coordinates from the left eye landmarks
            right_eye_coords = [face_landmarks.landmark[i] for i in RIGHT_EYE]
            eye_y = int(sum([p.y for p in left_eye_coords]) / len(left_eye_coords) * frame_height)
            # Draw eye points
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * frame_width)
                y = int(face_landmarks.landmark[idx].y * frame_height)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            break

    if elapsed > tracking_duration and box_coords is None:
        box_w, box_h = 150, 50
        x1 = max(eye_x - box_w // 2, 0)
        y1 = max(eye_y - box_h // 2, 0)
        x2 = min(x1 + box_w, frame_width)
        y2 = min(y1 + box_h, frame_height)
        box_coords = (x1, y1, x2, y2)

    if box_coords:
        cv2.rectangle(frame, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (255, 0, 255), 2)
        cv2.putText(frame, "Textbox", (box_coords[0], box_coords[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

    # Resize the frame
    resized_frame = cv2.resize(frame, (screen_width, screen_height))
    cv2.imshow("Pupil Tracker", resized_frame)
    if cv2.waitKey(1) & 0xFF == 27 or elapsed > (tracking_duration + draw_duration):
        break

cap.release()
cv2.destroyAllWindows()

if box_coords:
    print(f"\nRectangle Box Coordinates: Top-Left ({box_coords[0]}, {box_coords[1]}), "
          f"Bottom-Right ({box_coords[2]}, {box_coords[3]})")
    feedback = input("Are the box coordinates with the gaze coordinates correct? (yes/no): ").strip().lower()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_to_csv({
        'timestamp': timestamp,
        'eye_x': eye_x,
        'eye_y': eye_y,
        'box_x1': box_coords[0],
        'box_y1': box_coords[1],
        'box_x2': box_coords[2],
        'box_y2': box_coords[3],
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'feedback': feedback
    })

    if feedback == 'yes':
        df = pd.read_csv(csv_file)
        df = df[df['feedback'] == 'yes']
        if not df.empty:
            X = df[['eye_x', 'eye_y']]
            y = df[['box_x1', 'box_y1', 'box_x2', 'box_y2']]
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(X, y)
            joblib.dump(model, model_file)
            print("Model updated.")
else:
    print("No box was drawn.")

# Enter surname in the textbox
surname_element = driver.find_element(By.XPATH, '//*[@id="cognome"]')
surname_element.send_keys("Bhagat")

input('press enter to quit')
driver.quit()
