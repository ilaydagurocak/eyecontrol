import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cam.isOpened():
    print("Error: Camera could not be accessed.")
    exit()

# Get the screen size using pyautogui
screen_w, screen_h = pyautogui.size()

while True:
    ret, image = cam.read()

    # Check if the frame is read correctly
    if not ret:
        print("Error: Frame could not be read.")
        break

    # Flip the image for a selfie-view display
    image = cv2.flip(image, 1)

    # Get the window height and width
    window_h, window_w, _ = image.shape

    # Convert the color space from BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to find face landmarks
    processed_image = face_mesh.process(rgb_image)

    # Extract face landmark points
    all_face_landmark_points = processed_image.multi_face_landmarks

    if all_face_landmark_points:
        one_face_landmark_points = all_face_landmark_points[0].landmark

        for id, landmark_point in enumerate(one_face_landmark_points[474:478]):
            x = int(landmark_point.x * window_w)
            y = int(landmark_point.y * window_h)

            if id == 1:
                mouse_x = int(screen_w / window_w * x)
                mouse_y = int(screen_h / window_h * y)
                pyautogui.moveTo(mouse_x, mouse_y)

            cv2.circle(image, (x, y), 3, (0, 0, 255))

        left_eye = [one_face_landmark_points[145], one_face_landmark_points[159]]

        for landmark_point in left_eye:
            x = int(landmark_point.x * window_w)
            y = int(landmark_point.y * window_h)

            cv2.circle(image, (x, y), 3, (0, 255, 255))

            if left_eye[0].y - left_eye[1].y < 0.01:
                pyautogui.click()
                pyautogui.sleep(2)

    cv2.imshow("Eye Controlled Mouse", image)
    key = cv2.waitKey(100)

    if key == 27:  # Escape key to break the loop
        break

# Release the camera and close all OpenCV windows outside the loop
cam.release()
cv2.destroyAllWindows()
