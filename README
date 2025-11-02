Web-Based Face Recognition Attendance System
This project is a complete, self-contained web application for taking attendance using facial recognition. It allows users to train new faces and record attendance (check-in/check-out) directly from a web browser, making it accessible from both desktop and mobile devices like Android phones.
The application runs as a single Flask server on your computer, which handles all backend logic, database operations, and model training.
Features
All-in-One Web Interface: A single application handles both training new users and recording attendance.
Mobile-Friendly: Access the system from your phone's browser. It uses your phone's front or back camera for recognition.
Web-Based Training: A new /train page provides a 7-stage guided process to capture robust training data (front, left, right, up, down, close, far).
Secure (HTTPS): Runs over https:// (using an ad-hoc SSL certificate) to allow browser camera access on mobile devices.
Real-time Recognition: The browser sends frames to the server, which detects and recognizes faces in real-time.
Smart Attendance Logic: Enforces check-in / check-out status. A user cannot check in if already checked in, and vice-versa.
Persistent Storage: Uses a single SQLite database (attendance.db) to store all user info, statuses, and event logs.
LBPH Face Recognizer: Uses OpenCV's robust LBPH algorithm for face recognition.
File Structure
Your project directory must be set up as follows for the application to work correctly:
Face_Attendance_App/
│
├── app.py                  <-- The main web server (run this file)
│
├── haarcascade_frontalface_default.xml  <-- OpenCV model for *finding* faces
│
├── templates/              <-- Flask folder for HTML files
│   ├── index.html          <-- The main attendance page
│   └── train.html          <-- The user training page
│
├── dataset/                <-- (Created automatically) Stores training images
│
├── attendance.db           <-- (Created automatically) SQLite database
│
└── trainer.yml             <-- (Created automatically) The trained recognition model

Setup and Installation
1. Prerequisites
Python 3.x
pip (Python package installer)
2. Install Python Libraries
Open your terminal or command prompt and install the required libraries:
pip install Flask opencv-python numpy Pillow pyOpenSSL

Flask: The web server.
opencv-python: For all face detection and recognition.
numpy: Required by OpenCV.
Pillow: For image processing.
pyOpenSSL: Required for running Flask over https://.
3. Download the Haar Cascade
You need the pre-trained model that detects (finds) faces in an image.
Download the file: haarcascade_frontalface_default.xml
You can find it on the OpenCV GitHub repository.
Click "Raw", then right-click and "Save As...".
Save this file in the same directory as your app.py.
How to Run
Start the Server:
Open your terminal, navigate to your project folder, and run:
python app.py


Find Your IP Address:
The server will print your PC's local IP address. Look for a line like:
[INFO] To access from your phone, use https://192.168.1.10:5000
Access from your Phone:
Make sure your phone is on the same Wi-Fi network as your computer.
Open your phone's browser (Chrome, Safari, etc.).
Go to the https:// address shown in your terminal (e.g., https://192.168.1.10:5000).
Accept the Security Warning:
Your browser will show a security warning ("This connection is not private"). This is normal because the app is using a self-signed certificate.
Click "Advanced" and then "Proceed to [your IP address] (unsafe)".
How to Use
1. Training a New User
From the main page, click the "Train New User" button.
Enter the user's name in the text box and click "Start Training".
The app will now guide you through 7 stages (Look Straight, Look Left, Look Right, etc.).
Follow the on-screen instructions. A progress bar will show you how many images have been captured.
After all images are saved, the server will automatically train the model. This may take 10-30 seconds.
Once complete, you will see a "Training Complete!" message. You can then go "Back to App".
2. Taking Attendance
Open the main page (https://...:5000).
Point your phone's camera at your face.
The status box will change from "Looking for face..." to Detected: [Your Name] (xx%).
Once your name is visible, press the "Check In" or "Check Out" button.
A success or error message (like "Already checked in") will appear at the bottom of the screen.


