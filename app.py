import flask
import numpy as np
import odoorpc
import pickle
import os
import base64
import io
import time
import datetime
import cv2 # <-- Import for image conversion
from PIL import Image
from fpdf import FPDF
from flask import Flask, render_template, request, jsonify, make_response

# --- THIS IS THE NEW LIBRARY ---
# It does NOT use dlib.
from deepface import DeepFace

# --- Configuration ---
# Odoo config
ODOO_URL = ''
ODOO_DB = ''
ODOO_USER =''
ODOO_PASSWORD = 'API KEY'

# Face model config
ENCODINGS_FILE = 'encodings.pkl'
DEEPFACE_MODEL = 'VGG-Face' 
DEEPFACE_DETECTOR = 'opencv' 
RECOGNITION_THRESHOLD = 0.40 

# --- Initialize Flask App ---
app = Flask(__name__, template_folder='templates')

# --- Global State ---
app.last_action_time = 0
COOLDOWN_SECONDS = 5
known_encodings = {}

# --- Odoo Functions ---
def get_odoo_connection():
    """Helper function to connect to Odoo and return the client."""
    try:
        print(f"[Odoo] Attempting to connect to {ODOO_URL}...")
        odoo = odoorpc.ODOO(
            ODOO_URL.replace('https://', ''), 
            protocol='jsonrpc+ssl', 
            port=443
        )
        odoo.login(ODOO_DB, ODOO_USER, ODOO_PASSWORD)
        print(f"[Odoo] Connection successful.")
        return odoo
    except Exception as e:
        print(f"[Odoo] Connection Error: {e}")
        return None

def get_employee_and_state(odoo, user_name):
    """Finds an employee in Odoo and returns their record and state."""
    try:
        Employee = odoo.env['hr.employee']
        employee_ids = Employee.search([('name', '=', user_name)])
        
        if not employee_ids:
            msg = f"Odoo Error: Employee '{user_name}' not found."
            print(f"[Odoo] {msg}")
            return None, None, msg
            
        employee = Employee.browse(employee_ids[0])
        current_state = employee.attendance_state
        
        print(f"[Odoo] Found employee: {employee.name} (ID: {employee.id})")
        print(f"[Odoo] Employee current state: {current_state}")
        
        return employee, current_state, None
    except Exception as e:
        msg = f"Odoo Error during employee search: {e}"
        print(f"[Odoo] {msg}")
        return None, None, msg

def record_check_in(user_name):
    """Connects to Odoo and checks IN the employee."""
    try:
        odoo = get_odoo_connection()
        if odoo is None:
            return ("Could not connect to Odoo.", False)

        employee, current_state, error_msg = get_employee_and_state(odoo, user_name)
        
        if error_msg:
            return (error_msg, False)

        if current_state == 'checked_in':
            msg = f"'{user_name}' is already checked in."
            print(f"[Odoo] {msg}")
            return (msg, False)

        Attendance = odoo.env['hr.attendance']
        action_date = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"[Odoo] Action: Checking IN")
        vals = {
            'employee_id': employee.id,
            'check_in': action_date,
        }
        new_att_id = Attendance.create(vals)
        print(f"[Odoo] Created new attendance record (ID: {new_att_id})")
        message = f"'{user_name}' checked in successfully in Odoo."
        return (message, True)

    except Exception as e:
        msg = f"Odoo API Error: {str(e)}"
        print(f"[Odoo] {msg}")
        return (msg, False)

def record_check_out(user_name):
    """Connects to Odoo and checks OUT the employee."""
    try:
        odoo = get_odoo_connection()
        if odoo is None:
            return ("Could not connect to Odoo.", False)
            
        employee, current_state, error_msg = get_employee_and_state(odoo, user_name)
        
        if error_msg:
            return (error_msg, False)

        if current_state == 'checked_out':
            msg = f"'{user_name}' is already checked out."
            print(f"[Odoo] {msg}")
            return (msg, False)
        
        Attendance = odoo.env['hr.attendance']
        action_date = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        print(f"[Odoo] Action: Checking OUT")
        
        domain = [
            ('employee_id', '=', employee.id),
            ('check_out', '=', False)
        ]
        attendance_ids = Attendance.search(domain, limit=1)
        
        if not attendance_ids:
            msg = f"Odoo Error: Cannot check out. No open check-in record found for '{user_name}'."
            print(f"[Odoo] {msg}")
            return (msg, False)

        attendance_to_close = Attendance.browse(attendance_ids[0])
        attendance_to_close.write({'check_out': action_date})
        
        print(f"[Odoo] Closed attendance record (ID: {attendance_ids[0]})")
        message = f"'{user_name}' checked out successfully in Odoo."
        return (message, True)

    except Exception as e:
        msg = f"Odoo API Error: {str(e)}"
        print(f"[Odoo] {msg}")
        return (msg, False)


# --- Image Decoding Function ---
def decode_base64_image(base64_string):
    """
    Decodes a base64 string into an RGB numpy array using PIL.
    """
    try:
        if "," in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_bytes = base64.b64decode(base64_string)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Convert PIL Image to BGR format for deepface/opencv
        img_array = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_bgr
        
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

# --- Face Embedding Functions ---
def get_embedding(img):
    """
    Uses deepface to get the facial embedding (fingerprint).
    Returns None if no face or multiple faces are found.
    """
    try:
        embedding_objs = DeepFace.represent(
            img, 
            model_name=DEEPFACE_MODEL, 
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=True
        )
        if len(embedding_objs) > 1:
            print("[ERROR] Multiple faces detected.")
            return None, "Multiple faces detected. Please be alone."
        embedding = embedding_objs[0]['embedding']
        return embedding, None
    except ValueError as e:
        if "Face could not be detected" in str(e):
            print("[INFO] No face detected.")
            return None, "No face detected in frame."
        else:
            print(f"[ERROR] deepface error: {e}")
            return None, f"Library Error: {e}"
    except Exception as e:
        print(f"[ERROR] deepface error: {e}")
        return None, f"Library Error: {e}"

def find_best_match(unknown_embedding):
    """Calculates the cosine distance to find the best match."""
    if not known_encodings:
        return None, float('inf')
    best_name = None
    best_distance = float('inf')
    for name, enc_list in known_encodings.items():
        for known_embedding in enc_list:
            a = np.asarray(known_embedding)
            b = np.asarray(unknown_embedding)
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            distance = 1 - (dot_product / (norm_a * norm_b))
            if distance < best_distance:
                best_distance = distance
                best_name = name
    return best_name, best_distance

# --- Encodings File Functions ---
def load_encodings():
    """Loads known encodings from the pickle file."""
    global known_encodings
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                known_encodings = pickle.load(f)
            print(f"[INFO] Loaded {len(known_encodings)} known faces from '{ENCODINGS_FILE}'.")
        except Exception as e:
            print(f"[ERROR] Could not load encodings file: {e}. Starting fresh.")
            known_encodings = {}
    else:
        print(f"[INFO] Encodings file not found. Starting fresh.")
        known_encodings = {}

def save_encodings():
    """Saves the current known_encodings to the pickle file."""
    try:
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(known_encodings, f)
        print(f"[INFO] Saved {len(known_encodings)} faces to '{ENCODINGS_FILE}'.")
    except Exception as e:
        print(f"[ERROR] Could not save encodings file: {e}")

# --- Flask HTML Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

# --- Recognition Route ---
@app.route('/recognize_frame', methods=['POST'])
def recognize_frame():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"status": "error", "message": "No image data provided"}), 400
    img = decode_base64_image(data['image'])
    if img is None:
        return jsonify({"status": "error", "message": "Invalid image data"}), 400
    unknown_embedding, error_msg = get_embedding(img)
    if error_msg:
        return jsonify({"status": "no_face"})
    best_name, best_distance = find_best_match(unknown_embedding)
    if best_name and best_distance < RECOGNITION_THRESHOLD:
        confidence = (1.0 - best_distance) 
        return jsonify({
            "status": "known", 
            "user_name": best_name,
            "confidence_percent": round(confidence * 100, 2)
        })
    else:
        return jsonify({"status": "unknown"})


# --- Training Routes (Odoo version) ---
@app.route('/start_training', methods=['POST'])
def start_training():
    """Prepares to train a new user. Clears old data if any."""
    data = request.json
    user_name = data.get('name', '').strip()
    if not user_name:
        return jsonify({"success": False, "message": "Name cannot be empty."})

    # Odoo version doesn't need to create a user,
    # just clear the encodings in memory.
    known_encodings[user_name] = [] 
    
    print(f"[INFO] Starting training for new user: '{user_name}'. Old encodings cleared.")
    return jsonify({"success": True, "name": user_name})

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    """Receives a frame, finds a face, and stores its encoding in memory."""
    data = request.json
    user_name = data.get('name')
    image_data = data.get('image')
    if not all([user_name, image_data]):
        return jsonify({"success": False, "message": "Missing name or image data."})
    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({"success": False, "message": "Invalid image data."})
    
    embedding, error_msg = get_embedding(img)
    if error_msg:
        return jsonify({"success": False, "message": error_msg})

    if user_name not in known_encodings:
        known_encodings[user_name] = []
    known_encodings[user_name].append(embedding)
    
    print(f"[INFO] Captured frame {len(known_encodings[user_name])} for '{user_name}'")
    return jsonify({"success": True, "message": "Frame captured"})

@app.route('/run_model_training', methods=['POST'])
def run_model_training():
    """Saves all captured encodings to the file."""
    print("[INFO] Saving all captured encodings to disk...")
    save_encodings()
    return jsonify({"success": True, "message": "Training complete! Model saved."})


# --- API Routes (Odoo) ---
@app.route('/api/check_in', methods=['POST'])
def check_in_action():
    now = time.time()
    if now - app.last_action_time < COOLDOWN_SECONDS:
        return jsonify({"success": False, "message": "Please wait..."})
    data = request.json
    user_name = data.get('user_name')
    if not user_name:
        return jsonify({"success": False, "message": "No known face detected!"})
    message, is_success = record_check_in(user_name)
    if is_success:
        app.last_action_time = now
    return jsonify({"success": is_success, "message": message})

@app.route('/api/check_out', methods=['POST'])
def check_out_action():
    now = time.time()
    if now - app.last_action_time < COOLDOWN_SECONDS:
        return jsonify({"success": False, "message": "Please wait..."})
    data = request.json
    user_name = data.get('user_name')
    if not user_name:
        return jsonify({"success": False, "message": "No known face detected!"})
    message, is_success = record_check_out(user_name)
    if is_success:
        app.last_action_time = now
    return jsonify({"success": is_success, "message": message})


# --- Reporting Routes (Odoo) ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Attendance Report', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def report_table(self, data, col_widths):
        self.set_font('Arial', 'B', 10)
        headers = ['Name', 'Date', 'First In', 'Last Out', 'Total Duration', 'All Sessions']
        col_keys = ['name', 'date', 'first_in', 'last_out', 'duration', 'sessions']
        for i, header in enumerate(headers):
            self.cell(col_widths[col_keys[i]], 10, header, 1, 0, 'C')
        self.ln()
        self.set_font('Arial', '', 9)
        if not data:
            self.cell(sum(col_widths.values()), 10, 'No data found for this selection.', 1, 1, 'C')
            return
        for row in data:
            start_y = self.get_y()
            row_data_points = [
                str(row['name']), str(row['event_date']), str(row['first_check_in']),
                str(row['last_check_out']), str(row['total_duration_formatted']),
                str(row['all_sessions'])
            ]
            cell_heights = []
            current_x = self.get_x()
            for i, key in enumerate(col_keys):
                width = col_widths[key]
                self.multi_cell(width, 8, row_data_points[i], 0, 'L')
                cell_heights.append(self.get_y())
                current_x += width
                self.set_xy(current_x, start_y)
            max_y = max(cell_heights)
            current_x = self.l_margin
            self.set_xy(current_x, start_y)
            for i, key in enumerate(col_keys):
                width = col_widths[key]
                self.rect(current_x, start_y, width, max_y - start_y)
                current_x += width
            self.set_xy(self.l_margin, max_y)

def format_duration(seconds):
    if seconds is None or seconds < 0: return "N/A"
    if seconds == 0: return "0m 0s"
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    parts = []
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0: parts.append(f"{minutes}m")
    if seconds > 0 or not parts: parts.append(f"{seconds}s")
    return " ".join(parts)

def _get_report_data(user_id, start_date, end_date):
    """Internal function to query ODOO for a smart summary report."""
    try:
        print("[Odoo] Connecting for report...")
        odoo = get_odoo_connection()
        if odoo is None:
            return None, "Could not connect to Odoo for report."
        
        Attendance = odoo.env['hr.attendance']
        domain = []
        if user_id: domain.append(('employee_id', '=', int(user_id)))
        if start_date: domain.append(('check_in', '>=', f"{start_date} 00:00:00"))
        if end_date: domain.append(('check_in', '<=', f"{end_date} 23:59:59"))

        attendances = Attendance.search_read(
            domain,
            fields=['employee_id', 'check_in', 'check_out', 'worked_hours'],
            order='check_in asc'
        )
        report_data = {} 
        for att in attendances:
            employee_id = att['employee_id'][0]
            employee_name = att['employee_id'][1]
            check_in_str = att['check_in']
            if not check_in_str: continue 
            check_in_dt = datetime.datetime.strptime(check_in_str, '%Y-%m-%d %H:%M:%S')
            event_date_str = check_in_dt.strftime('%Y-%m-%d')
            key = (employee_id, event_date_str)
            if key not in report_data:
                report_data[key] = {
                    'name': employee_name, 'event_date': event_date_str,
                    'check_ins': [], 'check_outs': [], 'sessions': [],
                    'total_duration_seconds': 0
                }
            report_data[key]['check_ins'].append(check_in_dt)
            report_data[key]['total_duration_seconds'] += (att['worked_hours'] * 3600)
            check_out_str = att['check_out']
            if check_out_str:
                check_out_dt = datetime.datetime.strptime(check_out_str, '%Y-%m-%d %H:%M:%S')
                report_data[key]['check_outs'].append(check_out_dt)
                session_str = f"{check_in_dt.strftime('%H:%M:%S')} - {check_out_dt.strftime('%H:%M:%S')}"
            else:
                session_str = f"{check_in_dt.strftime('%H:%M:%S')} - (Still In)"
            report_data[key]['sessions'].append(session_str)

        final_list = []
        for key, data in sorted(report_data.items(), key=lambda item: (item[1]['event_date'], item[1]['name']), reverse=True):
            first_in = min(data['check_ins']).strftime('%H:%M:%S') if data['check_ins'] else "---"
            last_out = max(data['check_outs']).strftime('%H:%M:%S') if data['check_outs'] else "---"
            all_sessions = "\n".join(data['sessions']) if data['sessions'] else "No sessions"
            final_list.append({
                'name': data['name'], 'event_date': data['event_date'],
                'first_check_in': first_in, 'last_check_out': last_out,
                'total_duration_seconds': data['total_duration_seconds'],
                'total_duration_formatted': format_duration(data['total_duration_seconds']),
                'all_sessions': all_sessions
            })
        return final_list, None
    except Exception as e:
        msg = f"Odoo Report Error: {str(e)}"
        print(f"[Odoo] {msg}")
        return None, msg

@app.route('/reports')
def reports_page():
    return render_template('reports.html')

@app.route('/api/users', methods=['GET'])
def get_all_users():
    """Fetches all users from ODOO for the report filter dropdown."""
    try:
        print("[Odoo] Connecting for user list...")
        odoo = get_odoo_connection()
        if odoo is None:
            return jsonify({"error": "Could not connect to Odoo"}), 500
        
        Employee = odoo.env['hr.employee']
        employees = Employee.search_read([], fields=['id', 'name'], order='name asc')
        user_list = [{"user_id": emp['id'], "name": emp['name']} for emp in employees]
        print(f"[Odoo] Found {len(user_list)} employees.")
        return jsonify(user_list)
    except Exception as e:
        msg = f"Odoo User Fetch Error: {str(e)}"
        print(f"[Odoo] {msg}")
        return jsonify({"error": msg}), 500

@app.route('/api/attendance_report', methods=['GET'])
def get_attendance_report_json():
    """Provides the attendance report data as JSON for the web table."""
    user_id = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    data, error = _get_report_data(user_id, start_date, end_date)
    if error:
        return jsonify({"error": error}), 500
    return jsonify(data)

@app.route('/api/download_pdf', methods=['GET'])
def download_pdf_report():
    """Generates and serves the attendance report as a PDF download."""
    user_id = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    data, error = _get_report_data(user_id, start_date, end_date)
    if error:
        return f"Error generating report: {error}", 500
    pdf = PDF(orientation='L', format='A4')
    pdf.add_page()
    col_widths = {'name': 45, 'date': 25, 'first_in': 25, 'last_out': 25, 'duration': 30, 'sessions': 127}
    pdf.report_table(data, col_widths)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"attendance_report_{timestamp}.pdf"
    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response

# --- Main Block REMOVED ---
# The if __name__ == "__main__": block has been removed.
# All startup logic is now in run_server.py