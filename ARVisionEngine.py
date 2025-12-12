import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
import threading
import time

class ARVisionEngine:
    def __init__(self):
        self.detector_model = YOLO('yolov8n.pt')
        self.pose_analyzer = mp.solutions.pose
        self.pose_detector = self.pose_analyzer.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True
        )
        self.draw_utils = mp.solutions.drawing_utils
        self.pose_specs = mp.solutions.pose.PoseLandmark
        
    def process_frame(self, rgb_frame):
        identification_results = self.detector_model(rgb_frame, conf=0.5)
        return identification_results

class PersonActivityRecognizer:
    def __init__(self):
        self.pose_analyzer = mp.solutions.pose
        self.pose_processor = self.pose_analyzer.Pose(
            static_image_mode=False,
            model_complexity=1
        )
        self.pose_draw = mp.solutions.drawing_utils
        self.frame_history = deque(maxlen=30)
        
    def extract_pose(self, frame_rgb):
        landmark_output = self.pose_processor.process(frame_rgb)
        return landmark_output
    
    def classify_behavior(self, pose_data, frame_shape):
        if not pose_data.pose_landmarks:
            return "Unknown"
        
        landmarks = pose_data.pose_landmarks.landmark
        
        head_pos = landmarks[0]
        shoulder_l = landmarks[11]
        shoulder_r = landmarks[12]
        elbow_l = landmarks[13]
        elbow_r = landmarks[14]
        hip_l = landmarks[23]
        hip_r = landmarks[24]
        
        if head_pos.y > shoulder_l.y + 0.15 and head_pos.y > shoulder_r.y + 0.15:
            return "Sleeping/Bowing"
        
        mouth_dist = abs(landmarks[9].x - landmarks[10].x)
        if mouth_dist > 0.08:
            return "Laughing/Talking"
        
        elbow_angle_l = self.compute_angle(
            shoulder_l, elbow_l, landmarks[15]
        )
        elbow_angle_r = self.compute_angle(
            shoulder_r, elbow_r, landmarks[16]
        )
        
        if elbow_angle_l < 30 or elbow_angle_r < 30:
            return "Hand Raised"
        
        if abs(landmarks[15].x - landmarks[16].x) < 0.1:
            return "Standing Still"
        
        return "Moving"
    
    def compute_angle(self, p1, p2, p3):
        vec_a = np.array([p1.x - p2.x, p1.y - p2.y])
        vec_b = np.array([p3.x - p2.x, p3.y - p2.y])
        
        mag_a = np.linalg.norm(vec_a)
        mag_b = np.linalg.norm(vec_b)
        
        if mag_a == 0 or mag_b == 0:
            return 0
        
        cos_angle = np.dot(vec_a, vec_b) / (mag_a * mag_b)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        return np.degrees(angle)

class ObjectMetadataRegistry:
    def __init__(self):
        self.metadata_catalog = {
            'person': {'color': (0, 255, 0), 'icon': '👤', 'desc': 'Human'},
            'chair': {'color': (255, 0, 0), 'icon': '🪑', 'desc': 'Chair'},
            'table': {'color': (0, 0, 255), 'icon': '🛠️', 'desc': 'Table'},
            'laptop': {'color': (255, 255, 0), 'icon': '💻', 'desc': 'Computer'},
            'book': {'color': (255, 0, 255), 'icon': '📖', 'desc': 'Book'},
            'cup': {'color': (0, 255, 255), 'icon': '☕', 'desc': 'Cup'},
            'backpack': {'color': (128, 0, 128), 'icon': '🎒', 'desc': 'Backpack'},
            'handbag': {'color': (255, 128, 0), 'icon': '👜', 'desc': 'Handbag'},
            'tie': {'color': (128, 128, 0), 'icon': '👔', 'desc': 'Tie'},
            'suitcase': {'color': (128, 0, 128), 'icon': '🧳', 'desc': 'Suitcase'},
        }
    
    def fetch_metadata(self, label):
        return self.metadata_catalog.get(label.lower(), {
            'color': (200, 200, 200),
            'icon': '?',
            'desc': label
        })

class InteractiveARSystem:
    def __init__(self, display_widget):
        self.root_widget = display_widget
        self.frame_processor = ARVisionEngine()
        self.pose_analyzer = PersonActivityRecognizer()
        self.metadata_db = ObjectMetadataRegistry()
        
        self.vid_stream = None
        self.streaming_active = False
        self.selected_detection = None
        self.detection_buffer = []
        
        self.initialize_camera()
        
    def initialize_camera(self):
        self.vid_stream = cv2.VideoCapture(0)
        self.vid_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def capture_frame(self):
        success, frame_data = self.vid_stream.read()
        return (success, frame_data) if success else (False, None)
    
    def process_detections(self, frame_rgb, yolo_output):
        annotations = []
        
        for detection in yolo_output[0].boxes:
            conf = float(detection.conf[0])
            if conf < 0.5:
                continue
            
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            obj_class_id = int(detection.cls[0])
            obj_label = self.frame_processor.detector_model.names[obj_class_id]
            
            entry = {
                'label': obj_label,
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'class_id': obj_class_id,
                'activity': None
            }
            
            if obj_label.lower() == 'person':
                crop_h = y2 - y1
                crop_w = x2 - x1
                padding = int(min(crop_h, crop_w) * 0.1)
                
                y1_crop = max(0, y1 - padding)
                y2_crop = min(frame_rgb.shape[0], y2 + padding)
                x1_crop = max(0, x1 - padding)
                x2_crop = min(frame_rgb.shape[1], x2 + padding)
                
                person_region = frame_rgb[y1_crop:y2_crop, x1_crop:x2_crop]
                if person_region.size > 0:
                    pose_result = self.pose_analyzer.extract_pose(person_region)
                    activity_label = self.pose_analyzer.classify_behavior(
                        pose_result, person_region.shape
                    )
                    entry['activity'] = activity_label
            
            annotations.append(entry)
        
        self.detection_buffer = annotations
        return annotations
    
    def draw_overlays(self, frame_data, detection_list):
        display_frame = frame_data.copy()
        
        for idx, det in enumerate(detection_list):
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['conf']
            activity = det['activity']
            
            meta = self.metadata_db.fetch_metadata(label)
            bbox_color = meta['color']
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), bbox_color, 2)
            
            display_text = f"{label}: {conf:.2f}"
            if activity:
                display_text += f" | {activity}"
            
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            bg_rect = (x1, max(0, y1 - 25), x1 + text_size[0] + 5, y1)
            cv2.rectangle(display_frame, bg_rect[:2], bg_rect[2:], bbox_color, -1)
            cv2.putText(display_frame, display_text, (x1 + 2, y1 - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            info_y = y2 + 20
            cv2.putText(display_frame, f"ID: {idx}", (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, bbox_color, 1)
        
        return display_frame
    
    def render_stream(self):
        if not self.streaming_active:
            return
        
        success, frame_rgb = self.capture_frame()
        if not success:
            self.root_widget.after(10, self.render_stream)
            return
        
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        
        yolo_results = self.frame_processor.process_frame(frame_rgb)
        detection_list = self.process_detections(frame_rgb, yolo_results)
        
        annotated_frame = self.draw_overlays(frame_rgb, detection_list)
        
        img_pil = Image.fromarray(annotated_frame)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.root_widget.cam_label.config(image=img_tk)
        self.root_widget.cam_label.image = img_tk
        
        self.root_widget.after(10, self.render_stream)
    
    def fetch_details(self, det_idx):
        if 0 <= det_idx < len(self.detection_buffer):
            det = self.detection_buffer[det_idx]
            details = f"Object: {det['label'].upper()}\n"
            details += f"Confidence: {det['conf']:.2%}\n"
            details += f"Bounding Box: {det['bbox']}\n"
            
            if det['activity']:
                details += f"Activity: {det['activity']}\n"
            
            meta = self.metadata_db.fetch_metadata(det['label'])
            details += f"\nDescription: {meta['desc']}"
            
            return details
        return "No object selected"

class MainARApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AR Vision System - Object Detection & Activity Recognition")
        self.geometry("1200x700")
        self.config(bg='#1a1a1a')
        
        self.ar_engine = None
        self.processing_thread = None
        
        self.build_ui()
        self.initialize_system()
        
    def build_ui(self):
        header_frame = ttk.Frame(self)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        title_label = ttk.Label(header_frame, text="🎯 AR Object Detection & Activity Analyzer",
                               font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT)
        
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.start_btn = ttk.Button(button_frame, text="▶ Start Camera",
                                    command=self.activate_stream)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ Stop Camera",
                                   command=self.deactivate_stream, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        content_frame = ttk.Frame(self)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_section = ttk.Frame(content_frame)
        left_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.cam_label = tk.Label(left_section, bg='black', width=640, height=480)
        self.cam_label.pack(fill=tk.BOTH, expand=True)
        
        right_section = ttk.Frame(content_frame)
        right_section.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        control_label = ttk.Label(right_section, text="Detection Controls",
                                 font=("Arial", 12, "bold"))
        control_label.pack(pady=10)
        
        self.obj_idx_var = tk.StringVar(value="0")
        idx_frame = ttk.Frame(right_section)
        idx_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(idx_frame, text="Object Index:").pack(side=tk.LEFT)
        idx_spinbox = ttk.Spinbox(idx_frame, from_=0, to=20, textvariable=self.obj_idx_var, width=10)
        idx_spinbox.pack(side=tk.LEFT, padx=5)
        
        detail_btn = ttk.Button(right_section, text="📋 View Details",
                               command=self.show_object_info)
        detail_btn.pack(fill=tk.X, padx=10, pady=5)
        
        info_label = ttk.Label(right_section, text="Object Information",
                              font=("Arial", 11, "bold"))
        info_label.pack(pady=(20, 10))
        
        self.info_display = scrolledtext.ScrolledText(right_section, height=20, width=35,
                                                     wrap=tk.WORD, state=tk.DISABLED)
        self.info_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        stats_label = ttk.Label(right_section, text="Session Statistics",
                               font=("Arial", 11, "bold"))
        stats_label.pack(pady=(15, 10))
        
        self.stats_display = scrolledtext.ScrolledText(right_section, height=8, width=35,
                                                      wrap=tk.WORD, state=tk.DISABLED)
        self.stats_display.pack(fill=tk.BOTH, padx=10, pady=5)
    
    def initialize_system(self):
        try:
            self.ar_engine = InteractiveARSystem(self)
            self.update_stats()
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize: {str(e)}")
    
    def activate_stream(self):
        if self.ar_engine is None:
            messagebox.showerror("Error", "System not initialized")
            return
        
        self.ar_engine.streaming_active = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.processing_thread = threading.Thread(target=self.ar_engine.render_stream, daemon=True)
        self.processing_thread.start()
    
    def deactivate_stream(self):
        if self.ar_engine:
            self.ar_engine.streaming_active = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def show_object_info(self):
        try:
            obj_idx = int(self.obj_idx_var.get())
            details = self.ar_engine.fetch_details(obj_idx)
            
            self.info_display.config(state=tk.NORMAL)
            self.info_display.delete(1.0, tk.END)
            self.info_display.insert(tk.END, details)
            self.info_display.config(state=tk.DISABLED)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid object index")
    
    def update_stats(self):
        if self.ar_engine:
            total_detections = len(self.ar_engine.detection_buffer)
            person_count = sum(1 for d in self.ar_engine.detection_buffer if d['label'].lower() == 'person')
            
            stats_text = f"Total Objects Detected: {total_detections}\n"
            stats_text += f"People Detected: {person_count}\n"
            stats_text += f"Other Objects: {total_detections - person_count}\n"
            
            if person_count > 0:
                activities = [d['activity'] for d in self.ar_engine.detection_buffer
                            if d['label'].lower() == 'person' and d['activity']]
                if activities:
                    stats_text += f"\nActivities Detected:\n"
                    for act in set(activities):
                        count = activities.count(act)
                        stats_text += f"  • {act}: {count}\n"
            
            self.stats_display.config(state=tk.NORMAL)
            self.stats_display.delete(1.0, tk.END)
            self.stats_display.insert(tk.END, stats_text)
            self.stats_display.config(state=tk.DISABLED)
        
        self.after(1000, self.update_stats)
    
    def on_closing(self):
        if self.ar_engine and self.ar_engine.vid_stream:
            self.ar_engine.vid_stream.release()
        self.destroy()

if __name__ == "__main__":
    app = MainARApplication()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()