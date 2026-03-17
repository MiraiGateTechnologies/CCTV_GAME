"""
================================================
  STREAM MANAGER — Background clip downloader
  Downloads + validates clips in background
  Sirf "green light" (traffic moving) clips
  ko scheduler ke liye queue mein rakhta hai
================================================
"""

# ┌─────────────────────────────────────────────────────────────────────┐
# │  FILE   : stream_manager.py                                         │
# │  PURPOSE: Background mein continuously clips download + validate    │
# │           Agar traffic nahi chal rahi (red light) → clip reject     │
# │           Agar traffic chal rahi → clip approve → scheduler ko de   │
# │           Ye file STANDALONE hai — kisi aur file pe depend nahi     │
# ├─────────────────────────────────────────────────────────────────────┤
# │  CHANGE GUIDE — Agar kuch change karna ho:                          │
# │                                                                     │
# │  ► Clip kitne seconds ka ho                                         │
# │    → CLIP_DURATION = 35  (neeche, line ~35)                        │
# │                                                                     │
# │  ► Validation speed (har kitne frames check karo)                   │
# │    → VALIDATION_FPS_SKIP = 5  (har 5th frame check hota hai)       │
# │      Kam karo = slow but accurate, zyada karo = fast but rough      │
# │                                                                     │
# │  ► Kitne frames movement hone chahiye "green" ke liye               │
# │    → MIN_MOVEMENT_FRAMES = 10  (zyada karo = strict validation)     │
# │                                                                     │
# │  ► Validation ke liye YOLO model change karna ho                    │
# │    → YOLO_MODEL_PATH = "yolo11m.pt"                                 │
# │      (yolo11n.pt = fast/less accurate, yolo11x.pt = slow/accurate) │
# │                                                                     │
# │  ► Kaun se vehicles detect karo validation mein                     │
# │    → _validate_clip() ke andar classes=[2, 3, 5, 7] (line ~119)    │
# │      2=Car, 3=Motorcycle, 5=Bus, 7=Truck                           │
# │                                                                     │
# │  ► Minimum pixel movement threshold (5px → zyada = strict)          │
# │    → _validate_clip() ke andar: if closest_dist > 5.0 (line ~144)  │
# │                                                                     │
# │  ► Kitne clips ready queue mein rakhe (max)                         │
# │    → ClipManager.__init__: queue.Queue(maxsize=5) (line ~27)        │
# └─────────────────────────────────────────────────────────────────────┘

import os
import cv2
import sys
import time
import queue
import threading
import subprocess
import ultralytics
import tempfile
import shutil

# ─────────────────────────────────────────────
#  SETTINGS — Yahan se tweak karo
# ─────────────────────────────────────────────
CLIP_DURATION = 35         # Har clip kitne seconds ka ho
VALIDATION_FPS_SKIP = 5    # Har Nth frame check karo (speed vs accuracy)
MIN_MOVEMENT_FRAMES = 10   # Kitne frames movement chahiye "green" ke liye
YOLO_MODEL_PATH = "yolo11m.pt"  # Validation ke liye YOLO model

class ClipManager:
    """
    Background manager that constantly downloads clips from a live stream,
    validates them for traffic movement, and queues the good ones.
    """
    def __init__(self, stream_url, stream_name):
        self.stream_url = stream_url
        self.stream_name = stream_name
        self.ready_queue = queue.Queue(maxsize=5) # Holds valid clip paths
        self.is_running = True
        
        # Temp directories
        self.temp_dir = tempfile.mkdtemp(prefix="cctv_buffer_")
        self.download_dir = os.path.join(self.temp_dir, "downloads")
        self.validated_dir = os.path.join(self.temp_dir, "validated")
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.validated_dir, exist_ok=True)
        
        # Load YOLO model once for validation thread
        print(f"[MANAGER] Loading YOLOv8 model for background validation...")
        self.model = ultralytics.YOLO(YOLO_MODEL_PATH)
        
        # Start Threads
        self.download_thread = threading.Thread(target=self._download_loop, daemon=True)
        self.validate_thread = threading.Thread(target=self._validate_loop, daemon=True)
        
        self.download_queue = queue.Queue() # Holds unvalidated downloaded clip paths
        
        self.download_thread.start()
        self.validate_thread.start()

    def _extract_clip(self, output_path):
        """Uses yt-dlp to download a 35s chunk."""
        print(f"[DOWNLOADER] Fetching new clip for {self.stream_name}...")
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--external-downloader", "ffmpeg",
            "--external-downloader-args", f"ffmpeg:-ss 0 -t {CLIP_DURATION}",
            self.stream_url,
            "-o", output_path,
            "--force-overwrites",
            "--quiet", "--no-warnings"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return os.path.exists(output_path)
        except Exception as e:
            print(f"[DOWNLOADER] Error fetching clip: {e}")
            return False

    def _download_loop(self):
        """Continuously downloads clips pushing them to validate queue."""
        clip_counter = 0
        while self.is_running:
            # If we have enough validated clips and pending downloads, slow down
            if self.ready_queue.qsize() >= 3 and self.download_queue.qsize() >= 2:
                time.sleep(2)
                continue
                
            clip_counter += 1
            filename = f"raw_clip_{clip_counter}.mp4"
            out_path = os.path.join(self.download_dir, filename)
            
            success = self._extract_clip(out_path)
            if success:
                self.download_queue.put(out_path)
            else:
                time.sleep(5) # Delay on error

    def _validate_clip(self, clip_path):
        """Fast-forwards through a clip to check if cars are moving."""
        print(f"[VALIDATOR] Scanning {os.path.basename(clip_path)} for traffic movement...")
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return False
            
        frame_idx = 0
        movement_score = 0
        
        # A simple heuristic: if we detect vehicles and they have bounding boxes, 
        # we assume *some* movement. In a real red-light scenario, we'd compare box
        # coordinates frame-over-frame (displacement). For speed, we just ensure 
        # vehicles are detected consistently across frames.
        
        # Track previous centroids to check actual displacement
        prev_centroids = []
        consecutive_static = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % VALIDATION_FPS_SKIP != 0:
                continue # Skip frames for speed
                
            results = self.model.predict(
                frame, 
                classes=[2, 3, 5, 7], # Car, Motorcycle, Bus, Truck
                conf=0.3, 
                verbose=False
            )
            
            current_centroids = []
            valid_movement = False
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    current_centroids.append((cx, cy))
                    
            # Compare with previous centroids
            if prev_centroids and current_centroids:
                for cx, cy in current_centroids:
                    # Find closest in previous
                    closest_dist = float('inf')
                    for px, py in prev_centroids:
                        dist = ((cx - px)**2 + (cy - py)**2)**0.5
                        if dist < closest_dist:
                            closest_dist = dist
                    
                    if closest_dist > 5.0: # Minimum 5 pixel movement
                        valid_movement = True
                        break
            elif current_centroids:
                # First frame with cars
                valid_movement = True
                
            if valid_movement:
                movement_score += 1
                consecutive_static = 0
            else:
                consecutive_static += 1
                
            prev_centroids = current_centroids
            
            # If we've seen enough movement, early exit (Fast Validation)
            if movement_score >= MIN_MOVEMENT_FRAMES:
                cap.release()
                return True
                
        cap.release()
        return movement_score >= 5 # Lower threshold if clip was short

    def _validate_loop(self):
        """Pulls raw downloaded clips, validates them, queues good ones."""
        while self.is_running:
            try:
                # Wait for a newly downloaded clip
                raw_clip_path = self.download_queue.get(timeout=2)
                
                is_valid = self._validate_clip(raw_clip_path)
                
                if is_valid:
                    print(f"[VALIDATOR] Clip APPROVED! Moving to ReadyQueue.")
                    valid_name = f"valid_{os.path.basename(raw_clip_path)}"
                    valid_path = os.path.join(self.validated_dir, valid_name)
                    # Move to validated folder
                    shutil.move(raw_clip_path, valid_path)
                    
                    # Block here if queue is full, so we don't over-download
                    self.ready_queue.put(valid_path)
                else:
                    print(f"[VALIDATOR] Clip REJECTED (Red Light detected). Discarding.")
                    os.remove(raw_clip_path)
                    
                self.download_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VALIDATOR] Error: {e}")

    def get_next_valid_clip(self, timeout=None):
        """
        Main function called by scheduler.py to get the next playable clip.
        Blocks until a valid (Green Light) clip is ready.
        """
        try:
            return self.ready_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def mark_clip_done(self, clip_path):
        """Call this when finished playing a clip so it gets deleted."""
        try:
            if os.path.exists(clip_path):
                os.remove(clip_path)
        except Exception as e:
            print(f"[MANAGER] Failed to cleanup clip {clip_path}: {e}")

    def cleanup(self):
        """Stops the threads and deletes temporary files."""
        print(f"[MANAGER] Cleaning up stream buffers...")
        self.is_running = False
        shutil.rmtree(self.temp_dir, ignore_errors=True)
