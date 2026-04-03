"""
================================================================
  STREAM MANAGER — Sequential Download + Single YOLO Pipeline
================================================================

Architecture (thread-safe, GPU-safe):

  SequentialDownloader (1 thread, round-robin ALL streams, NO GPU):
    Stream A → resolve + download → quality check → push to download_queue
    Stream B → resolve + download → quality check → push to download_queue
    Stream C → ... (round-robin, one at a time)
    Queue full (3 clips)? → sleep until YOLO consumes

  YOLOWorker (SINGLE instance, OWNS GPU):
    Pick from download_queue → offline_count() → validate count
    → push approved clips to ready_queue

  Pipeline (coordinator):
    Manages downloader + YOLO worker
    Exposes ready_queue to scheduler

Benefits:
  - 1 download at a time: no YouTube 429 spam, no race conditions
  - Round-robin: every stream gets equal turn
  - Single YOLO thread: no ByteTrack tracker corruption
  - Queue target = 3: downloads match YOLO consumption rate
"""

import os
import sys
import time
import queue
import threading
import subprocess
import tempfile
import shutil
import cv2

try:
    import yt_dlp
    HAS_YT_DLP = True
except ImportError:
    HAS_YT_DLP = False

# ─────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────
CLIP_DURATION = 41            # Total clip length (35 counting + 6 waiting)
COUNT_DURATION = 35           # Only count vehicles in first 35 seconds
MIN_VEHICLE_COUNT = 5         # Minimum vehicles for a clip to be approved
DOWNLOAD_TIMEOUT = 90         # Max seconds for ffmpeg download
READY_QUEUE_MAX = 10           # Max pre-processed clips ready for rounds
DOWNLOAD_QUEUE_MAX = 5        # Max downloaded-but-unprocessed clips waiting for YOLO
DOWNLOAD_QUEUE_TARGET = 3     # Keep exactly 3 clips in download queue
MAX_PENDING_PER_STREAM = 2    # Max unprocessed clips per stream (throttle fast streams)
YT_URL_CACHE_TTL = 5 * 3600  # Refresh YouTube URLs every 5 hours (live URLs valid ~6h)
YT_URL_REFRESH_INTERVAL = 4 * 3600  # Background refresh every 4 hours


# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────

class DownloadedClip:
    """A clip downloaded and quality-checked, waiting for YOLO processing."""

    def __init__(self, clip_path: str, stream_name: str,
                 line_config_file: str, imgsz: int, confidence: float):
        self.clip_path = clip_path
        self.stream_name = stream_name
        self.line_config_file = line_config_file
        self.imgsz = imgsz
        self.confidence = confidence
        self.downloaded_at = time.time()


class PreProcessedClip:
    """Fully YOLO-processed clip ready for a game round."""

    def __init__(self, clip_path: str, stream_name: str, result: int,
                 detections: dict, counting_events: list,
                 total_frames: int, clip_fps: float):
        self.clip_path = clip_path
        self.stream_name = stream_name
        self.result = result
        self.detections = detections
        self.counting_events = counting_events
        self.total_frames = total_frames
        self.clip_fps = clip_fps
        self.processed_at = time.time()


# ─────────────────────────────────────────────
#  SEQUENTIAL DOWNLOADER — Single thread, round-robin all streams
# ─────────────────────────────────────────────

class SequentialDownloader:
    """
    Downloads clips from ALL streams using a SINGLE thread in round-robin order.
    One download at a time — no race conditions, no YouTube 429 spam.

    Flow:
      Stream 7 → resolve + download → push to queue
      Stream 16 → resolve + download → push to queue
      Stream 17 → resolve + download → push to queue
      (queue full? sleep until YOLO consumes)
      Stream 19 → ...
      ...back to Stream 7 (round-robin)
    """

    def __init__(self, download_queue: queue.Queue):
        self.download_queue = download_queue
        self.streams: list[dict] = []   # [{name, url, config, imgsz, conf}, ...]
        self.is_running = True
        self._prewarm_done = threading.Event()  # Signals when URL prewarm is complete

        # Per-stream YouTube URL cache: {youtube_url: (resolved_url, timestamp)}
        self._url_cache: dict[str, tuple[str, float]] = {}
        self._yt_fail_count = 0

        # Per-stream failure tracking: {stream_name: consecutive_failures}
        self._failures: dict[str, int] = {}

        # Stats
        self.clips_downloaded = 0
        self.clips_quality_rejected = 0
        self.download_failures = 0
        self.clip_counter = 0

        # Temp directory
        self.temp_dir = tempfile.mkdtemp(prefix="cctv_dl_seq_")

        # Single download thread (waits for prewarm before starting downloads)
        self.thread = threading.Thread(
            target=self._download_loop, daemon=True, name="DL-Sequential",
        )
        self.thread.start()

        # Background URL refresh thread (refreshes cached URLs every 4 hours)
        self._start_url_refresh_thread()

    def add_stream(self, name: str, url: str, config_path: str,
                   imgsz: int = 1600, confidence: float = 0.10):
        """Add a stream to the round-robin rotation."""
        # Don't add duplicates
        for s in self.streams:
            if s["name"] == name:
                return
        self.streams.append({
            "name": name, "url": url, "config": config_path,
            "imgsz": imgsz, "conf": confidence,
        })
        self._failures[name] = 0

    def remove_stream(self, name: str):
        """Remove a stream from the rotation."""
        self.streams = [s for s in self.streams if s["name"] != name]
        self._failures.pop(name, None)

    # ── URL Resolution (yt-dlp Python API — no subprocess, no timeout) ──

    def _is_youtube(self, url: str) -> bool:
        return "youtube.com" in url or "youtu.be" in url

    def _resolve_url_api(self, stream_name: str, stream_url: str) -> str | None:
        """Resolve YouTube URL using yt-dlp Python API — fast, no subprocess timeout.

        Uses browser cookies to avoid 429 rate limiting.
        Falls back to subprocess if yt-dlp Python API not available.
        """
        if not self._is_youtube(stream_url):
            return stream_url

        # Check cache — return instantly if valid
        cached = self._url_cache.get(stream_url)
        if cached and (time.time() - cached[1] < YT_URL_CACHE_TTL):
            return cached[0]

        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [RESOLVE] {stream_name}: Resolving YouTube URL...")

        if not HAS_YT_DLP:
            print(f"[RESOLVE] yt-dlp not installed! pip install yt-dlp")
            return None

        try:
            ydl_opts = {
                'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/best',
                'no_playlist': True,
                'quiet': True,
                'no_warnings': True,
            }

            # Try browser cookies (Chrome → Edge → Firefox)
            # This prevents 429 — YouTube sees you as logged-in user
            cookies_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'cookies.txt')
            if os.path.exists(cookies_file):
                ydl_opts['cookiefile'] = cookies_file
            else:
                # Try extracting from browser automatically
                for browser in ('chrome', 'edge', 'firefox'):
                    try:
                        ydl_opts['cookiesfrombrowser'] = (browser,)
                        break
                    except Exception:
                        continue

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(stream_url, download=False)
                url = info.get('url')
                if not url:
                    # For DASH/HLS, get manifest URL
                    formats = info.get('formats', [])
                    if formats:
                        url = formats[-1].get('url')

                if url:
                    self._url_cache[stream_url] = (url, time.time())
                    self._yt_fail_count = 0
                    print(f"[{time.strftime('%H:%M:%S')}] [RESOLVE] {stream_name}: OK")
                    return url

            print(f"[{time.strftime('%H:%M:%S')}] [RESOLVE] {stream_name}: No URL found")
            return None

        except Exception as e:
            err = str(e)[:200]
            print(f"[{time.strftime('%H:%M:%S')}] [RESOLVE] {stream_name}: Failed — {err}")
            if "429" in err or "Too Many Requests" in err:
                backoff = min(300, 10 * (2 ** self._yt_fail_count))
                self._yt_fail_count += 1
                print(f"[RESOLVE] {stream_name}: YouTube 429 — backoff {backoff}s")
                time.sleep(backoff)
            else:
                self._yt_fail_count = 0
            return None

    def prewarm_urls(self):
        """Pre-resolve ALL stream URLs at startup. Call BEFORE download loop starts.

        Resolves all YouTube URLs in one go so the download loop
        never waits for yt-dlp. If a URL fails, retries 3 times.
        """
        if not self.streams:
            return

        youtube_streams = [s for s in self.streams if self._is_youtube(s["url"])]
        if not youtube_streams:
            print("[PREWARM] No YouTube streams to resolve")
            return

        print(f"[PREWARM] Resolving {len(youtube_streams)} YouTube URLs...")

        for stream in youtube_streams:
            name = stream["name"]
            url = stream["url"]

            # Already cached?
            cached = self._url_cache.get(url)
            if cached and (time.time() - cached[1] < YT_URL_CACHE_TTL):
                print(f"[PREWARM] {name}: cached ✓")
                continue

            # Try up to 3 times
            for attempt in range(3):
                resolved = self._resolve_url_api(name, url)
                if resolved:
                    break
                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    print(f"[PREWARM] {name}: retry {attempt + 2}/3 in {wait}s...")
                    time.sleep(wait)

            if not resolved:
                print(f"[PREWARM] {name}: FAILED after 3 attempts — will retry in download loop")

        cached_count = sum(1 for s in youtube_streams if s["url"] in self._url_cache)
        print(f"[PREWARM] Done: {cached_count}/{len(youtube_streams)} URLs cached ✓")

    def _start_url_refresh_thread(self):
        """Start background thread that refreshes cached URLs periodically.

        Runs every YT_URL_REFRESH_INTERVAL (4 hours). Refreshes one stream
        at a time with 30 sec gap to avoid YouTube 429.
        """
        def _refresh_loop():
            while self.is_running:
                time.sleep(YT_URL_REFRESH_INTERVAL)
                if not self.is_running:
                    break

                youtube_streams = [s for s in self.streams if self._is_youtube(s["url"])]
                if not youtube_streams:
                    continue

                print(f"[URL-REFRESH] Refreshing {len(youtube_streams)} YouTube URLs...")
                for stream in youtube_streams:
                    if not self.is_running:
                        break
                    # Force refresh by clearing cache for this URL
                    self._url_cache.pop(stream["url"], None)
                    self._resolve_url_api(stream["name"], stream["url"])
                    # 30 sec gap between streams to avoid 429
                    time.sleep(30)

                cached_count = sum(1 for s in youtube_streams if s["url"] in self._url_cache)
                print(f"[URL-REFRESH] Done: {cached_count}/{len(youtube_streams)} refreshed ✓")

        thread = threading.Thread(target=_refresh_loop, daemon=True, name="URL-Refresh")
        thread.start()

    # ── Download ──

    def _download_clip(self, stream_name: str, stream_url: str,
                       output_path: str) -> bool:
        """Resolve URL + download clip via ffmpeg. No re-encoding (-c copy)."""
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [DL] {stream_name}: Downloading {CLIP_DURATION}s clip...")

        download_url = self._resolve_url_api(stream_name, stream_url)
        if not download_url:
            return False

        if os.path.exists(output_path):
            os.remove(output_path)

        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-i", download_url,
            "-t", str(CLIP_DURATION),
            "-r", "20",                      # ← force output to 30fps
            "-c:v", "h264_nvenc",              # ← re-encode needed for fps change
            "-preset", "p1",          # ← fastest encoding (minimal CPU)
            "-cq", "28",                    # ← quality (lower=better, 23=default)
            "-movflags", "+faststart",
            output_path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT
            )
            if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"[{time.strftime('%H:%M:%S')}] [DL] {stream_name}: "
                      f"Ready ({size_mb:.1f} MB)")
                return True
            else:
                err_msg = result.stderr.strip()[:300] if result.stderr else "unknown"
                print(f"[{time.strftime('%H:%M:%S')}] [DL] {stream_name}: "
                      f"Failed — {err_msg}")
                if self._is_youtube(stream_url):
                    self._url_cache.pop(stream_url, None)
                return False
        except subprocess.TimeoutExpired:
            print(f"[{time.strftime('%H:%M:%S')}] [DL] {stream_name}: "
                  f"Timed out ({DOWNLOAD_TIMEOUT}s)")
            self._safe_delete(output_path)
            return False
        except FileNotFoundError:
            print(f"[{time.strftime('%H:%M:%S')}] [DL] ERROR: ffmpeg not found!")
            return False

    # ── Quality Check ──

    def _check_quality(self, clip_path: str) -> tuple[bool, str]:
        """Quick blur + brightness check on sampled frames."""
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return False, "Cannot open clip"
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        duration = total_frames / fps
        if duration < CLIP_DURATION - 3:
            cap.release()
            return False, f"Too short ({duration:.1f}s < {CLIP_DURATION}s)"
        sample_points = [int(total_frames * p) for p in [0.1, 0.3, 0.5, 0.7]]

        for pos in sample_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
            bright_val = cv2.mean(gray)[0]

            if blur_val < 15:
                cap.release()
                return False, f"Too blurry ({blur_val:.1f})"
            if bright_val < 10 or bright_val > 250:
                cap.release()
                return False, f"Bad brightness ({bright_val:.1f})"

        cap.release()
        return True, "OK"

    # ── Main Download Loop (SINGLE thread, round-robin) ──

    def _download_loop(self):
        index = 0

        # Wait for URL prewarm to complete (max 120 sec, then start anyway)
        self._prewarm_done.wait(timeout=120)

        while self.is_running:
            # No streams added yet → wait
            if not self.streams:
                time.sleep(1)
                continue

            # Queue full → wait for YOLO to consume
            if self.download_queue.qsize() >= DOWNLOAD_QUEUE_TARGET:
                time.sleep(5)
                continue

            # Pick next stream (round-robin)
            stream = self.streams[index % len(self.streams)]
            index += 1
            name = stream["name"]

            # Skip streams with too many consecutive failures (try again next rotation)
            if self._failures.get(name, 0) >= 5:
                self._failures[name] = 0  # reset after skipping one rotation
                time.sleep(2)
                continue

            # Download
            self.clip_counter += 1
            safe_name = name.replace(' ', '_')
            clip_path = os.path.join(self.temp_dir, f"{safe_name}_{self.clip_counter}.mp4")

            if not self._download_clip(name, stream["url"], clip_path):
                self._failures[name] = self._failures.get(name, 0) + 1
                self.download_failures += 1
                time.sleep(3)
                continue

            # Reset failures on success
            self._failures[name] = 0
            self.clips_downloaded += 1

            # Quality check
            quality_ok, reason = self._check_quality(clip_path)
            if not quality_ok:
                print(f"[DL] {name}: Quality rejected — {reason}")
                self._safe_delete(clip_path)
                self.clips_quality_rejected += 1
                continue

            # Push to download queue for YOLO processing
            dl_clip = DownloadedClip(
                clip_path=clip_path,
                stream_name=name,
                line_config_file=stream["config"],
                imgsz=stream["imgsz"],
                confidence=stream["conf"],
            )
            self.download_queue.put(dl_clip)

    def _safe_delete(self, path: str):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def cleanup(self):
        self.is_running = False
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# ─────────────────────────────────────────────
#  YOLO WORKER — Single instance, owns the GPU
# ─────────────────────────────────────────────

class YOLOWorker:
    """
    Single worker thread that processes all downloaded clips with YOLO.
    Only ONE instance exists in the entire system — GPU safe, no tracker corruption.

    Reads:  download_queue (DownloadedClip from any stream)
    Writes: ready_queue (PreProcessedClip, approved clips only)
    """

    def __init__(self, model, download_queue: queue.Queue,
                 ready_queue: queue.Queue, debug_mode: bool = False):
        self.model = model
        self.download_queue = download_queue
        self.ready_queue = ready_queue
        self.debug_mode = debug_mode
        self.is_running = True

        # Current processing state (for debug GUI)
        self.current_stream_name = ""
        self.current_config_path = ""

        # Debug frame queue (--gui mode)
        self.debug_frame_queue = queue.Queue(maxsize=1) if debug_mode else None

        # Stats
        self.clips_processed = 0
        self.clips_approved = 0
        self.clips_rejected = 0

        self.thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="YOLOWorker",
        )
        self.thread.start()

    def _make_debug_callback(self, stream_name: str):
        """Create debug callback that pushes annotated frames to debug queue."""
        from core.counting import render_debug_frame

        def _cb(frame, frame_no, counter, results, counting_active):
            if self.debug_frame_queue is None:
                return
            annotated = frame.copy()
            render_debug_frame(
                annotated, counter, results, counting_active,
                frame_no, 25.0, stream_name=stream_name,
            )
            # Non-blocking swap: drop old, push new
            try:
                self.debug_frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.debug_frame_queue.put_nowait(annotated)
            except queue.Full:
                pass

        return _cb

    def _worker_loop(self):
        from core.counting import offline_count

        while self.is_running:
            # Get next downloaded clip (any stream — fastest wins)
            try:
                dl_clip = self.download_queue.get(timeout=5)
            except queue.Empty:
                continue

            # Update current state for debug GUI
            self.current_stream_name = dl_clip.stream_name
            self.current_config_path = dl_clip.line_config_file

            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [YOLO] Processing: {dl_clip.stream_name}")

            debug_cb = self._make_debug_callback(dl_clip.stream_name) if self.debug_mode else None

            start = time.time()
            result = offline_count(
                clip_path=dl_clip.clip_path,
                line_config_file=dl_clip.line_config_file,
                model=self.model,
                count_duration=COUNT_DURATION,
                confidence=dl_clip.confidence,
                imgsz=dl_clip.imgsz,
                debug_callback=debug_cb,
            )
            elapsed = time.time() - start
            self.clips_processed += 1

            if result is None:
                print(f"[YOLO] {dl_clip.stream_name}: Processing FAILED")
                self._safe_delete(dl_clip.clip_path)
                self.clips_rejected += 1
                continue

            count = result["result"]
            print(f"[YOLO] {dl_clip.stream_name}: count={count} | "
                  f"{result['total_frames_processed']} frames | {elapsed:.1f}s")

            if count < MIN_VEHICLE_COUNT:
                print(f"[YOLO] {dl_clip.stream_name}: REJECTED "
                      f"(count {count} < {MIN_VEHICLE_COUNT})")
                self._safe_delete(dl_clip.clip_path)
                self.clips_rejected += 1
                continue

            # APPROVED — create PreProcessedClip and push to ready queue
            clip = PreProcessedClip(
                clip_path=dl_clip.clip_path,
                stream_name=dl_clip.stream_name,
                result=count,
                detections=result["detections"],
                counting_events=result["counting_events"],
                total_frames=result["total_frames_processed"],
                clip_fps=result["clip_fps"],
            )
            self.clips_approved += 1
            print(f"[YOLO] {dl_clip.stream_name}: APPROVED (count={count}) | "
                  f"Ready: {self.ready_queue.qsize() + 1}/{READY_QUEUE_MAX}")

            # Blocks if ready_queue full (backpressure — slows everything down gracefully)
            self.ready_queue.put(clip)

    def get_debug_frame(self):
        """Get latest debug frame (non-blocking). Returns None if unavailable."""
        if self.debug_frame_queue is None:
            return None
        try:
            return self.debug_frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _safe_delete(self, path: str):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def cleanup(self):
        self.is_running = False


# ─────────────────────────────────────────────
#  PIPELINE — Coordinates everything
# ─────────────────────────────────────────────

class Pipeline:
    """
    Top-level coordinator for the parallel download + single YOLO pipeline.

    Usage:
        pipeline = Pipeline(model, debug_mode=True)
        pipeline.add_stream("Stream 1", "https://...", "line_configs/Stream_1.json")
        pipeline.add_stream("Stream 2", "https://...", "line_configs/Stream_2.json")

        clip = pipeline.get_next_clip(timeout=120)  # any stream, fastest wins
        pipeline.mark_clip_done(clip)
        pipeline.cleanup()
    """

    def __init__(self, model, debug_mode: bool = False):
        self.model = model
        self.debug_mode = debug_mode

        # Shared queues
        self.download_queue = queue.Queue(maxsize=DOWNLOAD_QUEUE_MAX)
        self.ready_queue = queue.Queue(maxsize=READY_QUEUE_MAX)

        # Sequential downloader (single thread, round-robin all streams)
        self.downloader = SequentialDownloader(download_queue=self.download_queue)

        # Track stream names (for stats/display)
        self.stream_names: list[str] = []

        # Single YOLO worker (owns the GPU)
        self.yolo_worker = YOLOWorker(
            model=model,
            download_queue=self.download_queue,
            ready_queue=self.ready_queue,
            debug_mode=debug_mode,
        )

    def add_stream(self, stream_name: str, stream_url: str,
                   line_config_file: str, imgsz: int = 1600,
                   confidence: float = 0.10):
        """Add a stream to the round-robin download rotation."""
        if stream_name in self.stream_names:
            return

        self.downloader.add_stream(stream_name, stream_url,
                                   line_config_file, imgsz, confidence)
        self.stream_names.append(stream_name)

        print(f"[PIPELINE] Added stream: {stream_name} "
              f"(total: {len(self.stream_names)} streams)")

    def prewarm_urls(self):
        """Pre-resolve all YouTube URLs before downloads begin.
        Call this AFTER all streams are added, BEFORE waiting for clips."""
        self.downloader.prewarm_urls()
        self.downloader._prewarm_done.set()  # Signal download loop to start

    def remove_stream(self, stream_name: str):
        """Remove a stream from the download rotation."""
        self.downloader.remove_stream(stream_name)
        if stream_name in self.stream_names:
            self.stream_names.remove(stream_name)

    def remove_streams_not_in(self, keep_names: list[str]):
        """Remove all streams NOT in the keep list (for time slot changes)."""
        to_remove = [n for n in self.stream_names if n not in keep_names]
        for name in to_remove:
            self.remove_stream(name)

    def get_next_clip(self, timeout: float = None) -> PreProcessedClip | None:
        """
        Get next approved clip from ANY stream (fastest wins).
        Blocks until available or timeout.
        """
        try:
            return self.ready_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def ready_count(self) -> int:
        return self.ready_queue.qsize()

    def mark_clip_done(self, clip: PreProcessedClip):
        """Delete clip file after it's been played in a round."""
        try:
            if os.path.exists(clip.clip_path):
                os.remove(clip.clip_path)
        except Exception:
            pass

    # ── Debug GUI support ──

    def get_debug_frame(self):
        """Get latest debug frame from YOLO worker (non-blocking)."""
        return self.yolo_worker.get_debug_frame()

    def get_debug_stream_info(self) -> dict:
        """Get info about what the YOLO worker is currently processing."""
        return {
            "current_stream": self.yolo_worker.current_stream_name,
            "config_path": self.yolo_worker.current_config_path,
        }

    # ── Stats ──

    def get_stats(self) -> dict:
        dl = self.downloader
        return {
            "downloaders": {
                "total_streams": len(self.stream_names),
                "downloaded": dl.clips_downloaded,
                "quality_rejected": dl.clips_quality_rejected,
                "failures": dl.download_failures,
            },
            "yolo": {
                "processed": self.yolo_worker.clips_processed,
                "approved": self.yolo_worker.clips_approved,
                "rejected": self.yolo_worker.clips_rejected,
                "current": self.yolo_worker.current_stream_name,
            },
            "ready_queue": self.ready_queue.qsize(),
            "download_queue": self.download_queue.qsize(),
        }

    def get_stats_summary(self) -> str:
        """One-line stats string for debug overlay."""
        s = self.get_stats()
        yolo = s["yolo"]
        dl = s["downloaders"]
        parts = [f"YOLO:{yolo['current'] or 'idle'}",
                 f"Ready:{s['ready_queue']}/{READY_QUEUE_MAX}",
                 f"DL_Q:{s['download_queue']}",
                 f"DL:{dl['downloaded']}",
                 f"OK:{yolo['approved']} Rej:{yolo['rejected']}"]
        return " | ".join(parts)

    def cleanup(self):
        """Stop everything and clean up temp files."""
        print("[PIPELINE] Cleaning up...")
        self.yolo_worker.cleanup()
        self.downloader.cleanup()
        self.stream_names.clear()
