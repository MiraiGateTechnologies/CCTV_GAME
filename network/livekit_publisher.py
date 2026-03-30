"""
================================================================
  LIVEKIT PUBLISHER — Publish game video frames to LiveKit room
================================================================

Replaces MJPEG streaming with WebRTC via LiveKit.
  - Receives frames from scheduler (same update_frame interface)
  - Publishes to LiveKit room as a video track
  - Frontend subscribes via LiveKit JS SDK

Usage:
  publisher = LiveKitPublisher(
      livekit_url="ws://localhost:7880",
      api_key="cctv_game_key",
      api_secret="cctv_game_secret_change_in_production",
      room_name="cctv-game",
  )
  publisher.start()

  # In scheduler loop:
  publisher.update_frame(frame)  # same interface as web_server.update_frame

  # Cleanup:
  publisher.stop()
"""

import asyncio
import threading
import time
import numpy as np
import cv2
from typing import Optional

from livekit import rtc, api


# ── Config ─────────────────────────────────────────────────────
DEFAULT_LIVEKIT_URL = "ws://localhost:7880"
DEFAULT_API_KEY = "devkey"
DEFAULT_API_SECRET = "cctv-game-secret-key-32chars-min!"
DEFAULT_ROOM = "cctv-game"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 25


class LiveKitPublisher:
    """
    Publishes OpenCV frames to a LiveKit room as a video track.

    Thread-safe: update_frame() can be called from any thread.
    The async publish loop runs in its own event loop thread.
    """

    def __init__(
        self,
        livekit_url: str = DEFAULT_LIVEKIT_URL,
        api_key: str = DEFAULT_API_KEY,
        api_secret: str = DEFAULT_API_SECRET,
        room_name: str = DEFAULT_ROOM,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
    ):
        self.livekit_url = livekit_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.room_name = room_name
        self.width = width
        self.height = height
        self.fps = fps

        # Frame buffer (thread-safe via lock)
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_new = False

        # Async event loop in background thread
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False

    def update_frame(self, frame: np.ndarray):
        """
        Update the latest frame to publish. Thread-safe.
        Same interface as web_server.update_frame() — drop-in replacement.
        Just stores reference + sets flag. Resize happens in publisher thread.
        """
        with self._frame_lock:
            self._frame = frame
            self._frame_new = True

    def start(self):
        """Start the publisher in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="LiveKit-Publisher"
        )
        self._thread.start()
        print(f"[LIVEKIT] Publisher started → {self.livekit_url} room={self.room_name}")

    def stop(self):
        """Stop the publisher."""
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def generate_viewer_token(self, identity: str = "viewer") -> str:
        """
        Generate a JWT token for a viewer to join the room.
        Called by web_server for /api/livekit/token endpoint.
        """
        token = api.AccessToken(self.api_key, self.api_secret)
        token.with_identity(identity)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=self.room_name,
        ))
        return token.to_jwt()

    def _run_loop(self):
        """Background thread: runs the async publish loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._publish_loop())
        except Exception as e:
            print(f"[LIVEKIT] Publisher loop error: {e}")
        finally:
            self._connected = False
            self._loop.close()

    async def _publish_loop(self):
        """Async: connect to room, create video track, publish frames."""
        room = rtc.Room()

        # Generate publisher token
        token = api.AccessToken(self.api_key, self.api_secret)
        token.with_identity("cctv-game-publisher")
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=self.room_name,
            room_create=True,
            can_publish=True,
        ))
        jwt_token = token.to_jwt()

        # Connect to LiveKit server (with retry — server needs ~5-10s to start)
        print(f"[LIVEKIT] Connecting to {self.livekit_url}...")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                await room.connect(self.livekit_url, jwt_token)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"[LIVEKIT] Connection attempt {attempt + 1}/{max_retries} failed, "
                          f"retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"[LIVEKIT] Connection failed after {max_retries} attempts: {e}")
                    print("[LIVEKIT] MJPEG fallback active. Start LiveKit: .\\start_livekit.bat")
                    return

        self._connected = True
        print(f"[LIVEKIT] Connected! Room: {self.room_name}")

        # Create video source + track
        source = rtc.VideoSource(self.width, self.height)
        track = rtc.LocalVideoTrack.create_video_track("game-stream", source)

        # Publish track
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
        )
        await room.local_participant.publish_track(track, options)
        print(f"[LIVEKIT] Video track published: {self.width}x{self.height} @ {self.fps}fps")

        # Frame publish loop
        frame_interval = 1.0 / self.fps
        while self._running:
            start = time.perf_counter()

            with self._frame_lock:
                frame = self._frame
                is_new = self._frame_new
                self._frame_new = False

            if frame is not None and is_new:
                h, w = frame.shape[:2]
                if w != self.width or h != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                lk_frame = rtc.VideoFrame(
                    self.width, self.height,
                    rtc.VideoBufferType.RGBA, rgba.tobytes(),
                )
                source.capture_frame(lk_frame)

            # Frame pacing
            elapsed = time.perf_counter() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        # Cleanup
        await room.disconnect()
        self._connected = False
        print("[LIVEKIT] Disconnected")


# ── Module-level singleton (optional, for drop-in use) ─────────
_publisher: Optional[LiveKitPublisher] = None


def init_publisher(
    livekit_url: str = DEFAULT_LIVEKIT_URL,
    api_key: str = DEFAULT_API_KEY,
    api_secret: str = DEFAULT_API_SECRET,
    room_name: str = DEFAULT_ROOM,
) -> LiveKitPublisher:
    """Initialize and start the global publisher singleton."""
    global _publisher
    _publisher = LiveKitPublisher(
        livekit_url=livekit_url,
        api_key=api_key,
        api_secret=api_secret,
        room_name=room_name,
    )
    _publisher.start()
    return _publisher


def update_frame(frame: np.ndarray):
    """Global update_frame — same interface as web_server.update_frame."""
    if _publisher:
        _publisher.update_frame(frame)


def get_publisher() -> Optional[LiveKitPublisher]:
    """Get the global publisher instance."""
    return _publisher
