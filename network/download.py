"""
==============================================
  VIDEO DOWNLOADER — Save live stream clips
  Saves to: videos/ folder automatically
==============================================

Usage:
  python download.py --url "YOUTUBE_URL" --name video1
  python download.py --url "YOUTUBE_URL" --name video1 --duration 300
  python download.py --url "YOUTUBE_URL" --name traffic_cam_morning --duration 600

Then run counter:
  python main.py --video video1
  python main.py --video traffic_cam_morning
"""

# ┌─────────────────────────────────────────────────────────────────────┐
# │  FILE   : download.py                                               │
# │  PURPOSE: YouTube/stream se video download karna + URL resolve      │
# │           Ye file STANDALONE hai — kisi aur file pe depend nahi     │
# ├─────────────────────────────────────────────────────────────────────┤
# │  CHANGE GUIDE — Agar kuch change karna ho:                          │
# │                                                                     │
# │  ► Videos folder location badalni ho                                │
# │    → VIDEOS_DIR variable (line ~25)                                 │
# │                                                                     │
# │  ► Download quality change karni ho (1080p → 720p etc.)            │
# │    → download() function ke andar cmd list (line ~106-112)          │
# │      "-f bestvideo[ext=mp4][height<=1080]..." wali line change karo │
# │                                                                     │
# │  ► Stream URL resolve timeout badhaana ho (default: 30s)            │
# │    → get_stream_url() function, timeout=30 (line ~31)               │
# │                                                                     │
# │  ► Clip download timeout badhaana ho (default: 120s)                │
# │    → extract_clip() function, timeout=120 (line ~69)                │
# │                                                                     │
# │  ► Clip quality change karni ho (720p → 480p etc.)                  │
# │    → extract_clip() function ke andar cmd list                      │
# │      "best[ext=mp4][height<=720]" wali line change karo (line ~60)  │
# └─────────────────────────────────────────────────────────────────────┘

import subprocess
import sys
import os
import argparse
from datetime import datetime

VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")


def get_stream_url(youtube_url: str) -> str:
    """Resolve a YouTube URL to a direct playable stream URL via yt-dlp."""
    print("[INFO] YouTube stream URL fetch kar raha hoon...")
    try:
        cmd = [sys.executable, "-m", "yt_dlp", "--get-url", "-f", "best[ext=mp4]/best", "--no-playlist", youtube_url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        url = result.stdout.strip().split('\n')[0]
        if url:
            print("[OK] Stream URL mila!")
            return url
        print(f"[ERROR] URL nahi mila: {result.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] yt-dlp install nahi hai!")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("[ERROR] Timeout!")
        sys.exit(1)


def extract_clip(stream_url, duration_secs, output_path):
    """Download a clip of `duration_secs` from a live stream URL using yt-dlp.

    Returns True on success, False on failure.
    """
    from datetime import datetime as _dt
    def _ts():
        return _dt.now().strftime("%H:%M:%S")

    if os.path.exists(output_path):
        os.remove(output_path)

    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4][height<=720]/best[ext=mp4]/best",
        "--no-playlist",
        "--download-sections", f"*0-{duration_secs}",
        "--force-keyframes-at-cuts",
        "-o", output_path,
        stream_url,
    ]
    try:
        print(f"[{_ts()}] Downloading {duration_secs}s clip...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[{_ts()}] Clip ready: {size_mb:.1f} MB")
            return True
        else:
            print(f"[{_ts()}] ERROR: Clip download failed")
            print(f"  stderr: {result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[{_ts()}] ERROR: Download timed out (120s)")
        return False
    except FileNotFoundError:
        print(f"[{_ts()}] ERROR: yt-dlp not found! pip install yt-dlp")
        return False


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def download(url: str, name: str, duration_secs: int = None):
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    output_path = os.path.join(VIDEOS_DIR, f"{name}.mp4")

    if os.path.exists(output_path):
        overwrite = input(f"[?] '{name}.mp4' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            log("Cancelled.")
            return

    log(f"Downloading → videos/{name}.mp4")
    log(f"URL: {url}")
    if duration_secs:
        log(f"Duration limit: {duration_secs}s ({duration_secs//60}m {duration_secs%60}s)")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", output_path,
    ]
    if duration_secs:
        cmd += ["--download-sections", f"*0-{duration_secs}"]
    cmd.append(url)

    try:
        subprocess.run(cmd, check=True)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        log(f"Done! Saved: videos/{name}.mp4  ({size_mb:.1f} MB)")
        log(f"Run counter: python main.py --video {name}")
    except subprocess.CalledProcessError:
        log("ERROR: Download failed. Check URL.")
    except FileNotFoundError:
        log("ERROR: yt-dlp not found! Run: pip install yt-dlp")


def list_videos():
    if not os.path.exists(VIDEOS_DIR):
        print("No videos/ folder yet.")
        return
    files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4")]
    if not files:
        print("videos/ folder is empty.")
        return
    print("\nAvailable videos:")
    for i, f in enumerate(sorted(files), 1):
        path = os.path.join(VIDEOS_DIR, f)
        mb = os.path.getsize(path) / (1024 * 1024)
        name = f[:-4]  # strip .mp4
        print(f"  {i}. {name:<30} ({mb:.1f} MB)")
    print(f"\nRun: python main.py --video <name>")


def main():
    parser = argparse.ArgumentParser(description="Download YouTube/live stream clips to videos/ folder")
    parser.add_argument("--url",      type=str, help="YouTube or stream URL")
    parser.add_argument("--name",     type=str, help="Video name (e.g. video1, traffic_morning)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Max seconds to download (e.g. 300 = 5 min). Default: full video")
    parser.add_argument("--list",     action="store_true", help="List downloaded videos")
    args = parser.parse_args()

    if args.list:
        list_videos()
        return

    if not args.url or not args.name:
        parser.print_help()
        print("\nExamples:")
        print('  python download.py --url "https://youtube.com/live/..." --name video1 --duration 300')
        print('  python download.py --list')
        sys.exit(0)

    download(args.url, args.name, args.duration)


if __name__ == "__main__":
    main()
