"""
ESL Platform — Lean Server
Serves pre-rendered skeleton videos + stitched sentence videos.
Tiny memory footprint — no MediaPipe, no numpy at startup.
"""
import json, os, uuid
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from video_stitcher import get_stitched_video, STITCHED_DIR, VIDEOS_DIR

BASE     = Path(__file__).parent.parent
SIGNS    = {p.stem.upper() for p in VIDEOS_DIR.glob("*.mp4")}
print(f"[ESL] {len(SIGNS)} skeleton videos ready")

# ── OpenAI gloss ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def get_gloss(text: str) -> list[str]:
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are an Emirati Sign Language (ESL) interpreter. "
                        "Convert input text to ESL gloss tokens. "
                        "Output ONLY uppercase tokens separated by spaces. Max 8 tokens. "
                        "Known sign tokens (use these exactly): HOW_ARE_YOU, DOCTOR, FAMILY, SCHOOL, WORK, "
                        "MORNING, SLEEP, OPEN, PUSH, RELAX, RECOMMENDED, PLAYS, PLAYS_GUITAR, WATERING, SEW, "
                        "SENDS, SELL, RUSH, REMOVE, PULLS, PLOW, SHOUTS, RUBBING, HELPS, "
                        "PHOTOGRAPHER, OUT, HOME_LAWN, WOW, PUNISHMENT, REQUESTS. "
                        "IMPORTANT: For Arabic words or unknown English words, output the ARABIC word itself "
                        "in Arabic script as a single token (e.g. مرحبا) — do NOT transliterate. "
                        "The system will finger-spell Arabic words character by character automatically."
                    )},
                    {"role": "user", "content": text}
                ],
                max_tokens=50, temperature=0.1,
            )
            tokens = r.choices[0].message.content.strip().upper().split()
            return [t.strip('.,!?') for t in tokens if t.strip('.,!?')][:8] or ["HOW_ARE_YOU"]
        except Exception as e:
            print(f"[OpenAI] Error: {e}")

    # Rule-based fallback
    ar = {
        "مرحبا": "HOW_ARE_YOU", "كيف": "HOW_ARE_YOU", "حالك": "HOW_ARE_YOU",
        "دكتور": "DOCTOR", "طبيب": "DOCTOR",
        "عائلة": "FAMILY", "مدرسة": "SCHOOL", "عمل": "WORK",
        "صباح": "MORNING", "نوم": "SLEEP", "مساعدة": "HELPS",
    }
    en = {
        "hello": "HOW_ARE_YOU", "hi": "HOW_ARE_YOU", "how": "HOW_ARE_YOU",
        "doctor": "DOCTOR", "family": "FAMILY", "school": "SCHOOL",
        "work": "WORK", "morning": "MORNING", "sleep": "SLEEP",
        "help": "HELPS", "open": "OPEN", "push": "PUSH", "relax": "RELAX",
    }
    tokens = []
    for w in text.split():
        wc = w.strip(".,!?؟،").lower()
        if wc in ar: tokens.append(ar[wc])
        elif wc in en: tokens.append(en[wc])
        elif len(wc) > 1: tokens.append(wc.upper())
    return tokens[:8] or ["HOW_ARE_YOU"]

# ── CORS ──────────────────────────────────────────────────────────────────────
CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[API] {fmt % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        for k, v in CORS.items(): self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, path: Path, mime: str, cache: int = 86400):
        size = path.stat().st_size
        self.send_response(200)
        for k, v in CORS.items(): self.send_header(k, v)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", f"public, max-age={cache}")
        self.end_headers()
        # Stream in 64KB chunks — avoids loading whole file into RAM
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(65536)
                if not chunk: break
                self.wfile.write(chunk)

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in CORS.items(): self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path

        if p in ["/health", "/api/v1/health"]:
            self.send_json({"status": "ok", "signs": len(SIGNS),
                            "ai": "openai" if OPENAI_API_KEY else "rules"})

        elif p == "/api/v1/skeleton-signs":
            self.send_json({"signs": sorted(SIGNS)})

        elif p.startswith("/api/v1/skeleton-video/"):
            sign = p.split("/")[-1].upper().replace(".MP4", "")
            vid = VIDEOS_DIR / f"{sign}.mp4"
            if vid.exists(): self.serve_file(vid, "video/mp4")
            else: self.send_json({"error": f"No video for {sign}"}, 404)

        elif p.startswith("/api/v1/video/"):
            # Serve stitched videos
            name = p.split("/")[-1]
            vid = STITCHED_DIR / name
            if vid.exists(): self.serve_file(vid, "video/mp4", cache=3600)
            else: self.send_json({"error": "Video not found"}, 404)

        elif p == "/api/v1/models/status":
            self.send_json({"gloss_model": {"loaded": True, "device": "lean"}})

        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        p = urlparse(self.path).path

        if p == "/api/v1/gloss":
            text = body.get("text", "")
            tokens = get_gloss(text)
            self.send_json({"gloss_tokens": tokens, "gloss_string": " ".join(tokens)})

        elif p == "/api/v1/translate":
            text = body.get("text", "")
            tokens = get_gloss(text)
            print(f"[Translate] {text!r} → {tokens}")

            # Stitch all tokens into one video (with letter-by-letter fallback)
            stitched = get_stitched_video(tokens)
            if stitched:
                vid_name = Path(stitched).name
                video_url = f"/api/v1/video/{vid_name}"
            else:
                video_url = None

            self.send_json({
                "request_id": str(uuid.uuid4())[:8],
                "input_text": text,
                "gloss_tokens": tokens,
                "gloss_string": " ".join(tokens),
                "video_url": video_url,
                "skeleton_videos": [video_url] if video_url else [],
                "status": "completed",
            })

        else:
            self.send_json({"error": "Not found"}, 404)


if __name__ == "__main__":
    port = 8001
    print(f"[ESL] Lean server :{port} | signs={len(SIGNS)} | openai={'yes' if OPENAI_API_KEY else 'no'}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
