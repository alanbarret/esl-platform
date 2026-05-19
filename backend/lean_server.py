"""
ESL Platform — Lean Server
Serves pre-rendered skeleton videos. No heavy libraries loaded.
Tiny memory footprint — no MediaPipe, no numpy, no JSON loading at startup.
"""
import json, os, uuid
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

BASE = Path(__file__).parent.parent
VIDEOS_DIR = BASE / "data" / "skeleton_videos"
SIGNS = {p.stem.upper() for p in VIDEOS_DIR.glob("*.mp4")}
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
                    {"role":"system","content":"Convert input to ESL gloss tokens. Output ONLY uppercase tokens separated by spaces. Max 8 tokens. Use: HELLO, HOW, YOU, THANK_YOU, YES, NO, GOOD, MORNING, DOCTOR, FAMILY, SCHOOL, WORK, SLEEP, OPEN, HELP, WATER, FOOD, HOME, WELCOME, GOODBYE, SORRY"},
                    {"role":"user","content":text}
                ],
                max_tokens=40, temperature=0.1,
            )
            tokens = r.choices[0].message.content.strip().upper().split()
            return [t for t in tokens if t][:8] or ["HELLO"]
        except: pass
    # Fallback
    ar = {"مرحبا":"HELLO","كيف":"HOW","حالك":"YOU","شكرا":"THANK_YOU","نعم":"YES","لا":"NO","دكتور":"DOCTOR","عائلة":"FAMILY","مدرسة":"SCHOOL","عمل":"WORK","نوم":"SLEEP"}
    en = {"hello":"HELLO","hi":"HELLO","how":"HOW","you":"YOU","thank":"THANK_YOU","yes":"YES","no":"NO","good":"GOOD","morning":"MORNING","doctor":"DOCTOR","family":"FAMILY","school":"SCHOOL","work":"WORK","sleep":"SLEEP","help":"HELP","water":"WATER","food":"FOOD","home":"HOME"}
    tokens = []
    for w in text.split():
        wc = w.strip(".,!?؟،").lower()
        if wc in ar: tokens.append(ar[wc])
        elif wc in en: tokens.append(en[wc])
        elif len(wc) > 2: tokens.append(wc.upper())
    return tokens[:8] or ["HELLO"]

# ── CORS headers ──────────────────────────────────────────────────────────────
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
        for k,v in CORS.items(): self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        for k,v in CORS.items(): self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path

        if p in ["/health", "/api/v1/health"]:
            self.send_json({"status":"ok","signs":len(SIGNS)})

        elif p == "/api/v1/skeleton-signs":
            self.send_json({"signs": sorted(SIGNS)})

        elif p.startswith("/api/v1/skeleton-video/"):
            sign = p.split("/")[-1].upper().replace(".MP4","")
            vid = VIDEOS_DIR / f"{sign}.mp4"
            if vid.exists():
                data = vid.read_bytes()
                self.send_response(200)
                for k,v in CORS.items(): self.send_header(k, v)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_json({"error": f"No video for {sign}"}, 404)

        elif p == "/api/v1/models/status":
            self.send_json({"gloss_model":{"loaded":True,"device":"lean"}})

        else:
            self.send_json({"error":"Not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        p = urlparse(self.path).path

        if p in ["/api/v1/translate", "/api/v1/gloss"]:
            text = body.get("text","")
            tokens = get_gloss(text)
            vids = [f"/api/v1/skeleton-video/{t}" for t in tokens if t.upper() in SIGNS]
            self.send_json({
                "request_id": str(uuid.uuid4())[:8],
                "input_text": text,
                "gloss_tokens": tokens,
                "gloss_string": " ".join(tokens),
                "skeleton_videos": vids,
                "video_url": vids[0] if vids else None,
                "status": "completed",
            })
        else:
            self.send_json({"error":"Not found"}, 404)

if __name__ == "__main__":
    port = 8001
    print(f"[ESL] Lean server on :{port} | {len(SIGNS)} signs | OpenAI={'yes' if OPENAI_API_KEY else 'no'}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
