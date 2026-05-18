"""
ESL Platform — Demo Server with OpenAI Gloss Generation
Converts Arabic/English text → ESL gloss using GPT-4o,
then animates the Arab Man GLB avatar.
"""
import json, os, uuid, math
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ── Dataset ───────────────────────────────────────────────────────────────────
SIGNS_PATH = Path(__file__).parent.parent / "data" / "raw" / "uae_signs_raw.json"
signs_data = json.loads(SIGNS_PATH.read_text(encoding="utf-8")) if SIGNS_PATH.exists() else []
KNOWN_GLOSSES = {s["english"].upper().replace(" ", "_") for s in signs_data}
KNOWN_GLOSSES.update({s["english"].upper() for s in signs_data})

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def gloss_with_openai(text: str) -> list[str]:
    """Use GPT-4o to convert Arabic/English text to ESL gloss sequence."""
    if not OPENAI_API_KEY:
        return gloss_fallback(text)

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system = """You are an Emirati Sign Language (ESL) interpreter.
Convert the input sentence into an ESL gloss sequence.

Rules:
- Output ONLY uppercase gloss tokens separated by spaces
- Use simplified sign order (topic-comment structure)
- Remove articles, prepositions, conjugations
- Use common ESL glosses like: HELLO, YOU, HOW, THANK_YOU, YES, NO, GOOD, MORNING, EVENING, NAME, I, PLEASE, HELP, WATER, FOOD, WELCOME, GOODBYE, SORRY, DOCTOR, HOSPITAL, SCHOOL, WORK, FAMILY, HOME, TIME, TODAY, TOMORROW, MONEY, BUY, GIVE, GO, COME, SEE, KNOW, WANT, NEED
- For unknown words, use the English word in CAPS
- Output 1-8 gloss tokens maximum

Examples:
Input: "مرحبا كيف حالك" → HELLO YOU HOW
Input: "شكرا جزيلا" → THANK_YOU VERY_MUCH
Input: "Hello, how are you?" → HELLO YOU HOW
Input: "I need help please" → I NEED HELP PLEASE
Input: "Where is the hospital?" → HOSPITAL WHERE"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            max_tokens=60,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        tokens = [t.strip() for t in raw.upper().split() if t.strip()]
        return tokens[:8] if tokens else gloss_fallback(text)
    except Exception as e:
        print(f"[OpenAI Error] {e} — falling back to rule-based")
        return gloss_fallback(text)


def gloss_fallback(text: str) -> list[str]:
    """Rule-based fallback when OpenAI unavailable."""
    arabic_map = {
        "مرحبا": "HELLO", "كيف": "HOW", "حالك": "YOU", "شكرا": "THANK_YOU",
        "نعم": "YES", "لا": "NO", "صباح": "MORNING", "الخير": "GOOD",
        "اسمي": "MY_NAME", "انا": "I", "مساء": "EVENING", "من": "WHO",
        "اين": "WHERE", "متى": "WHEN", "ماذا": "WHAT", "لماذا": "WHY",
    }
    eng_map = {
        "hello": "HELLO", "hi": "HELLO", "how": "HOW", "you": "YOU",
        "thank": "THANK_YOU", "thanks": "THANK_YOU", "yes": "YES", "no": "NO",
        "good": "GOOD", "morning": "MORNING", "my": "MY", "name": "NAME",
        "i": "I", "am": "AM", "please": "PLEASE", "help": "HELP",
        "water": "WATER", "food": "FOOD", "welcome": "WELCOME",
        "bye": "GOODBYE", "goodbye": "GOODBYE", "sorry": "SORRY",
        "where": "WHERE", "when": "WHEN", "what": "WHAT", "who": "WHO",
    }
    words = text.strip().split()
    glosses = []
    for w in words:
        wc = w.strip(".,!?؟،")
        if wc in arabic_map:
            glosses.append(arabic_map[wc])
        else:
            mapped = eng_map.get(wc.lower(), wc.upper() if len(wc) > 2 else "")
            if mapped:
                glosses.append(mapped)
    return [g for g in glosses if g] or ["HELLO"]


# ── Animation ─────────────────────────────────────────────────────────────────

def quat(ax, ay, az):
    cx,cy,cz = math.cos(ax/2),math.cos(ay/2),math.cos(az/2)
    sx,sy,sz = math.sin(ax/2),math.sin(ay/2),math.sin(az/2)
    return [sx*cy*cz+cx*sy*sz, cx*sy*cz-sx*cy*sz, cx*cy*sz+sx*sy*cz, cx*cy*cz-sx*sy*sz]

# Angles are ADDITIVE offsets from rest pose (applied via multiply in Three.js)
# X = forward/back tilt, Y = twist, Z = raise/lower (positive = raise for right arm)
# Real poses from UAE sign videos (spine/head removed to prevent bending)
SIGN_POSES = {
    "THANK_YOU": {"RightArm": (-0.4, 0, -0.3), "RightForeArm": (-0.3, 0, 0.3)},
    "YES": {"RightArm": (-0.2, 0, -0.2), "RightForeArm": (-0.2, 0, 0.2)},
    "NO": {"RightArm": (-0.2, 0, -0.2), "RightForeArm": (-0.2, 0, 0.25)},
    "GOOD": {"RightArm": (-0.3, 0, -0.3), "RightForeArm": (-0.25, 0, 0.3)},
    "PLEASE": {"RightArm": (-0.4, 0, -0.2), "RightForeArm": (-0.3, 0, 0.3)},
    "WELCOME": {"RightArm": (-0.3, 0, -0.5), "LeftArm": (-0.3, 0, 0.5), "RightForeArm": (-0.25, 0, 0.3), "LeftForeArm": (-0.25, 0, -0.3)},
    "SORRY": {"RightArm": (-0.3, 0, -0.2), "RightForeArm": (-0.25, 0, 0.25)},
    "WHERE": {"RightArm": (-0.3, 0, -0.3), "RightForeArm": (-0.3, 0, 0.4)},
    "WHAT": {"RightArm": (-0.2, 0, -0.3), "RightForeArm": (-0.25, 0, 0.3), "LeftArm": (-0.2, 0, 0.3)},
    "NAME": {"RightArm": (-0.2, 0, -0.25), "RightForeArm": (-0.3, 0, 0.4)},
    "WATER": {"RightArm": (-0.3, 0, -0.2), "RightForeArm": (-0.25, 0, 0.3)},
    "FOOD": {"RightArm": (-0.5, 0, -0.2), "RightForeArm": (-0.3, 0, 0.5)},
    "HOME": {"RightArm": (-0.4, 0, -0.3), "LeftArm": (-0.4, 0, 0.3)},
    "MORNING": {"RightArm": (-0.4, 0, -0.4), "RightForeArm": (-0.25, 0, 0.3)},
    "GOODBYE": {"RightArm": (-0.3, 0, -0.5), "RightForeArm": (-0.2, 0, 0.3)},
    "HELP": {"RightArm": (-0.3, 0, -0.4), "LeftArm": (-0.3, 0, 0.4)},
}

DEFAULT_POSE = {"RightArm": (-0.3, 0, -0.2), "RightForeArm": (-0.25, 0, 0.3)}


ALL_BONES = {
    "Hips","Spine","Spine1","Spine2","Neck","Head",
    "LeftShoulder","LeftArm","LeftForeArm","LeftHand",
    "RightShoulder","RightArm","RightForeArm","RightHand",
    "LeftHandThumb1","LeftHandThumb2","LeftHandIndex1","LeftHandIndex2",
    "LeftHandMiddle1","LeftHandRing1","LeftHandPinky1",
    "RightHandThumb1","RightHandThumb2","RightHandIndex1","RightHandIndex2",
    "RightHandMiddle1","RightHandRing1","RightHandPinky1",
}

REST_Q = [0.0, 0.0, 0.0, 1.0]

def make_animation(gloss_tokens: list[str]) -> dict:
    fps = 30
    sign_dur = 1.2
    total_dur = len(gloss_tokens) * sign_dur

    bone_tracks: dict[str, dict] = {b: {"times": [], "rotations": []} for b in ALL_BONES}

    for idx, gloss in enumerate(gloss_tokens):
        t0 = idx * sign_dur
        t_wind = t0 + sign_dur * 0.25
        t_peak = t0 + sign_dur * 0.55
        t_hold = t0 + sign_dur * 0.72
        t_end  = t0 + sign_dur

        # Match pose: exact → first word → default
        pose = SIGN_POSES.get(gloss) or SIGN_POSES.get(gloss.split("_")[0]) or DEFAULT_POSE

        for bone in ALL_BONES:
            angles = pose.get(bone, (0, 0, 0))
            peak_q = quat(angles[0], angles[1], angles[2])
            wind_q = quat(angles[0]*0.35, angles[1]*0.35, angles[2]*0.35)
            tr = bone_tracks[bone]
            tr["times"] += [t0, t_wind, t_peak, t_hold, t_end]
            tr["rotations"] += REST_Q + wind_q + peak_q + peak_q + REST_Q

    channels, samplers = [], []
    for bone, data in bone_tracks.items():
        samplers.append({"input": data["times"], "interpolation": "LINEAR", "output": data["rotations"]})
        channels.append({"sampler": len(samplers)-1, "target": {"node": bone, "path": "rotation"}})

    return {"name": "_".join(gloss_tokens[:5]), "channels": channels, "samplers": samplers, "duration": total_dur, "fps": fps}


# ── HTTP Server ───────────────────────────────────────────────────────────────

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type": "application/json",
}

class ESLHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[API] {fmt % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        for k, v in CORS.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in CORS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path
        if p in ["/api/v1/health", "/health"]:
            self.send_json({"status": "ok", "service": "ESL Platform API",
                            "ai": "openai" if OPENAI_API_KEY else "rule-based",
                            "signs_loaded": len(signs_data)})
        elif p == "/api/v1/models/status":
            self.send_json({"gloss_model": {"loaded": True, "device": "openai" if OPENAI_API_KEY else "cpu-rules"}})
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        p = urlparse(self.path).path

        if p == "/api/v1/translate":
            text = body.get("text", "")
            print(f"[Translate] {text!r}")
            gloss_tokens = gloss_with_openai(text)
            print(f"[Gloss] {gloss_tokens}")
            animation = make_animation(gloss_tokens)
            self.send_json({
                "request_id": str(uuid.uuid4())[:8],
                "input_text": text,
                "detected_language": "ar" if any(ord(c) > 0x600 for c in text) else "en",
                "gloss_tokens": gloss_tokens,
                "total_duration": animation["duration"],
                "status": "completed",
                "gltf_animation": animation,
                "video_url": None,
            })
        elif p == "/api/v1/gloss":
            text = body.get("text", "")
            gloss = gloss_with_openai(text)
            self.send_json({"input_text": text, "gloss_tokens": gloss, "gloss_string": " ".join(gloss)})
        else:
            self.send_json({"error": "Not found"}, 404)


if __name__ == "__main__":
    port = 8001
    ai_mode = "OpenAI GPT-4o-mini" if OPENAI_API_KEY else "rule-based fallback (set OPENAI_API_KEY)"
    print(f"ESL Platform API :{port}")
    print(f"AI Mode: {ai_mode}")
    print(f"Signs loaded: {len(signs_data)}")
    HTTPServer(("0.0.0.0", port), ESLHandler).serve_forever()
