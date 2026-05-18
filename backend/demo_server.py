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
# Poses extracted from real UAE Sign Language videos via MediaPipe Holistic
SIGN_POSES = {
    "HELLO": {"RightArm": (-0.163, -0.092, -0.312), "RightForeArm": (-0.236, 0.023, 0.437), "LeftArm": (-0.278, 0.091, 0.292), "LeftForeArm": (0.083, -0.021, -0.469), "Head": (-1.534, -0.002, -0.003), "Spine2": (-2.034, 0.009, 0.005)},
    "HOW": {"RightArm": (-0.163, -0.092, -0.312), "RightForeArm": (-0.236, 0.023, 0.437), "LeftArm": (-0.278, 0.091, 0.292), "LeftForeArm": (0.083, -0.021, -0.469), "Head": (-1.534, -0.002, -0.003), "Spine2": (-2.034, 0.009, 0.005)},
    "YOU": {"RightArm": (-0.163, -0.092, -0.312), "RightForeArm": (-0.236, 0.023, 0.437), "LeftArm": (-0.278, 0.091, 0.292), "LeftForeArm": (0.083, -0.021, -0.469), "Head": (-1.534, -0.002, -0.003), "Spine2": (-2.034, 0.009, 0.005)},
    "DOCTOR": {"RightArm": (-2.104, -0.053, -0.145), "RightForeArm": (-1.58, 0.371, 0.83), "LeftArm": (-0.785, 0.117, 0.253), "LeftForeArm": (-0.613, -0.016, -0.418), "Head": (-1.514, -0.0, -0.0), "Spine2": (-2.667, -0.109, -0.021)},
    "WORK": {"RightArm": (-1.991, -0.128, -0.155), "RightForeArm": (-1.501, 0.199, 0.656), "LeftArm": (-1.851, 0.078, 0.158), "LeftForeArm": (-1.511, -0.239, -0.702), "Head": (-1.491, -0.01, -0.015), "Spine2": (-2.447, 0.02, 0.006)},
    "FAMILY": {"RightArm": (-1.877, -0.155, -0.167), "RightForeArm": (-1.539, 0.234, 0.69), "LeftArm": (-2.018, 0.114, 0.152), "LeftForeArm": (-1.539, -0.23, -0.684), "Head": (-1.452, -0.001, -0.002), "Spine2": (-2.251, -0.002, -0.001)},
    "SCHOOL": {"RightArm": (0.532, -0.027, -0.335), "RightForeArm": (1.919, 0.079, 0.412), "LeftArm": (0.643, 0.037, 0.449), "LeftForeArm": (1.835, 0.119, -0.544), "Head": (-1.37, -0.043, -0.074), "Spine2": (-2.035, -0.042, -0.023)},
    "SLEEP": {"RightArm": (-2.215, -0.106, -0.139), "RightForeArm": (-1.692, 0.217, 0.638), "LeftArm": (-1.045, 0.074, 0.201), "LeftForeArm": (-0.663, -0.014, -0.412), "Head": (-1.52, -0.062, -0.091), "Spine2": (-2.958, 0.233, 0.01)},
    "OPEN": {"RightArm": (-0.557, -0.208, -0.374), "RightForeArm": (-1.069, 0.091, 0.575), "LeftArm": (-0.688, 0.124, 0.271), "LeftForeArm": (-1.024, -0.009, -0.418), "Head": (-1.53, -0.017, -0.025), "Spine2": (-2.513, 0.055, 0.015)},
    "OUT": {"RightArm": (-2.157, -0.192, -0.142), "RightForeArm": (-1.684, 0.168, 0.59), "LeftArm": (-0.73, 0.121, 0.263), "LeftForeArm": (-0.492, -0.01, -0.376), "Head": (-1.505, -0.08, -0.119), "Spine2": (-2.888, 0.053, 0.004)},
    "PLAYS": {"RightArm": (-1.968, -0.323, -0.174), "RightForeArm": (-1.543, 0.124, 0.561), "LeftArm": (-2.025, 0.246, 0.16), "LeftForeArm": (-1.543, -0.118, -0.555), "Head": (-1.473, 0.001, 0.001), "Spine2": (-2.518, 0.048, 0.013)},
    "SELL": {"RightArm": (-2.115, -0.307, -0.149), "RightForeArm": (-1.542, 0.204, 0.654), "LeftArm": (-2.051, 0.342, 0.161), "LeftForeArm": (-1.517, -0.241, -0.703), "Head": (-1.482, 0.002, 0.003), "Spine2": (-2.483, -0.024, -0.007)},
    "PUSH": {"RightArm": (-2.151, -0.083, -0.143), "RightForeArm": (-1.44, 0.093, 0.535), "LeftArm": (-1.998, 0.153, 0.156), "LeftForeArm": (-1.397, -0.116, -0.57), "Head": (-1.487, -0.019, -0.028), "Spine2": (-2.847, 0.097, 0.01)},
    "REMOVE": {"RightArm": (-2.231, -0.49, -0.121), "RightForeArm": (-1.506, 0.244, 0.709), "LeftArm": (-2.057, 0.084, 0.148), "LeftForeArm": (-1.603, -0.213, -0.651), "Head": (-1.429, 0.035, 0.056), "Spine2": (-2.559, 0.259, 0.065)},
    "RELAX": {"RightArm": (-0.758, -0.088, -0.232), "RightForeArm": (-1.012, 0.054, 0.51), "LeftArm": (-0.745, 0.014, 0.171), "LeftForeArm": (-0.995, -0.021, -0.441), "Head": (-1.534, 0.001, 0.001), "Spine2": (-2.616, 0.041, 0.009)},
    "RUSH": {"RightArm": (-2.043, -0.211, -0.155), "RightForeArm": (-1.391, 0.276, 0.784), "LeftArm": (-1.972, 0.06, 0.151), "LeftForeArm": (-1.524, -0.203, -0.656), "Head": (-1.527, -0.005, -0.007), "Spine2": (-2.613, 0.061, 0.013)},
    "SEW": {"RightArm": (-2.071, -0.163, -0.151), "RightForeArm": (-1.561, 0.362, 0.827), "LeftArm": (-2.156, 0.228, 0.142), "LeftForeArm": (-1.507, -0.452, -0.954), "Head": (-1.518, -0.034, -0.05), "Spine2": (-3.095, 0.431, -0.01)},
    "SHOUTS": {"RightArm": (-1.619, -0.48, -0.28), "RightForeArm": (-1.369, 0.043, 0.474), "LeftArm": (-0.539, 0.088, 0.254), "LeftForeArm": (-0.967, -0.061, -0.528), "Head": (-1.508, 0.003, 0.005), "Spine2": (-2.538, 0.103, 0.027)},
    "RECOMMENDED": {"RightArm": (-2.101, -0.166, -0.148), "RightForeArm": (-1.722, 0.195, 0.611), "LeftArm": (-2.086, 0.174, 0.149), "LeftForeArm": (-1.694, -0.229, -0.65), "Head": (-1.503, -0.004, -0.006), "Spine2": (-2.808, 0.046, 0.005)},
    "THANK_YOU": {"RightArm": (-0.4, 0, -0.3), "RightForeArm": (-0.3, 0, 0.3), "Head": (0.1, 0, 0)},
    "YES": {"RightArm": (-0.2, 0, -0.2), "RightForeArm": (-0.2, 0, 0.2), "Head": (0.15, 0, 0)},
    "NO": {"Head": (0, 0.25, 0), "RightArm": (-0.2, 0, -0.2)},
    "GOOD": {"RightArm": (-0.3, 0, -0.3), "RightForeArm": (-0.25, 0, 0.3), "RightHandThumb1": (0, 0, -0.7)},
    "PLEASE": {"RightArm": (-0.4, 0, -0.2), "RightForeArm": (-0.3, 0, 0.3), "RightHand": (0.6, 0, 0)},
    "WELCOME": {"RightArm": (-0.3, 0, -0.5), "LeftArm": (-0.3, 0, 0.5), "RightForeArm": (-0.25, 0, 0.3), "LeftForeArm": (-0.25, 0, -0.3)},
    "SORRY": {"RightArm": (-0.3, 0, -0.2), "RightForeArm": (-0.25, 0, 0.25), "Head": (0.08, 0, 0)},
    "WHERE": {"RightArm": (-0.3, 0, -0.3), "RightForeArm": (-0.3, 0, 0.4), "Head": (0, 0.1, 0)},
    "WHAT": {"RightArm": (-0.2, 0, -0.3), "RightForeArm": (-0.25, 0, 0.3), "LeftArm": (-0.2, 0, 0.3)},
    "NAME": {"RightArm": (-0.2, 0, -0.25), "RightForeArm": (-0.3, 0, 0.4), "RightHandIndex1": (-0.3, 0, 0)},
    "WATER": {"RightArm": (-0.3, 0, -0.2), "RightForeArm": (-0.25, 0, 0.3), "RightHandIndex1": (-0.3, 0, 0)},
    "FOOD": {"RightArm": (-0.5, 0, -0.2), "RightForeArm": (-0.3, 0, 0.5), "RightHand": (0.4, 0, 0)},
    "HOME": {"RightArm": (-0.4, 0, -0.3), "LeftArm": (-0.4, 0, 0.3), "RightForeArm": (-0.25, 0, 0.3)},
    "MORNING": {"RightArm": (-0.4, 0, -0.4), "RightForeArm": (-0.25, 0, 0.3)},
    "GOODBYE": {"RightArm": (-0.3, 0, -0.5), "RightForeArm": (-0.2, 0, 0.3), "RightHand": (0.2, 0.3, 0)},
    "HELP": {"RightArm": (-0.3, 0, -0.4), "LeftArm": (-0.3, 0, 0.4), "RightForeArm": (-0.25, 0, 0.3), "LeftForeArm": (-0.25, 0, -0.3)},
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
