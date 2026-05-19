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
from arab_renderer import get_or_render_avatar, AVATAR_DIR, render_avatar_video
from pathlib import Path as _Path

def _extract_and_render(sign: str):
    """Extract mocap from skeleton video on demand, then render Arab avatar."""
    import cv2, json
    import mediapipe as mp
    skel = VIDEOS_DIR / f"{sign.upper()}.mp4"
    mocap_out = _Path(__file__).parent.parent / 'data' / 'processed' / 'mocap' / f"{sign.upper()}.json"
    if mocap_out.exists(): # already extracted
        render_avatar_video(sign)
        return
    # Quick extraction with lite model
    cap = cv2.VideoCapture(str(skel))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    mp_hol = mp.solutions.holistic
    with mp_hol.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4,
                         model_complexity=0, static_image_mode=False) as hol:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            small = cv2.resize(frame, (320, 180))
            res = hol.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            fd = {}
            if res.pose_landmarks:
                fd['pose'] = [[lm.x,lm.y,lm.z,lm.visibility] for lm in res.pose_landmarks.landmark]
            if res.right_hand_landmarks:
                fd['rhand'] = [[lm.x,lm.y,lm.z] for lm in res.right_hand_landmarks.landmark]
            if res.left_hand_landmarks:
                fd['lhand'] = [[lm.x,lm.y,lm.z] for lm in res.left_hand_landmarks.landmark]
            frames.append(fd)
    cap.release()
    mocap_out.parent.mkdir(exist_ok=True)
    with open(mocap_out, 'w') as f:
        json.dump({'fps': fps, 'frames': frames}, f, separators=(',', ':'))
    render_avatar_video(sign)
    print(f"[OnDemand] {sign} → {(AVATAR_DIR/f'{sign.upper()}.mp4').stat().st_size//1024 if (AVATAR_DIR/f'{sign.upper()}.mp4').exists() else 0}KB")

BASE  = Path(__file__).parent.parent
SIGNS = {p.stem.upper() for p in VIDEOS_DIR.glob("*.mp4")}
AVATAR_SIGNS = {p.stem.upper() for p in AVATAR_DIR.glob("*.mp4")}
print(f"[ESL] {len(SIGNS)} skeleton | {len(AVATAR_SIGNS)} avatar videos ready")

# ── OpenAI gloss ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """You are an ESL (Emirati Sign Language) interpreter.
Convert input text to a list of sign tokens.

Step 1: For each word/concept, check if a matching sign exists in the known sign library.
Step 2: If a match exists, output the sign name (uppercase, underscores for spaces).
Step 3: If NO match exists, translate the word to Arabic and output the Arabic word in Arabic script.
        The system will finger-spell it letter by letter.

Known sign library (598 signs):
ABU_DHABI_MEDIA_COMPANY, AEEN, AIN_KHAT, AIR_CONDITIONER, AIR_HOST, AL, ALARM_CLOCK, ALFIGATE, ALGAE, ALIF, AL_MUMZAR, AL_QASBAH, AMBASSADOR, AMMA, ANGRY, ANNOYED, APARTMENT, ARABIAN_JASMINE, ARABIC_DRAWING_ROOM, ARCHERY, ARRANGING, ASEEDA, ASSISTANT, ASST_TEACHER, ASTRONAUT, ASTRONOMER, AUNT, AUTHORITY, AWAFI, BAA, BAJEELA, BAKER, BARBECUE, BARBER, BASIL, BASKETBALL, BATHROOM, BATHROOM_SHOWER, BED, BEDROOM, BED_SHEET, BELALAYT, BELL, BETHHETHA, BILLIARD, BIRTH_CERTIFICATE, BIRYANI, BLAME, BOHERA_KHALID, BORROW, BOWLING, BOXING, BREAK, BRIGADIER, BROCCOLI, BROTHER, BROTHERS_WIFE, BROTHER_IN_LAW, BUILD, BURJ_AL_ARAB, BURJ_KHALIFA, BURNS, BUTCHER, CALL, CALLING_DUAA, CAMEL_RACING, CAMERAMAN, CAPTAIN, CAR, CARDAMOM, CARPENTER, CARPET, CARRY, CAR_OWNERSHIP_CARD, CAR_RACING, CASHIER, CELEBRATE, CENTURY, CERTIFICATE_OF_GOOD_CONDUCT, CERTIFICATE_OF_STUDY, CHAIR, CHANDELIER, CHARACTERIZED, CHEF, CHESS, CHOOSE, CID, CINNAMON, CITIZENSHIP_PAPER, CITY_NUMBER_IN_CITIZENSHIP, CIVILIAN, CLEARANCE, CLERK, CLOSE, CLOTH_IRONER, CLOVES, CLUB, COFFEE_CUP, COLLEGE_DEGREE, COLONEL, COMBING, COMMITTEE, COMMON_VERBS, COMPETITION, COMPLEMENT, CONCENTRATE, CONFIRM, CONTEMPLATION, COOKING, COPY, CORN, COUPLE, COURT, CROWN_PRINCE, CRY, CURTAIN, CUSTOMS, CYCLING, DAAD, DAAL, DABS_AL_TAMAR, DANCING, DAUGHTER, DAY, DAYTIME, DEATH_CERTIFICATE, DECADE, DECEIVE, DECIDE, DEFENSE_MINISTER, DELAYED, DEMOLISH, DESIGNER, DESPISE, DETERIORATING, DEWAN, DIG, DINNING_ROOM, DISABLED_CARD, DISAPPEARS, DISCHARGED, DISOBEY, DIVE, DIVORCE_CERTIFICATE, DOCTOR, DOOR, DRAWS, DREAMING, DRINK, DRIVER, DRIVING, DRIVING_LICENSE, DUE, DVD_PLAYER, EARLY, EARN, EIGHT, ELECTRICITY, ELECTRIC_KETTLE, ELECTRIC_WIRE, ELEVEN, EMBASSY, EMIRATES_IDENTITY_AUTHORITY, EMIRATES_ID_CARD, EMIRATES_UNIVERSITY, EMPLOYEE, END, ENGINEER, ENTER, ENTRANCE_DOOR, EVOLVE, EXAMINE, EXPLAIN, FAA, FALCONER, FALL, FAMILY, FAN, FARMER, FATHER, FEDERAL_NATIONAL_COUNCIL, FEEDS, FIND, FIREFIGHTER, FIVE, FLIRT, FLOWER, FLOWER_VASE, FLY, FOLDS, FOOTBALL, FORGIVES, FORK, FOUR, FOUR_SEASONS, FRANKINCENSE, FREEZER, FRIDAY, FULL, GARAGE, GAS_CYLINDER, GCC, GENERAL, GENERAL_CIVIL_AVIATION_AUTHORITY, GINGER, GIVE, GLASS, GOALBALL, GOLF, GRANDFATHER, GRANDMOTHER, GRASS, GREGORIAN_CALENDAR, GROWS, GUARD, GUEST_HOUSE, GUEST_ROOM, GUIDE, GUM_TREE, HAA, HADIQA_HAIWANAAT, HAIR_STYLER, HAMZA_NABIRA, HAMZA_SATR, HAMZA_WOW, HAMZA_YAA, HANDBALL, HANGER, HANNAYA, HAREES, HARVEST, HATES, HAZZA_BIN_ZAYED, HEALTH_CARD, HEARS, HEATER, HELPS, HE_GOES, HIJRI_CALANDER, HITS, HOD, HOME_LAWN, HORSE_RACING, HOUR, HOUSE, HOUSE_OWNERSHIP, HOUSING_LOAN, HOW_ARE_YOU, HUNTING, IGNORES, IMAM, INFORMER, INITIATIVE, INSPECTOR, INSURANCE_AGAINST_OTHERS, INSURANCE_CARD, INTERPRETER, INTERVIEWER, INVITES_INVITATION, IRON, ISLAMIC_AFFAIRS_AND_ENDOWMENTS, ITIKAAF, JABAL_HAFEET, JABAL_JAIS, JABBAB, JAMEYA, JAMI, JAMIYA_SHAIKH_ZAYD, JASHAA, JASMINE, JAZERA_NAKHLA, JBR, JEEM, JEWELER, JOKEY, JOURNALIST, JUDGED, JUMAIRAH, JUMP, KAAF, KARATE, KASIR_AMWAAJ, KEYS, KHABEESA, KHANFAROOSH, KHUBZ_KHAMEER, KHUBZ_RAQAAQ, KING, KITCHEN, KNEADS, KNIFE, LAA, LAAM, LABOR, LACES_TIE, LACTATION_BROTHER, LADDER, LAMP, LAND_OWNERSHIP, LAQEEMAT, LATE, LAUGHS, LAWYER, LEAGUE_CHAMPIONSHIP, LEAGUE_OF_ARAB_STATES, LIBRARY, LIEUTENANT, LIEUTENANT_COLONEL, LIFTS, LIGHT_BULB, LOOKING, LOSE, LOSES, LOVE, LT_GENERAL, MABKHARA, MAESTRO, MAHILLI_ZAYED, MAIDS_ROOM, MAJBOOS, MAJLIS, MAJOR, MAJOR_GENERAL, MAKEUP, MALEH, MANAGER, MARATHON, MARQOOQA, MARRIAGE_CERTIFICATE, MARRIAGE_FUND, MASJID_AL_BIDYAH, MASJID_SHAIKH_ZAYED, MATTRESS, MEDICAL_CERTIFICATE, MEEM, MEMBERSHIP_CARD, MICROWAVE, MINISTER, MINISTRY_OF_COMMUNITY_DEVELOPMENT, MINISTRY_OF_DEFENSE, MINISTRY_OF_ENDOWMENTS, MINISTRY_OF_FINANCE, MINISTRY_OF_FOREIGN_AFFAIRS, MINISTRY_OF_HEALTH, MINISTRY_OF_INTERIOR, MINISTRY_OF_JUSTICE, MINISTRY_OF_PRESIDENTIAL_AFFAIRS, MINT, MINUTE, MIRROR, MISTAKE, MONDAY, MORNING, MOSQUE_PREACHER, MOTHER, MUG, MULHAQ, MUMBAZZARA, MUNTAZAH_SHARJAH, MURDER, MUSICIAN, MUZAMMAR, NAGHAR, NAKHI, NAPHEW, NATIONAL_CENTER_OF_METEOROLOGY, NIGHT, NINE, NOON, NOTARY, OBEYS, OFFICER, OLYMPIAD, ONE, OPEN, OUT, OVEN, PAINTER, PAINTING, PALM, PARTICIPATE, PENALTY, PENALTY_CARD, PENSIONS_AND_SOCIAL_INSURANCE, PERFORMS_ABLUTION, PHOTOGRAPH, PHOTOGRAPHER, PHYSIOTHERAPIST, PILLOW, PILOT, PLANTING, PLATE, PLAYGROUND, PLAYS, PLAYS_GUITAR, PLOW, PLUMBER, POLICE_ACADEMY, POPULAR_CUISINES, PORTER, POWER_OF_ATTORNEY, PRAY, PREVENT, PROGRAMMER, PROSECUTOR, PSYCHOLOGIST, PULLS, PUNISHMENT, PUSH, PUTS, QAAF, QARIYATUL_ALAMIYA, QARS, QASHEED, QUATER_HOUR, RAA, RACING, RATION_CARD, READ, RECEIVES, RECOMMENDED, REFEREE_FLAG, REFRIGERATOR, RELAX, REMOTE, REMOVE, REPORTER, REQUESTS, ROOM, ROSE, ROWING_BOAT, RUBBER_TREE, RUBBING, RULE, RUNNING_RACE, RUSH, SAA, SAAD, SAFE, SAFFRON, SALARY_CERTIFICATE, SALOONA, SAQOO, SATURDAY, SCHOOL, SCULPTOR, SECOND, SECOND_WIFE, SECRETARY, SEEN, SELL, SENDS, SERVENT, SEVEN, SEW, SHAHADA_LA_YAMLAK, SHEEN, SHEIKH, SHEPHERD, SHOOTING, SHOT_PUT, SHOUTS, SIBLING, SIGN, SIGN_LANGUAGE, SILENCE, SING, SISTER, SIX, SKATING, SLEEP, SMELLS, SMITH, SMOKES, SOLDIER, SON, SOUQ_AL_MARKAZI_SHARJAH, SOUQ_JUMMAH, SPEAKS, SPOON, SPORTS, SPORTS_UNION, SPRING, STABLEMAN, STANDING, STEALS, STORAGE, SUGAR_CANE, SUITCASE, SUMMER, SUNDAY, SUNFLOWER, SUNRISE, SUNSET, SUREED, SURFACE, SURGEON, SWIMMING, SWIMMING_POOL, SWIMS, SWITCH, TAA, TABLE, TABLE_TENNIS, TAPE_RECORDER, TARGET, TASTES, TEACHER, TEACUP, TEAPOT, TELEPHONE, TELEPHONE_OPERATOR, TEN, TENNIS, TENT, TENT_ROOM, THE_MINISTRY_OF_EDUCATION, THE_MINISTRY_OF_HIGHER_EDUCATION, THE_MINISTRY_OF_PUBLIC_WORKS, THINK, THORN, THREE, THROW_IN, THURSDAY, TIME, TOMMORW, TRAVELS, TRAY, TROPHY, TRYING, TUA, TUESDAY, TV, TWELVE, TWINS, TWO, TYPIST, UAE_RED_CRESCENT_AUTHORITY, UNCLE, UNIVERSITY_CITY_SHARJAH, URSIYA, VASE, VILLA, VISITS, VOLLYBALL, VOLUNTEER, WAITER, WAKEEL, WAKES_UP, WALK, WALL, WALL_CLOCK, WANTS, WAR, WARDROBE, WARNING, WASHBASIN, WASHING, WASHING_MACHINE, WATCH, WATCHMAN, WATERING, WATER_AND_ELECTRICITY_AUTHORITY, WATER_HEATER, WATER_POLO, WAVE, WEARING, WEDNESDAY, WEEK, WEIGHS, WEIGHT_LIFTING, WELDER, WHEAT, WHISTLE, WIFE, WINDOW, WINS, WINTER, WIPE, WORK, WOW, WRITE, YAA, YANZAA, YATAYAMAM, YEAR, YEQAT, YESTERDAY, YOUTH_AND_SPORTS_WELFARE, ZAAI, ZAAL, ZAKAT_FUND, ZAYED_HIGHER_INSTITUTE, ZAYED_HOUSING_PROGRAM, ZAYED_MILITARY_COLLEGE, ZAYED_UNIVERSITY, ZERO, ZIZIPHUS

Rules:
- Output ONLY tokens separated by spaces. Max 10 tokens.
- Match concepts broadly: "airport" → AIR_HOST if no AIRPORT sign, "closed" → check CLOSE/CLOSING
- For proper nouns with no match (Abu Dhabi, London) → output in Arabic: أبوظبي
- For unknown concepts → translate to Arabic word

Examples:
Input: good morning doctor
Output: MORNING DOCTOR

Input: airport closed due to war
Output: AIR_CONDITIONER مغلق WAR

Input: I need help at school
Output: HELPS SCHOOL

Output ONLY the tokens."""


def get_gloss(text: str) -> list[str]:
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                max_tokens=80, temperature=0.1,
            )
            raw = r.choices[0].message.content.strip()
            # Split preserving Arabic tokens
            tokens = [t.strip('.,!?؟،') for t in raw.split() if t.strip('.,!?؟،')]
            return tokens[:8] or ["HOW_ARE_YOU"]
        except Exception as e:
            print(f"[OpenAI] Error: {e}")

    # Rule-based fallback — map common words to Arabic for finger-spelling
    ar_map = {
        "مرحبا": "مرحبا", "كيف": "كيف", "دكتور": "DOCTOR", "طبيب": "DOCTOR",
        "عائلة": "FAMILY", "مدرسة": "SCHOOL", "عمل": "WORK",
    }
    en_map = {
        "hello": "مرحبا", "hi": "مرحبا", "doctor": "DOCTOR",
        "family": "FAMILY", "school": "SCHOOL", "work": "WORK",
        "morning": "MORNING", "sleep": "SLEEP", "help": "مساعدة",
        "airport": "مطار", "war": "حرب", "closed": "مغلق",
        "hospital": "مستشفى", "police": "شرطة", "fire": "نار",
    }
    tokens = []
    for w in text.split():
        wc = w.strip(".,!?؟،").lower()
        if wc in ar_map: tokens.append(ar_map[wc])
        elif wc in en_map: tokens.append(en_map[wc])
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

        elif p.startswith("/api/v1/avatar-video/"):
            raw_name = p.split("/")[-1].replace(".mp4","").replace(".MP4","")
            # Check stitched avatar cache first (hash is lowercase)
            stitched = AVATAR_DIR / "stitched" / f"{raw_name}.mp4"
            if stitched.exists() and stitched.stat().st_size > 5000:
                self.serve_file(stitched, "video/mp4"); return
            # Single sign avatar (sign names are uppercase)
            vid_path = get_or_render_avatar(raw_name.upper())
            if vid_path:
                self.serve_file(Path(vid_path), "video/mp4")
            else:
                self.send_json({"error": f"No avatar video for {raw_name}"}, 404)

        elif p.startswith("/api/v1/video/"):
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

            stitched = get_stitched_video(tokens)
            if stitched:
                vid_name = Path(stitched).name
                video_url = f"/api/v1/video/{vid_name}"
            else:
                video_url = None

            # Stitch avatar video — resolve each token to an avatar clip
            import hashlib
            from video_stitcher import stitch_videos, ARABIC_TO_ENGLISH, find_best_sign
            key = hashlib.md5("_".join(tokens).encode()).hexdigest()[:10]
            avatar_stitch_dir = AVATAR_DIR / "stitched"
            avatar_stitch_dir.mkdir(exist_ok=True)
            avatar_stitched_path = avatar_stitch_dir / f"{key}.mp4"
            if not (avatar_stitched_path.exists() and avatar_stitched_path.stat().st_size > 5000):
                avatar_clips = []
                for t in tokens:
                    # Resolve token to an avatar video path
                    # 1. Try direct English sign name
                    ap = AVATAR_DIR / f"{t.upper()}.mp4"
                    if not (ap.exists() and ap.stat().st_size > 5000):
                        # 2. Arabic token → translate to English → look up avatar
                        is_ar = any('\u0600' <= c <= '\u06ff' for c in t)
                        if is_ar:
                            eng = ARABIC_TO_ENGLISH.get(t) or ARABIC_TO_ENGLISH.get(t.strip('\u0627\u0644'))
                            if eng:
                                ap = AVATAR_DIR / f"{eng.upper()}.mp4"
                        # 3. Similarity search
                        if not (ap.exists() and ap.stat().st_size > 5000):
                            best = find_best_sign(t.upper(), threshold=0.85)
                            if best:
                                ap = AVATAR_DIR / f"{best}.mp4"
                        # 4. Letter-by-letter spelling using avatar alphabet signs
                        if not (ap.exists() and ap.stat().st_size > 5000):
                            from video_stitcher import ARABIC_CHAR_MAP, ENGLISH_LETTER_MAP
                            # Determine what to spell: Arabic word or English word
                            spell_word = t  # the original token
                            if not is_ar and eng:
                                spell_word = eng  # spell the English translation
                            letter_clips = []
                            if any('\u0600' <= c <= '\u06ff' for c in spell_word):
                                # Spell Arabic chars
                                for ch in spell_word:
                                    if ch in ARABIC_CHAR_MAP:
                                        lp2 = AVATAR_DIR / f"{ARABIC_CHAR_MAP[ch]}.mp4"
                                        if lp2.exists(): letter_clips.append(str(lp2))
                            else:
                                # Spell English chars via Arabic alphabet avatars
                                for ch in spell_word.upper():
                                    if ch in ENGLISH_LETTER_MAP:
                                        lp2 = AVATAR_DIR / f"{ENGLISH_LETTER_MAP[ch]}.mp4"
                                        if lp2.exists(): letter_clips.append(str(lp2))
                            avatar_clips.extend(letter_clips)
                            continue  # skip the single-clip append below
                    if ap.exists() and ap.stat().st_size > 5000:
                        avatar_clips.append(str(ap))
                if avatar_clips:
                    stitch_videos(avatar_clips, str(avatar_stitched_path))
            avatar_url = f"/api/v1/avatar-video/{key}" if (avatar_stitched_path.exists() and avatar_stitched_path.stat().st_size > 5000) else None

            self.send_json({
                "request_id": str(uuid.uuid4())[:8],
                "input_text": text,
                "gloss_tokens": tokens,
                "gloss_string": " ".join(tokens),
                "video_url": video_url,
                "avatar_video_url": avatar_url,
                "skeleton_videos": [video_url] if video_url else [],
                "status": "completed",
            })

        else:
            self.send_json({"error": "Not found"}, 404)


if __name__ == "__main__":
    port = 8001
    print(f"[ESL] Lean server :{port} | signs={len(SIGNS)} | openai={'yes' if OPENAI_API_KEY else 'no'}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
