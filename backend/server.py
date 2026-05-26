"""
ESL Platform — Server
=====================
Text → Emirati Sign Language → 3D avatar pipeline.

Endpoints:
  GET  /health                       quick liveness probe + counts
  POST /api/v1/gloss                 text → list of gloss tokens
  POST /api/v1/translate             text → tokens + avatar URLs
  GET  /api/v1/avatar-glb/{TOKEN}    merged GLB (avatar + animation) for live Three.js playback
  GET  /api/v1/avatar-glb/{TOKEN}?list=1   JSON: resolved sign sequence for a token
  GET  /api/v1/avatar-video/{KEY}    pre-rendered avatar MP4 (lazy stitched + cached)

Single-process, single-file. Subprocess-based 3D pipeline lives in
`avatar_3d_renderer.py` (extract → retarget → merge → render).

No skeleton/stick-figure dependencies. ThreadingHTTPServer so slow
renders don't block other requests.
"""
import json, os, uuid, hashlib
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

from avatar_3d_renderer import (
    get_or_render_avatar_3d, stitch_avatar_videos,
    RENDER_DIR as AVATAR_DIR,
    STITCHED_DIR as AVATAR_STITCHED_DIR,
    MOTION_DB_DIR,
    AVATAR_GLB_DIR,
)
from video_stitcher import get_stitched_video, STITCHED_DIR, VIDEOS_DIR

BASE = Path(__file__).parent.parent
AVATAR_SIGNS = {p.stem.upper() for p in MOTION_DB_DIR.glob("*.mp4")}
SKELETON_SIGNS = {p.stem.upper() for p in VIDEOS_DIR.glob("*.mp4")}
print(f"[ESL] {len(SKELETON_SIGNS)} skeleton | {len(AVATAR_SIGNS)} 3D avatar source videos available")


# ── Letter maps (inlined; previously in video_stitcher.py) ──────────────────
ENGLISH_LETTER_MAP = {
    'A': 'ALIF', 'B': 'BAA',  'C': 'SEEN', 'D': 'DAAL', 'E': 'AEEN',
    'F': 'FAA',  'G': 'JEEM', 'H': 'HAA',  'I': 'ALIF', 'J': 'JEEM',
    'K': 'KAAF', 'L': 'LAAM', 'M': 'MEEM', 'N': 'NOON', 'O': 'AEEN',
    'P': 'FAA',  'Q': 'QAAF', 'R': 'RAA',  'S': 'SEEN', 'T': 'TAA',
    'U': 'AEEN', 'V': 'FAA',  'W': 'WOW',  'X': 'SEEN', 'Y': 'YAA',
    'Z': 'ZAAI',
}

ARABIC_CHAR_MAP = {
    'ا': 'ALIF', 'أ': 'ALIF', 'إ': 'ALIF', 'آ': 'ALIF',
    'ب': 'BAA',
    'ت': 'TAA',
    'ث': 'TUA',
    'ج': 'JEEM',
    'ح': 'HAA',
    'خ': 'HAA',
    'د': 'DAAL',
    'ذ': 'ZAAL',
    'ر': 'RAA',
    'ز': 'ZAAI',
    'س': 'SEEN',
    'ش': 'SHEEN',
    'ص': 'SAAD',
    'ض': 'DAAD',
    'ط': 'TAA',
    'ظ': 'TUA',
    'ع': 'AEEN',
    'غ': 'AEEN',
    'ف': 'FAA',
    'ق': 'QAAF',
    'ك': 'KAAF',
    'ل': 'LAAM',
    'م': 'MEEM',
    'ن': 'NOON',
    'ه': 'HAA',
    'و': 'WOW',
    'ي': 'YAA',
    'ى': 'YAA',
    'ة': 'TAA',
    'ئ': 'YAA',
    'ؤ': 'WOW',
}

ARABIC_TO_ENGLISH = {
    # Words → known English sign names (only ones we have direct signs for)
    "دكتور": "DOCTOR", "طبيب": "DOCTOR",
    "عائلة": "FAMILY", "أسرة": "FAMILY",
    "مدرسة": "SCHOOL",
    "عمل": "WORK", "وظيفة": "WORK",
}


# ── OpenAI gloss + en→ar translation ─────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_EN2AR_CACHE_PATH = BASE / "data" / "processed" / "en2ar_cache.json"
_EN2AR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
try:
    _EN2AR_CACHE = json.loads(_EN2AR_CACHE_PATH.read_text()) if _EN2AR_CACHE_PATH.exists() else {}
except Exception:
    _EN2AR_CACHE = {}


def translate_to_arabic(word):
    """English word → Arabic equivalent. Cached on disk. Returns None on failure."""
    word = (word or '').strip()
    if not word: return None
    key = word.lower()
    if key in _EN2AR_CACHE:
        return _EN2AR_CACHE[key]
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                    "Translate the English word to its single most common Arabic equivalent. "
                    "Output ONLY the Arabic word, no diacritics, no English, no punctuation."},
                {"role": "user", "content": word},
            ],
            max_tokens=20, temperature=0,
        )
        ar = (r.choices[0].message.content or '').strip().strip('.,!?؟،"\'')
        if ar and any('\u0600' <= c <= '\u06ff' for c in ar):
            _EN2AR_CACHE[key] = ar
            try:
                _EN2AR_CACHE_PATH.write_text(json.dumps(_EN2AR_CACHE, ensure_ascii=False, indent=2))
            except Exception:
                pass
            return ar
    except Exception as e:
        print(f"[en2ar] error for {word!r}: {e}")
    return None


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
- Match concepts broadly: "closed" → check CLOSE/CLOSING
- For proper nouns with no match (Abu Dhabi, London) → output in Arabic: أبوظبي
- For unknown concepts → translate to Arabic word

Examples:
Input: good morning doctor
Output: MORNING DOCTOR

Input: airport closed due to war
Output: مطار مغلق WAR

Input: I need help at school
Output: HELPS SCHOOL

Output ONLY the tokens."""


def get_gloss(text):
    """OpenAI-backed text → gloss tokens with rule-based fallback."""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=80, temperature=0.1,
            )
            raw = r.choices[0].message.content.strip()
            tokens = [t.strip('.,!?؟،') for t in raw.split() if t.strip('.,!?؟،')]
            return tokens[:8] or ["HOW_ARE_YOU"]
        except Exception as e:
            print(f"[OpenAI] Error: {e}")

    # Rule-based fallback: map a few common words; otherwise pass-through uppercase
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


def resolve_renderable(token):
    """Resolve a gloss token (English sign name or Arabic word) into a list of
    renderable sign names that have source videos in motion_db.

    Fallback chain:
      1. Direct sign match (e.g. SCHOOL → [SCHOOL])
      2. Arabic→English mapping for known words
      3. English word with no sign → translate to Arabic via OpenAI, then fingerspell
      4. Letter-spell directly using Arabic / English char maps
    """
    t = (token or '').strip()
    if not t: return []
    is_ar = any('\u0600' <= c <= '\u06ff' for c in t)
    up = t if is_ar else t.upper()
    # 1. Direct match
    if (MOTION_DB_DIR / f"{up}.mp4").exists():
        return [up]
    # 2. Arabic → English mapping
    if is_ar:
        eng = ARABIC_TO_ENGLISH.get(t) or ARABIC_TO_ENGLISH.get(t.strip('\u0627\u0644'))
        if eng and (MOTION_DB_DIR / f"{eng.upper()}.mp4").exists():
            return [eng.upper()]
    # 3. English token with no sign match → translate to Arabic
    if not is_ar:
        ar = translate_to_arabic(t)
        if ar:
            t = ar
            is_ar = True
    # 4. Letter-spell
    lmap = ARABIC_CHAR_MAP if is_ar else ENGLISH_LETTER_MAP
    chars = t if is_ar else t.upper()
    letters = []
    for ch in chars:
        s2 = lmap.get(ch)
        if s2 and (MOTION_DB_DIR / f"{s2}.mp4").exists():
            letters.append(s2)
    return letters


# ── HTTP ─────────────────────────────────────────────────────────────────────
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

        if p in ("/health", "/api/v1/health"):
            self.send_json({
                "status": "ok",
                "signs": len(AVATAR_SIGNS),
                "skeleton_signs": len(SKELETON_SIGNS),
                "avatar_signs": len(AVATAR_SIGNS),
                "ai": "openai" if OPENAI_API_KEY else "rules",
            })

        elif p == "/api/v1/signs":
            self.send_json({"signs": sorted(AVATAR_SIGNS)})

        elif p == "/api/v1/skeleton-signs":
            self.send_json({"signs": sorted(SKELETON_SIGNS)})

        elif p.startswith("/api/v1/skeleton-video/"):
            sign = p.split("/")[-1].upper().replace(".MP4", "")
            vid = VIDEOS_DIR / f"{sign}.mp4"
            if not vid.exists():
                # Lazy build: if we have the source video, run the skeleton extractor.
                src_mp4 = MOTION_DB_DIR / f"{sign}.mp4"
                if src_mp4.exists():
                    import subprocess
                    try:
                        subprocess.run(
                            ["python3", str(BASE / "scripts" / "build_skeletons.py"),
                             "--tokens", sign, "--workers", "1"],
                            cwd=str(BASE), capture_output=True, timeout=120,
                        )
                    except Exception as ex:
                        print(f"[skeleton lazy build] {ex}")
            if vid.exists() and vid.stat().st_size > 1000:
                self.serve_file(vid, "video/mp4")
            else:
                self.send_json({"error": f"No skeleton video for {sign}"}, 404)

        elif p.startswith("/api/v1/video/"):
            # Stitched skeleton sentence video (lazy-built from a token list cache)
            name = p.split("/")[-1]
            vid = STITCHED_DIR / name
            if not (vid.exists() and vid.stat().st_size > 5000):
                key = name.replace('.mp4', '')
                sk_cache = STITCHED_DIR / f"{key}.tokens.json"
                if sk_cache.exists():
                    try:
                        toks = json.loads(sk_cache.read_text())
                        result = get_stitched_video(toks)
                        if result: vid = Path(result)
                    except Exception as ex:
                        print(f"[skeleton stitch] {ex}")
            if vid.exists() and vid.stat().st_size > 5000:
                self.serve_file(vid, "video/mp4", cache=3600)
            else:
                self.send_json({"error": "Video not found"}, 404)

        elif p.startswith("/api/v1/avatar-glb/"):
            raw_name = unquote(p.split("/")[-1]).replace(".glb", "").replace(".GLB", "")
            qs = parse_qs(urlparse(self.path).query)

            # ?list=1: return the resolved sign sequence for this token
            if qs.get('list'):
                signs = resolve_renderable(raw_name)
                self.send_json({"token": raw_name, "signs": signs})
                return

            # Otherwise serve the GLB for the FIRST resolved sign
            signs = resolve_renderable(raw_name)
            if not signs:
                self.send_json({"error": f"No avatar GLB for {raw_name}"}, 404)
                return
            target = signs[0]
            glb = AVATAR_GLB_DIR / f"arab_sheik_{target}.glb"
            if not (glb.exists() and glb.stat().st_size > 5000):
                # Lazy render the MP4; side-effect: builds the merged GLB
                get_or_render_avatar_3d(target)
            if glb.exists() and glb.stat().st_size > 5000:
                self.serve_file(glb, "model/gltf-binary", cache=3600)
            else:
                self.send_json({"error": f"GLB build failed for {raw_name} → {target}"}, 500)

        elif p.startswith("/api/v1/avatar-video/"):
            raw_name = p.split("/")[-1].replace(".mp4", "").replace(".MP4", "")
            stitched = AVATAR_STITCHED_DIR / f"{raw_name}.mp4"

            # Lazy render: build from token cache if needed
            if not (stitched.exists() and stitched.stat().st_size > 5000):
                token_cache = AVATAR_STITCHED_DIR / f"{raw_name}.tokens.json"
                if token_cache.exists():
                    try:
                        toks = json.loads(token_cache.read_text())
                        renderable = []
                        for t in toks[:6]:
                            renderable.extend(resolve_renderable(t))
                        if renderable:
                            stitch_avatar_videos(renderable, stitched)
                    except Exception as ex:
                        print(f"[avatar-video stitch] {ex}")
                else:
                    # Single sign
                    vid_path = get_or_render_avatar_3d(raw_name.upper())
                    if vid_path:
                        import shutil
                        shutil.copyfile(vid_path, stitched)

            if stitched.exists() and stitched.stat().st_size > 5000:
                self.serve_file(stitched, "video/mp4", cache=0)
            else:
                self.send_json({"error": f"No avatar video for {raw_name}"}, 404)

        else:
            # Static frontend fallback (SPA): serve files from frontend/dist,
            # falling back to index.html for client-side routes.
            self._serve_static(p)

    def _serve_static(self, path: str):
        dist = BASE / 'frontend' / 'dist'
        if not dist.exists():
            self.send_json({"error": "Not found"}, 404)
            return
        # Strip leading slash; default to index.html
        rel = path.lstrip('/') or 'index.html'
        candidate = (dist / rel).resolve()
        # Prevent escaping dist/
        try:
            candidate.relative_to(dist.resolve())
        except ValueError:
            self.send_json({"error": "Forbidden"}, 403)
            return
        if candidate.is_dir():
            candidate = candidate / 'index.html'
        if not candidate.exists():
            # SPA fallback
            candidate = dist / 'index.html'
            if not candidate.exists():
                self.send_json({"error": "Not found"}, 404)
                return
        ext = candidate.suffix.lower()
        mime = {
            '.html': 'text/html; charset=utf-8',
            '.js':   'application/javascript; charset=utf-8',
            '.mjs':  'application/javascript; charset=utf-8',
            '.css':  'text/css; charset=utf-8',
            '.json': 'application/json; charset=utf-8',
            '.svg':  'image/svg+xml',
            '.png':  'image/png',
            '.jpg':  'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif':  'image/gif',
            '.webp': 'image/webp',
            '.ico':  'image/x-icon',
            '.woff': 'font/woff',
            '.woff2':'font/woff2',
            '.ttf':  'font/ttf',
            '.map':  'application/json',
            '.mp4':  'video/mp4',
            '.glb':  'model/gltf-binary',
        }.get(ext, 'application/octet-stream')
        # No cache for HTML (so updates show up), long cache for hashed assets.
        cache = 0 if ext == '.html' else 86400
        self.serve_file(candidate, mime, cache=cache)

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

            # Same key drives both stitched outputs (skeleton + avatar).
            key = hashlib.md5("_".join(tokens).encode()).hexdigest()[:10]

            # Skeleton sentence: cache the token list so /api/v1/video/{KEY} can build lazily.
            sk_cache = STITCHED_DIR / f"{key}.tokens.json"
            sk_cache.parent.mkdir(parents=True, exist_ok=True)
            if not sk_cache.exists():
                sk_cache.write_text(json.dumps(tokens, ensure_ascii=False))
            skeleton_video_url = f"/api/v1/video/{key}.mp4"

            # 3D avatar sentence: same idea, served by /api/v1/avatar-video/{KEY}.
            av_cache = AVATAR_STITCHED_DIR / f"{key}.tokens.json"
            av_cache.parent.mkdir(parents=True, exist_ok=True)
            if not av_cache.exists():
                av_cache.write_text(json.dumps(tokens, ensure_ascii=False))
            avatar_video_url = f"/api/v1/avatar-video/{key}"

            self.send_json({
                "request_id": str(uuid.uuid4())[:8],
                "input_text": text,
                "gloss_tokens": tokens,
                "gloss_string": " ".join(tokens),
                "video_url": skeleton_video_url,
                "skeleton_videos": [skeleton_video_url],
                "avatar_video_url": avatar_video_url,
                "status": "completed",
            })

        else:
            self.send_json({"error": "Not found"}, 404)


if __name__ == "__main__":
    port = int(os.environ.get("ESL_PORT", 8001))
    print(f"[ESL] Server :{port} | skeleton={len(SKELETON_SIGNS)} | avatar={len(AVATAR_SIGNS)} | openai={'yes' if OPENAI_API_KEY else 'no'}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
