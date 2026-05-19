"""
Stitches skeleton video clips together into one MP4.
Uses ffmpeg concat — no Python video libraries needed.
"""
import subprocess, os, tempfile, hashlib
from pathlib import Path
from difflib import SequenceMatcher

# Cache available signs list
_SIGNS_CACHE: list[str] | None = None

def get_available_signs() -> list[str]:
    global _SIGNS_CACHE
    if _SIGNS_CACHE is None:
        _SIGNS_CACHE = [p.stem.upper() for p in VIDEOS_DIR.glob('*.mp4')
                        if p.stem.isascii() and p.stem.replace('_','').isalpha()]
    return _SIGNS_CACHE

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.upper(), b.upper()).ratio()

def find_best_sign(word: str, threshold: float = 0.85) -> str | None:
    """Find best matching sign name with similarity >= threshold."""
    signs = get_available_signs()
    word_up = word.upper()
    best_score = 0.0
    best_sign = None
    for sign in signs:
        s = similarity(word_up, sign)
        if s > best_score:
            best_score = s
            best_sign = sign
    if best_score >= threshold:
        print(f"[Stitch] '{word}' ~~ {best_sign} ({best_score:.0%})")
        return best_sign
    return None

VIDEOS_DIR = Path(__file__).parent.parent / "data" / "skeleton_videos"
STITCHED_DIR = Path(__file__).parent.parent / "data" / "stitched_videos"
STITCHED_DIR.mkdir(exist_ok=True)

# English letter → Arabic sign approximation (for finger-spelling unknown words)
ENGLISH_LETTER_MAP = {
    'A': 'ALIF', 'B': 'BAA',  'C': 'SEEN', 'D': 'DAAL', 'E': 'AEEN',
    'F': 'FAA',  'G': 'JEEM', 'H': 'HAA',  'I': 'ALIF', 'J': 'JEEM',
    'K': 'KAAF', 'L': 'LAA',  'M': 'MEEM', 'N': 'NOON', 'O': 'AEEN',
    'P': 'FAA',  'Q': 'QAAF', 'R': 'RAA',  'S': 'SEEN', 'T': 'TAA',
    'U': 'AEEN', 'V': 'FAA',  'W': 'WOW',  'X': 'SEEN', 'Y': 'YAA',
    'Z': 'ZAAI',
}

# Arabic letter → sign name mapping
ARABIC_CHAR_MAP = {
    'ا': 'ALIF', 'أ': 'ALIF', 'إ': 'ALIF', 'آ': 'ALIF',
    'ب': 'BAA',
    'ت': 'TAA',
    'ث': 'TUA',
    'ج': 'JEEM',
    'ح': 'HAA',
    'خ': 'HAA',   # fallback to HAA
    'د': 'DAAL',
    'ذ': 'ZAAL',
    'ر': 'RAA',
    'ز': 'ZAAI',
    'س': 'SEEN',
    'ش': 'SHEEN',
    'ص': 'SAAD',
    'ض': 'DAAD',
    'ط': 'TAA',   # fallback
    'ظ': 'TUA',   # fallback
    'ع': 'AEEN',
    'غ': 'AEEN',  # fallback
    'ف': 'FAA',
    'ق': 'QAAF',
    'ك': 'KAAF',
    'ل': 'LAA',
    'م': 'MEEM',
    'ن': 'NOON',
    'ه': 'HAA',
    'و': 'WOW',
    'ي': 'YAA', 'ى': 'YAA',
    'ة': 'TAA',
    'لا': 'LAAM',
}

# Arabic word → English translation for sign lookup
ARABIC_TO_ENGLISH = {
    'مطار': 'AIRPORT', 'حرب': 'WAR', 'مغلق': 'CLOSED', 'مفتوح': 'OPEN',
    'دكتور': 'DOCTOR', 'طبيب': 'DOCTOR', 'مستشفى': 'HOSPITAL',
    'مدرسة': 'SCHOOL', 'عمل': 'WORK', 'عائلة': 'FAMILY',
    'صباح': 'MORNING', 'نوم': 'SLEEP', 'مساعدة': 'HELPS',
    'ماء': 'WATER', 'طعام': 'FOOD', 'بيت': 'HOME', 'منزل': 'HOME',
    'شرطة': 'POLICE', 'نار': 'FIRE', 'طوارئ': 'EMERGENCY',
    'مطعم': 'RESTAURANT', 'فندق': 'HOTEL', 'بنك': 'BANK',
    'سيارة': 'CAR', 'طريق': 'ROAD', 'جسر': 'BRIDGE',
    'مرحبا': 'HOW_ARE_YOU', 'شكرا': 'THANK_YOU', 'نعم': 'YES', 'لا': 'NO',
    'صديق': 'FRIEND', 'أخ': 'BROTHER', 'أخت': 'SISTER', 'أم': 'MOTHER', 'أب': 'FATHER',
    'كبير': 'BIG', 'صغير': 'SMALL', 'جديد': 'NEW', 'قديم': 'OLD',
    'سعيد': 'HAPPY', 'حزين': 'SAD', 'غاضب': 'ANGRY',
    'الخير': 'GOOD', 'جيد': 'GOOD', 'سيء': 'BAD',
    'أبوظبي': 'ABU_DHABI', 'دبي': 'DUBAI', 'الإمارات': 'UAE',
    'بسبب': 'BECAUSE', 'في': 'IN', 'على': 'ON', 'من': 'FROM',
    'أحتاج': 'NEED', 'أريد': 'WANT', 'أعرف': 'KNOW',
}

def word_to_clips(word: str, available: set) -> list[str]:
    """Return list of video paths for a word.
    For Arabic: translate to English first, try exact match, then finger-spell.
    For English: try exact match, then synonyms.
    """
    word_up = word.upper()
    is_arabic = any('\u0600' <= c <= '\u06ff' for c in word)

    # --- Arabic word ---
    if is_arabic:
        # 1. Translate to English and try exact/similarity match
        english = ARABIC_TO_ENGLISH.get(word) or ARABIC_TO_ENGLISH.get(word.strip('\u0627\u0644'))
        if english:
            v = VIDEOS_DIR / f"{english}.mp4"
            if v.exists():
                print(f"[Stitch] '{word}' → {english} (translated exact)")
                return [str(v)]
            # Similarity search on translated English word
            best = find_best_sign(english, threshold=0.85)
            if best:
                v = VIDEOS_DIR / f"{best}.mp4"
                if v.exists(): return [str(v)]
        # 2. Similarity search on the Arabic word itself (romanized)
        best = find_best_sign(word_up, threshold=0.85)
        if best:
            v = VIDEOS_DIR / f"{best}.mp4"
            if v.exists(): return [str(v)]
        # 3. Finger-spell Arabic letter by letter
        clips = []
        for char in word:
            if char in ARABIC_CHAR_MAP:
                sign = ARABIC_CHAR_MAP[char]
                p = VIDEOS_DIR / f"{sign}.mp4"
                if p.exists():
                    clips.append(str(p))
        if clips:
            print(f"[Stitch] '{word}' → finger-spell ({len(clips)} letters)")
        return clips

    # --- English word ---
    # 1. Exact match
    vid = VIDEOS_DIR / f"{word_up}.mp4"
    if vid.exists():
        print(f"[Stitch] '{word}' → exact match")
        return [str(vid)]

    # 2. Similarity search at 85%+
    best = find_best_sign(word_up, threshold=0.85)
    if best:
        v = VIDEOS_DIR / f"{best}.mp4"
        if v.exists(): return [str(v)]

    # Try common synonyms
    synonyms = {
        'HELP': 'HELPS', 'SELL': 'SELLS', 'SEND': 'SENDS',
        'PLAY': 'PLAYS', 'PULL': 'PULLS', 'SHOUT': 'SHOUTS',
        'HOW': 'HOW_ARE_YOU', 'HELLO': 'HOW_ARE_YOU', 'HI': 'HOW_ARE_YOU',
    }
    if word_up in synonyms:
        alt = VIDEOS_DIR / f"{synonyms[word_up]}.mp4"
        if alt.exists():
            return [str(alt)]

    # Reverse synonyms
    rev = {v: k for k, v in synonyms.items()}
    if word_up in rev:
        alt = VIDEOS_DIR / f"{rev[word_up]}.mp4"
        if alt.exists():
            return [str(alt)]

    # Unknown English word → finger-spell using English→Arabic sign map
    print(f"[Stitch] Finger-spelling '{word}'")
    clips = []
    for char in word.upper():
        if char in ENGLISH_LETTER_MAP:
            sign = ENGLISH_LETTER_MAP[char]
            p = VIDEOS_DIR / f"{sign}.mp4"
            if p.exists():
                clips.append(str(p))
    return clips


def stitch_videos(clip_paths: list[str], output_path: str) -> bool:
    """Concatenate clips using ffmpeg filter_complex for smooth joins."""
    if not clip_paths:
        return False
    if len(clip_paths) == 1:
        import shutil
        shutil.copy(clip_paths[0], output_path)
        return True

    # Write concat list
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
        list_file = f.name

    try:
        result = subprocess.run([
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', list_file,
            '-c:v', 'libx264', '-crf', '20', '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=640:360',
            output_path
        ], capture_output=True, timeout=60)
        return result.returncode == 0
    finally:
        os.unlink(list_file)


def get_stitched_video(gloss_tokens: list[str]) -> str | None:
    """
    Given gloss tokens, stitch skeleton videos together.
    Returns path to stitched MP4 (cached by token hash).
    """
    global _SIGNS_CACHE
    _SIGNS_CACHE = None  # refresh on every stitch call
    available = {p.stem.upper() for p in VIDEOS_DIR.glob("*.mp4")}

    # Collect all clips
    all_clips = []
    clip_labels = []  # for debugging
    for token in gloss_tokens:
        clips = word_to_clips(token, available)
        all_clips.extend(clips)
        clip_labels.append(f"{token}→{len(clips)}clips")

    if not all_clips:
        return None

    print(f"[Stitch] {clip_labels} → {len(all_clips)} clips total")

    # Cache key from token list
    key = hashlib.md5("_".join(gloss_tokens).encode()).hexdigest()[:10]
    out = STITCHED_DIR / f"{key}.mp4"
    if out.exists():
        print(f"[Stitch] Cache hit: {out}")
        return str(out)

    ok = stitch_videos(all_clips, str(out))
    if ok:
        print(f"[Stitch] Done: {out} ({out.stat().st_size//1024}KB)")
        return str(out)
    return None


if __name__ == "__main__":
    # Test
    result = get_stitched_video(["DOCTOR", "FAMILY", "SCHOOL"])
    print("Result:", result)
