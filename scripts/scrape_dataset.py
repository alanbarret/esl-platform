"""Scrape UAE Sign Language Dictionary from za.gov.ae"""
import urllib.request, json, re, pathlib
from collections import Counter

BASE = "https://www.za.gov.ae/en/sxa/search/results/"
PARAMS = "?s={9D475605-DBA6-4E8A-BA9A-7949E0F32CF5}&v={FF1E6744-6C43-40A7-835E-D75CE4092535}&l=en&itemid={5596362A-FE9B-4ABC-844F-921767C0AE1F}&sig=mastercard"
headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

req = urllib.request.Request(BASE + PARAMS + "&p=1246", headers=headers)
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.loads(r.read())

signs = []
for item in data["Results"]:
    html = item.get("Html", "")
    path = item.get("Path", "")
    url = item.get("Url", "")

    title_match = re.search(r'ex-master-card__title[^>]*>([^<]+)<', html)
    title = title_match.group(1).strip() if title_match else ""

    img_match = re.search(r'src="([^"]+SignLanguage[^"]+)"', html)
    img_url = ("https://www.za.gov.ae" + img_match.group(1)) if img_match else ""

    video_match = re.search(r'(https?://[^"\']+\.mp4)', html)
    video_url = video_match.group(1) if video_match else ""

    parts = path.split("/")
    category = parts[-2].replace("_", " ").replace("-", " ") if len(parts) >= 2 else ""

    arabic_match = re.search(r'([\u0600-\u06FF][^\s<,]{1,}(?:\s+[\u0600-\u06FF][^\s<,]{1,})*)', html)
    arabic_name = arabic_match.group(1) if arabic_match else ""

    if title:
        signs.append({
            "id": item.get("Id", ""),
            "english": title,
            "arabic": arabic_name,
            "category": category,
            "image_url": img_url,
            "video_url": video_url,
            "path": url,
        })

print(f"Parsed {len(signs)} signs")

cats = Counter(s["category"] for s in signs)
print("\nTop categories:")
for cat, count in cats.most_common(20):
    print(f"  {cat}: {count}")

out_dir = pathlib.Path(__file__).parent.parent / "data" / "raw"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "uae_signs_raw.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(signs, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(signs)} signs to {out_path}")

# Also generate a gloss training dataset JSONL
gloss_path = out_dir.parent / "processed" / "esl_gloss.jsonl"
gloss_path.parent.mkdir(parents=True, exist_ok=True)
with open(gloss_path, "w", encoding="utf-8") as f:
    for s in signs:
        gloss = s["english"].upper().replace(" ", "_")
        if s["arabic"]:
            f.write(json.dumps({"input": s["arabic"], "output": gloss, "lang": "ar", "category": s["category"]}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"input": s["english"], "output": gloss, "lang": "en", "category": s["category"]}, ensure_ascii=False) + "\n")

print(f"Saved gloss training data to {gloss_path}")
