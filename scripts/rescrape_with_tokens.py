"""
Re-scrape UAE Sign Language with full signed video URLs.
The original scrape had unsigned URLs — this gets the real tokenized download URLs.
"""
import urllib.request, json, re
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "raw" / "uae_signs_full.json"

url = ("https://www.za.gov.ae/en/sxa/search/results/"
       "?s={9D475605-DBA6-4E8A-BA9A-7949E0F32CF5}"
       "&v={FF1E6744-6C43-40A7-835E-D75CE4092535}"
       "&l=en&itemid={5596362A-FE9B-4ABC-844F-921767C0AE1F}"
       "&sig=mastercard&p=1246")

req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.loads(r.read())

results = data["Results"]
print(f"Total: {len(results)} signs")

signs = []
for item in results:
    html = item.get("Html","")
    path = item.get("Path","")

    title_m = re.search(r'ex-master-card__title[^>]*>([^<]+)<', html)
    img_m   = re.search(r'src="(/-/media[^"]+\.(?:jpg|jpeg|png))"', html)
    # Get full signed URL (with ?s= token)
    vid_m   = re.search(r'href="(https://player\.vimeo\.com/external/[^"]+\.mp4[^"]*)"', html)
    # Download URL has &download=1
    dl_m    = re.search(r'href=.{1,3}(https://player\.vimeo\.com/external/[^\'"]+&amp;download=1[^\'"]*).{1,3}', html)

    title    = title_m.group(1).strip() if title_m else ""
    img_url  = ("https://www.za.gov.ae" + img_m.group(1)) if img_m else ""
    vid_url  = vid_m.group(1) if vid_m else ""
    dl_url   = dl_m.group(1).replace("&amp;","&") if dl_m else vid_url

    parts = path.split("/")
    category = parts[-2].replace("_"," ").replace("-"," ") if len(parts)>=2 else ""

    if title:
        signs.append({
            "english":   title,
            "category":  category,
            "image_url": img_url,
            "video_url": dl_url or vid_url,
            "path":      item.get("Url",""),
        })

print(f"Parsed {len(signs)} signs with URLs")

# Save
OUT.write_text(json.dumps(signs, ensure_ascii=False, indent=2))
print(f"Saved to {OUT}")

# Quick stats
with_video = sum(1 for s in signs if s["video_url"])
print(f"Signs with video: {with_video}/{len(signs)}")
print(f"Sample: {signs[0]['english']} -> {signs[0]['video_url'][:70]}")
