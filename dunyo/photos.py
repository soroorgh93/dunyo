"""Real weather photographs for the app background — openly licensed, from Wikimedia Commons, credited on screen.

For each scene (summer, autumn, rain, storm, snow, fog, haze, night…) the app picks a real photograph:
  1. a photo you put in  photos/<scene>.jpg  (your own pictures always win),
  2. otherwise one fetched once from Wikimedia Commons and cached in  photos/  with its author + licence.
     It prefers Commons "Featured pictures" and "Quality images" (reviewed, high-resolution, free licences:
     CC0 / public domain / CC BY / CC BY-SA), landscape orientation, no paintings or diagrams.
  python -m dunyo photos   downloads all scenes now (do it once, then commit the photos/ folder).
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from .config import ROOT

DIR = ROOT / "photos"
UA = {"User-Agent": "Dunyo/1.1 (climate hackathon app; https://github.com/; open-source) python-requests"}
API = "https://commons.wikimedia.org/w/api.php"
FP = 'incategory:"Featured_pictures_on_Wikimedia_Commons"'
QI = 'incategory:"Quality_images"'
SCENES: dict[str, dict[str, Any]] = {
    "summer": {"label": "Summer · sunny", "files": [], "q": ["summer meadow blue sky", "summer landscape cumulus", "sunny field summer", "summer landscape"]},
    "heat": {"label": "Heat", "files": [], "q": ["heat haze sun", "desert sun hot", "sun summer drought field", "summer sun sky"]},
    "spring": {"label": "Spring", "files": [], "q": ["cherry blossom spring", "spring blossom trees", "spring meadow flowers landscape"]},
    "autumn": {"label": "Autumn", "files": [], "q": ["autumn forest colours", "autumn foliage trees", "autumn landscape", "fall foliage"]},
    "rain": {"label": "Rain", "files": [], "q": ["rain street umbrella", "rainy street", "rain drops window", "rain clouds landscape", "rain"]},
    "storm": {"label": "Thunderstorm", "files": ["File:Lightning NOAA.jpg"], "q": ["lightning storm", "thunderstorm lightning", "supercell storm", "lightning"]},
    "snow": {"label": "Snow", "files": [], "q": ["snowfall forest winter", "snow covered trees", "winter snow landscape", "snow"]},
    "winter_clear": {"label": "Winter", "files": [], "q": ["winter landscape snow sunny", "frost winter morning", "winter landscape"]},
    "fog": {"label": "Fog", "files": ["File:A foggy winter morning.jpg"], "q": ["fog morning landscape", "foggy forest", "fog valley", "mist morning"]},
    "haze": {"label": "Haze & smog", "files": [], "q": ["smog city", "haze city skyline", "air pollution city", "dust haze"]},
    "wind": {"label": "Windy", "files": [], "q": ["wind storm waves", "storm waves sea", "windy trees", "wind"]},
    "cloudy": {"label": "Cloudy", "files": [], "q": ["overcast sky clouds landscape", "cloudy sky landscape", "clouds landscape"]},
    "night": {"label": "Clear night", "files": [], "q": ["starry sky night landscape", "milky way landscape", "city skyline night", "night sky stars"]},
}
BAD = re.compile(r"paint|drawing|illustration|diagram|map\b|chart|logo|engraving|lithograph|poster|stamp|portrait|insect|bird|museum|statue|interior|macro", re.I)
_LOCK = threading.Lock()


def key_for(season: str, weather: str, night: bool) -> str:
    if weather in ("storm",):
        return "storm"
    if weather == "rain":
        return "rain"
    if weather == "snow":
        return "snow"
    if weather == "fog":
        return "fog"
    if weather == "haze":
        return "haze"
    if night:
        return "night"
    if weather == "wind":
        return "wind"
    if weather == "heat":
        return "heat"
    if weather == "cloudy":
        return "cloudy"
    return {"spring": "spring", "summer": "summer", "autumn": "autumn", "winter": "winter_clear"}.get(season, "summer")


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html or "")).strip()


def _ok(ii: dict, title: str) -> bool:
    em = ii.get("extmetadata") or {}
    lic = (em.get("LicenseShortName") or {}).get("value", "")
    cats = (em.get("Categories") or {}).get("value", "")
    if ii.get("mime") != "image/jpeg" or not ii.get("thumburl"):
        return False
    w, h = ii.get("width", 0), ii.get("height", 1)
    if w < 1600 or w < 1.25 * h:
        return False
    if BAD.search(title) or BAD.search(cats):
        return False
    return bool(re.search(r"CC0|Public domain|CC BY|PD|Attribution", lic, re.I)) and "NC" not in lic and "ND" not in lic


def _meta(ii: dict, title: str) -> dict:
    em = ii.get("extmetadata") or {}
    return {"title": title.replace("File:", ""), "author": _strip((em.get("Artist") or {}).get("value", "")) or "Unknown",
            "license": (em.get("LicenseShortName") or {}).get("value", ""), "license_url": (em.get("LicenseUrl") or {}).get("value", ""),
            "source": ii.get("descriptionurl", ""), "url": ii.get("thumburl")}


def _search(session, query: str) -> list[tuple[str, dict]]:
    r = session.get(API, params={"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6, "gsrlimit": 15,
                                 "gsrsearch": f"{query} filetype:bitmap", "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
                                 "iiurlwidth": 1920}, headers=UA, timeout=15)
    pages = list(((r.json().get("query") or {}).get("pages") or {}).values())
    pages.sort(key=lambda p: p.get("index", 99))
    return [(p["title"], (p.get("imageinfo") or [{}])[0]) for p in pages]


def _files(session, titles: list[str]) -> list[tuple[str, dict]]:
    if not titles:
        return []
    r = session.get(API, params={"action": "query", "format": "json", "titles": "|".join(titles), "prop": "imageinfo",
                                 "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1920}, headers=UA, timeout=15)
    return [(p["title"], (p.get("imageinfo") or [{}])[0]) for p in ((r.json().get("query") or {}).get("pages") or {}).values() if p.get("imageinfo")]


def fetch(key: str) -> Path | None:
    """Find, download and cache one photo for a scene. Returns the local JPEG path or None."""
    import requests
    spec = SCENES[key]
    s = requests.Session()
    cands: list[tuple[str, dict]] = []
    try:
        cands += _files(s, spec["files"])
    except Exception:  # noqa: BLE001
        pass
    for q in spec["q"]:
        for cat in (FP, QI, ""):
            if any(_ok(ii, t) for t, ii in cands):
                break
            try:
                cands += _search(s, f"{q} {cat}".strip())
            except Exception:  # noqa: BLE001
                continue
        if any(_ok(ii, t) for t, ii in cands):
            break
    for title, ii in cands:
        if not _ok(ii, title):
            continue
        try:
            img = s.get(ii["thumburl"], headers=UA, timeout=40)
            if img.status_code == 200 and img.content[:2] == b"\xff\xd8":
                DIR.mkdir(exist_ok=True)
                p = DIR / f"{key}.jpg"
                p.write_bytes(img.content)
                (DIR / f"{key}.json").write_text(json.dumps(_meta(ii, title), indent=1, ensure_ascii=False), encoding="utf-8")
                return p
        except Exception:  # noqa: BLE001
            continue
    return None


def local(key: str) -> Path | None:
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = DIR / f"{key}.{ext}"
        if p.exists() and p.stat().st_size > 5000:
            return p
    return None


def get(key: str, online: bool) -> Path | None:
    if key not in SCENES:
        return None
    p = local(key)
    if p or not online:
        return p
    with _LOCK:
        return local(key) or fetch(key)


def credits() -> dict[str, dict]:
    out = {}
    for k, v in SCENES.items():
        meta = {"label": v["label"], "available": local(k) is not None}
        j = DIR / f"{k}.json"
        if j.exists():
            try:
                meta.update(json.loads(j.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
        elif meta["available"]:
            meta.update({"author": "your photo", "license": ""})
        out[k] = meta
    return out


def prefetch_all(verbose: bool = False) -> dict[str, bool]:
    res = {}
    for k in SCENES:
        p = get(k, online=True)
        res[k] = p is not None
        if verbose:
            c = credits()[k]
            print(f"  {'✓' if p else '✗'} {k:13s} {c.get('title', '') if p else 'not found'}  {('· ' + c.get('author', '') + ' · ' + c.get('license', '')) if p else ''}")
    return res
