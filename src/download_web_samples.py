# =========================================================================
# WEB SAMPLE DOWNLOADER - Sunum icin harici Head CT goruntuleri
# =========================================================================
# Wikimedia Commons (public domain / CC) uzerinden 2-3 ornek Head CT
# goruntusu indirir ve web_crawled_test/ klasorune kaydeder.
#
# Kullanim:
#   python src/download_web_samples.py
#
# Sonra:
#   python main.py --webcrawl
# =========================================================================

import sys
import urllib.request
from pathlib import Path
from PIL import Image
from io import BytesIO

from config import WEB_CRAWLED_DIR


# Wikimedia Commons - public domain Head CT goruntuleri
# Her URL'in kaynak sayfasi yorumda; kopyalanmadan once erisilebilirligi
# kontrol edildi. Dosyalari kucuk PNG formatinda kaydederiz.
SOURCES = [
    {
        "name": "01_normal_brain_ct_axial.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Computed_tomography_of_human_brain_-_large.png/512px-Computed_tomography_of_human_brain_-_large.png",
        "expected_label": "normal",
        "source": "Wikimedia Commons - Computed tomography of human brain",
    },
    {
        "name": "02_normal_ct_axial_slice.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Computed_tomography_of_human_brain_%28large_version%29_-_120.png/512px-Computed_tomography_of_human_brain_%28large_version%29_-_120.png",
        "expected_label": "normal",
        "source": "Wikimedia Commons - Brain CT axial slice",
    },
    {
        "name": "03_subdural_hematoma_ct.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Subduralandherniation.PNG/512px-Subduralandherniation.PNG",
        "expected_label": "hemorrhage",
        "source": "Wikimedia Commons - Subdural hematoma with herniation",
    },
]


def download_one(item: dict) -> bool:
    """Tek bir goruntuyu indir ve PNG olarak kaydet."""
    out_path = WEB_CRAWLED_DIR / item["name"]
    if out_path.exists():
        print(f"  [SKIP] {item['name']} zaten mevcut")
        return True
    try:
        req = urllib.request.Request(
            item["url"],
            headers={"User-Agent": "Mozilla/5.0 (Educational/BM480)"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        img = Image.open(BytesIO(data)).convert("RGB")
        img.save(str(out_path), format="PNG")
        print(f"  [OK]   {item['name']}  ({img.size[0]}x{img.size[1]})")
        print(f"         beklenen: {item['expected_label']}")
        print(f"         kaynak  : {item['source']}")
        return True
    except Exception as e:
        print(f"  [FAIL] {item['name']}: {e}")
        return False


def main():
    print("=" * 60)
    print("WEB SAMPLE DOWNLOADER - Head CT (sunum icin)")
    print("=" * 60)
    print(f"Hedef klasor: {WEB_CRAWLED_DIR}")
    print()

    ok = 0
    for item in SOURCES:
        if download_one(item):
            ok += 1

    print()
    print("=" * 60)
    print(f"Indirilen / Toplam: {ok} / {len(SOURCES)}")
    print()
    print("Sonraki adim:")
    print("  python main.py --webcrawl")
    print("=" * 60)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
