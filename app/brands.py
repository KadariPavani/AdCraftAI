# Brand knowledge base and fuzzy brand matcher using difflib.

import difflib
from collections import Counter, defaultdict
from typing import Dict, List

from app.models import BrandMatch

# ---------------------------------------------------------------------------
# Brand Taglines (40+ brands)
# ---------------------------------------------------------------------------

BRAND_TAGLINES = {
    "Nike": "Just Do It",
    "Adidas": "Impossible Is Nothing",
    "Puma": "Forever Faster",
    "Reebok": "Be More Human",
    "Skechers": "Comfort That Performs",
    "Sparx": "Spark Your Style",
    "Bata": "Comfortable. Stylish. Affordable.",
    "Campus": "Walk With Confidence",
    "Wood_Land": "ProPlanet",
    "Red_Chief": "Step Out In Style",
    "FILA": "Play It Your Way",
    "HRX": "Push Your Boundaries",
    "Nivia": "Play With Passion",
    "Bahamas_footwear": "Walk In Comfort",
    "Flying_Machine": "Live The Change",
    "Raymonds": "The Complete Man",
    "Allen_Solly": "My World. My Way.",
    "Peter_England": "The Honest Shirt",
    "Louis_Philippe": "The Upper Crest",
    "Monte_Carlo": "Live It Up",
    "Biba": "Celebrate Tradition",
    "Fabindia": "Celebrate India",
    "Blackberry": "Sharp. Smart. Iconic.",
    "Tommy_Hilfiger_watches": "Classic American Cool",
    "Titan_watches": "Be More",
    "Casio_watches": "Creativity and Contribution",
    "Rolex_watches": "A Crown for Every Achievement",
    "Fastrack_watches": "Move On",
    "Sonata_watches": "Better Everyday",
    "Joyalukkas_jewellary": "The World's Favourite Jeweller",
    "Tanishq_jewellary": "A Tata Product",
    "Malabar_gold_and_diamond": "The True Value of Gold",
    "Kalyan_jewellers": "Trust Is Everything",
    "Amul": "The Taste of India",
    "Patanjali": "Prakriti Ka Ashirvaad",
    "Bisleri": "Play Safe",
    "Coca_Cola": "Open Happiness",
    "Pepsi": "The Joy of Pepsi",
    "Samsung_mobiles": "Do What You Can't",
    "Apple_mobiles": "Think Different",
    "Motorola_mobiles": "Hello Moto",
    "One-Plus_mobiles": "Never Settle",
    "maruti_suzuki": "Way of Life",
    "tata": "Connecting Aspirations",
    "mahindra": "Rise",
    "hyundai": "New Thinking. New Possibilities.",
    "honda": "The Power of Dreams",
    "toyota": "Let's Go Places",
    "tvs": "Inspired by Life",
    "hero_motocorp": "Hum Mein Hai Hero",
    "bajaj": "Distinctly Ahead",
    "ICICI_Bank": "Hum Hai Na",
    "State_Bank_Of_India": "The Banker to Every Indian",
    "LIC": "Zindagi Ke Saath Bhi, Zindagi Ke Baad Bhi",
    "Air_India": "Air India. Fly With Pride.",
    "Yatra": "India's Most Trusted Travel Brand",
    "Make_MyTrip": "Dil Toh Roaming Hai",
    "Lux_soaps": "Filmon Ka Sabun",
    "Pantene_shampoo": "Strong Is Beautiful",
    "Sunsilk_shampoo": "Life Can't Wait",
    "Head_&_Shoulders_shampoo": "You Never Get a Second Chance",
    "Lotus_Herbal": "Naturally Beautiful",
    "Chicco_baby_products": "Close to You",
}


# ---------------------------------------------------------------------------
# Brand Matcher
# ---------------------------------------------------------------------------

class BrandMatcher:
    STOP_WORDS = frozenset({
        "ad", "ads", "advertisement", "pamphlet", "poster", "banner",
        "premium", "best", "buy", "new", "latest", "good", "great",
        "the", "a", "an", "for", "with", "and", "of", "in", "on",
        "is", "are", "was", "were", "be", "been", "being",
        "i", "me", "my", "we", "our", "you", "your",
        "want", "need", "make", "create", "generate", "show",
        "person", "people", "man", "woman", "model",
        "running", "walking", "playing", "wearing", "drinking", "eating",
        "shoes", "shoe", "bottle", "car", "bike", "phone", "mobile",
        "water", "food", "clothes", "shirt", "pant", "dress",
    })

    def __init__(self, id_to_metadata: Dict[int, dict]):
        brand_data: Dict[str, Dict] = defaultdict(
            lambda: {"indices": [], "categories": Counter(), "subcategories": Counter()}
        )
        for idx, meta in id_to_metadata.items():
            brand_raw = meta.get("brand", "")
            if not isinstance(brand_raw, str):
                continue
            brand = brand_raw.strip()
            if brand and brand.lower() not in ("", "nan", "unknown"):
                brand_data[brand]["indices"].append(idx)
                brand_data[brand]["categories"][meta.get("category", "")] += 1
                brand_data[brand]["subcategories"][meta.get("subcategory", "")] += 1

        self.brand_index: Dict[str, Dict] = {}
        self.brand_names: List[str] = []
        self.brand_names_lower: List[str] = []

        for brand, data in brand_data.items():
            self.brand_names.append(brand)
            self.brand_names_lower.append(brand.lower().replace("_", ""))
            top_cat = data["categories"].most_common(1)
            top_sub = data["subcategories"].most_common(1)
            self.brand_index[brand] = {
                "indices": data["indices"],
                "category": top_cat[0][0] if top_cat else "",
                "subcategory": top_sub[0][0] if top_sub else "",
                "count": len(data["indices"]),
            }

        self.brand_names_with_spaces: List[str] = [
            b.lower().replace("_", " ") for b in self.brand_names
        ]

    def match(self, query: str) -> BrandMatch:
        words = query.lower().split()
        best_brand = None
        best_ratio = 0.0
        best_token = ""

        for word in words:
            if word in self.STOP_WORDS or len(word) < 2:
                continue
            normalized = word.replace(" ", "")
            matches = difflib.get_close_matches(
                normalized, self.brand_names_lower, n=1, cutoff=0.5
            )
            if matches:
                ratio = difflib.SequenceMatcher(None, normalized, matches[0]).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    idx = self.brand_names_lower.index(matches[0])
                    best_brand = self.brand_names[idx]
                    best_token = word

        for i in range(len(words) - 1):
            if words[i] in self.STOP_WORDS and words[i + 1] in self.STOP_WORDS:
                continue
            pair = f"{words[i]} {words[i + 1]}"
            matches = difflib.get_close_matches(
                pair, self.brand_names_with_spaces, n=1, cutoff=0.5
            )
            if matches:
                ratio = difflib.SequenceMatcher(None, pair, matches[0]).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    idx = self.brand_names_with_spaces.index(matches[0])
                    best_brand = self.brand_names[idx]
                    best_token = pair

        for i in range(len(words) - 2):
            triple = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            matches = difflib.get_close_matches(
                triple, self.brand_names_with_spaces, n=1, cutoff=0.5
            )
            if matches:
                ratio = difflib.SequenceMatcher(None, triple, matches[0]).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    idx = self.brand_names_with_spaces.index(matches[0])
                    best_brand = self.brand_names[idx]
                    best_token = triple

        if best_brand:
            info = self.brand_index[best_brand]
            return BrandMatch(
                matched_brand=best_brand,
                query_token=best_token,
                confidence=best_ratio,
                category=info["category"],
                subcategory=info["subcategory"],
                image_count=info["count"],
            )
        return BrandMatch()

    def get_brand_indices(self, brand_name: str) -> List[int]:
        if brand_name in self.brand_index:
            return self.brand_index[brand_name]["indices"]
        return []
