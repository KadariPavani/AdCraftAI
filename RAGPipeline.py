"""
MAdVerse Pamphlet Pipeline v5 - RAG-Based Ad Pamphlet Generator

Improvements over v4:
  - TRUE RAG pipeline: ALL content derived from actual dataset images
  - CLIP Descriptor RAG: ranks features/styles against retrieved ad images
  - Gemini Vision RAG: extracts text + generates prompts from dataset ads
  - NO hardcoded CATEGORY_FEATURES or CATEGORY_PROMPTS dictionaries
  - Redesigned professional template with pill-style features
  - Web logo fetching with transparent backgrounds
  - Logo overlay on generated product images

Pipeline:
  Query --> [Brand Match] --> [Filtered FAISS Retrieval]
    --> [Color Extraction] --> [Logo Extraction]
    --> [RAG Content Generation (Gemini Vision > CLIP Descriptor > fallback)]
    --> [Image Generation] --> [Logo Overlay] --> [Pamphlet Composition]
    --> Final pamphlet PNG
"""

import difflib
import faiss
import io
import json
import numpy as np
import os
import pickle
import re
import time
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sklearn.cluster import KMeans
from transformers import CLIPModel, CLIPProcessor

load_dotenv()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievedAd:
    rank: int
    similarity: float
    distance: float
    image_path: str
    image_id: str
    brand: str
    category: str
    subcategory: str
    language: str
    ad_type: str
    source: str


@dataclass
class BrandMatch:
    matched_brand: Optional[str] = None
    query_token: str = ""
    confidence: float = 0.0
    category: str = ""
    subcategory: str = ""
    image_count: int = 0


@dataclass
class PamphletContent:
    brand_name: str = ""
    tagline: str = ""
    features: List[str] = field(default_factory=list)
    accent_color: str = "#1a1a2e"
    secondary_color: str = "#0f3460"
    background_tint: str = "#f8f8ff"
    product_image: Optional[Image.Image] = None
    logo_image: Optional[Image.Image] = None
    thumbnail_images: List[Image.Image] = field(default_factory=list)
    thumbnail_paths: List[str] = field(default_factory=list)
    diffusion_prompt: str = ""
    negative_prompt: str = ""
    category: str = ""
    subcategory: str = ""


@dataclass
class PamphletResult:
    query: str = ""
    timestamp: str = ""
    brand_match: Optional[Dict[str, Any]] = None
    stage_timings: Dict[str, float] = field(default_factory=dict)
    retrieved_ads: List[Dict[str, Any]] = field(default_factory=list)
    extracted_colors: List[str] = field(default_factory=list)
    content: Optional[Dict[str, Any]] = None
    pamphlet_path: Optional[str] = None
    product_image_path: Optional[str] = None
    image_generator_used: str = ""
    gemini_used: bool = False
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Brand Knowledge Base
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
# Content Generation: RAG-based (v5)
# ---------------------------------------------------------------------------
# CATEGORY_FEATURES and CATEGORY_PROMPTS removed in v5.
# All content (features, taglines, diffusion prompts) is now generated via:
#   Priority 1: Gemini Vision RAG - analyzes actual retrieved dataset ad images
#   Priority 2: CLIP Descriptor RAG - ranks descriptions against dataset images
#   Priority 3: Minimal brand-based fallback (no hardcoded category dicts)
# ---------------------------------------------------------------------------

DEFAULT_NEGATIVE_PROMPT = (
    "text, words, letters, numbers, typography, watermark, logo, label, "
    "blurry, low quality, distorted, deformed, ugly, oversaturated, "
    "extra fingers, extra limbs, disfigured face, bad anatomy, "
    "poorly drawn hands, poorly drawn face, mutation"
)


# ---------------------------------------------------------------------------
# Brand Matcher
# ---------------------------------------------------------------------------

class BrandMatcher:
    """Fuzzy match brand names from user queries against dataset brands."""

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

        # Also build a version with underscores preserved for matching
        self.brand_names_with_spaces: List[str] = [
            b.lower().replace("_", " ") for b in self.brand_names
        ]

    def match(self, query: str) -> BrandMatch:
        """Extract brand name from query using fuzzy matching."""
        words = query.lower().split()
        best_brand = None
        best_ratio = 0.0
        best_token = ""

        # Strategy 1: try each non-stop word individually
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

        # Strategy 2: try consecutive word pairs (e.g. "flying machine")
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

        # Strategy 3: try three-word combinations
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


# ---------------------------------------------------------------------------
# Color Extractor
# ---------------------------------------------------------------------------

class ColorExtractor:
    """Extract dominant colors from images using K-means clustering."""

    RESIZE_DIM = 100

    def extract_from_images(
        self, image_paths: List[str], n_colors: int = 5
    ) -> List[str]:
        all_pixels = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                img_small = img.resize((self.RESIZE_DIM, self.RESIZE_DIM))
                pixels = np.array(img_small).reshape(-1, 3)
                all_pixels.append(pixels)
            except Exception:
                continue

        if not all_pixels:
            return ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560"]

        combined = np.vstack(all_pixels)
        if len(combined) > 30000:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(combined), 30000, replace=False)
            combined = combined[indices]

        km = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
        km.fit(combined)

        colors_rgb = km.cluster_centers_.astype(int)
        counts = Counter(km.labels_)
        sorted_labels = sorted(counts.items(), key=lambda x: -x[1])

        hex_colors = []
        for label, _ in sorted_labels:
            c = colors_rgb[label]
            hex_colors.append(f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}")

        return hex_colors

    @staticmethod
    def get_accent_color(hex_colors: List[str]) -> str:
        """Pick the most vibrant color (highest saturation, mid-brightness)."""
        best = hex_colors[0] if hex_colors else "#1a1a2e"
        best_score = -1

        for h in hex_colors:
            r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            sat = (max_c - min_c) / max_c if max_c > 0 else 0
            brightness = (r + g + b) / 3
            # Prefer saturated, mid-brightness colors
            if 30 < brightness < 220:
                score = sat * 100 + (1 - abs(brightness - 128) / 128) * 30
            else:
                score = sat * 20
            if score > best_score:
                best_score = score
                best = h

        return best

    @staticmethod
    def lighten_color(hex_color: str, factor: float = 0.92) -> str:
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def darken_color(hex_color: str, factor: float = 0.6) -> str:
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Logo Extractor
# ---------------------------------------------------------------------------

class LogoExtractor:
    """Extract logo crops from ad images using heuristic corner analysis."""

    @staticmethod
    def extract_logo_crop(image_paths: List[str]) -> Optional[Image.Image]:
        """Scan corner regions of multiple ad images, return best logo candidate.

        Strategy: crop 5 candidate regions per image (4 corners + center-bottom),
        score by edge density + color variance, filter out uniform regions,
        return the highest-scoring crop.
        """
        candidates: List[Tuple[float, Image.Image]] = []

        for path in image_paths[:5]:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue

            w, h = img.size
            if w < 50 or h < 50:
                continue

            # Define candidate regions
            regions = [
                img.crop((0, 0, int(w * 0.35), int(h * 0.25))),                   # top-left
                img.crop((int(w * 0.65), 0, w, int(h * 0.25))),                   # top-right
                img.crop((0, int(h * 0.75), int(w * 0.35), h)),                   # bottom-left
                img.crop((int(w * 0.65), int(h * 0.75), w, h)),                   # bottom-right
                img.crop((int(w * 0.30), int(h * 0.78), int(w * 0.70), h)),       # center-bottom
            ]

            for crop in regions:
                arr = np.array(crop).astype(float)
                if arr.size == 0:
                    continue

                mean_val = arr.mean()
                # Skip near-white or near-black regions
                if mean_val > 240 or mean_val < 15:
                    continue

                # Variance score (diverse colors = more likely logo)
                variance = arr.var() / 1000.0

                # Edge density (sharp transitions = text/logo shapes)
                edges_h = np.abs(np.diff(arr, axis=0)).mean()
                edges_v = np.abs(np.diff(arr, axis=1)).mean()
                edge_score = (edges_h + edges_v) / 2.0

                score = variance * 0.4 + edge_score * 0.6
                candidates.append((score, crop))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]


# ---------------------------------------------------------------------------
# Web Logo Fetcher
# ---------------------------------------------------------------------------

class WebLogoFetcher:
    """Fetch brand logos from the web and remove background for transparency."""

    BRAND_DOMAINS = {
        "Nike": "nike.com",
        "Adidas": "adidas.com",
        "Puma": "puma.com",
        "Reebok": "reebok.com",
        "Skechers": "skechers.com",
        "FILA": "fila.com",
        "Bata": "bata.in",
        "HRX": "hrx.in",
        "Bisleri": "bisleri.com",
        "Coca_Cola": "coca-cola.com",
        "Pepsi": "pepsi.com",
        "Amul": "amul.com",
        "Patanjali": "patanjaliayurved.net",
        "Samsung_mobiles": "samsung.com",
        "Apple_mobiles": "apple.com",
        "Motorola_mobiles": "motorola.com",
        "One-Plus_mobiles": "oneplus.com",
        "maruti_suzuki": "marutisuzuki.com",
        "tata": "tata.com",
        "mahindra": "mahindra.com",
        "hyundai": "hyundai.co.in",
        "honda": "honda.com",
        "toyota": "toyota.com",
        "tvs": "tvsmotor.com",
        "hero_motocorp": "heromotocorp.com",
        "bajaj": "bajajauto.com",
        "ICICI_Bank": "icicibank.com",
        "State_Bank_Of_India": "sbi.co.in",
        "LIC": "licindia.in",
        "Tanishq_jewellary": "tanishq.co.in",
        "Titan_watches": "titan.co.in",
        "Tommy_Hilfiger_watches": "tommyhilfiger.com",
        "Casio_watches": "casio.com",
        "Rolex_watches": "rolex.com",
        "Fastrack_watches": "fastrack.in",
        "Allen_Solly": "allensolly.com",
        "Louis_Philippe": "louisphilippe.com",
        "Peter_England": "peterengland.com",
        "Monte_Carlo": "montecarlo.in",
        "Raymonds": "raymond.in",
        "Flying_Machine": "flyingmachine.co.in",
        "Lux_soaps": "lux.com",
        "Pantene_shampoo": "pantene.com",
        "Sunsilk_shampoo": "sunsilk.com",
        "Head_&_Shoulders_shampoo": "headandshoulders.com",
        "Lotus_Herbal": "lotusherbals.com",
        "Air_India": "airindia.com",
        "Make_MyTrip": "makemytrip.com",
        "Yatra": "yatra.com",
        "Joyalukkas_jewellary": "joyalukkas.com",
        "Malabar_gold_and_diamond": "malabargoldanddiamonds.com",
        "Kalyan_jewellers": "kalyanjewellers.net",
        "Blackberrys": "blackberrys.in",
        "Sparx": "sparxshoes.com",
        "Woodland": "woodlandworldwide.com",
        "Levi_s": "levi.com",
        "Wrangler": "wrangler.com",
        "Lux_Cozy": "luxinnerwear.com",
        "Van_Heusen": "vanheusen.com",
        "US_Polo": "uspoloassn.com",
        "Arrow": "arrowlife.com",
        "Park_Avenue": "parkavenue.in",
        "Wildcraft": "wildcraft.com",
        "Parle": "parleproducts.com",
        "Cadbury": "cadbury.co.in",
        "Britannia": "britannia.co.in",
        "ITC": "itcportal.com",
        "Dabur": "dabur.com",
        "Himalaya": "himalayawellness.in",
        "Lakme": "lakmeindia.com",
        "Garnier": "garnier.in",
        "Dove": "dove.com",
        "Nivea": "nivea.in",
        "Oppo_mobiles": "oppo.com",
        "Vivo_mobiles": "vivo.com",
        "Realme_mobiles": "realme.com",
        "Xiaomi_mobiles": "mi.com",
        "Kingfisher": "kingfisherworld.com",
        "Thums_Up": "coca-colacompany.com",
    }

    @classmethod
    def fetch(cls, brand_name: str) -> Optional[Image.Image]:
        """Fetch brand logo from web sources, return RGBA image with transparent bg."""
        domain = cls._get_domain(brand_name)

        # Try Clearbit Logo API
        logo = cls._try_clearbit(domain)
        if logo:
            processed = cls._ensure_transparency(logo)
            if processed:
                print(f"    Logo fetched from Clearbit ({domain})")
                return processed

        # Try Google Favicon (high-res)
        logo = cls._try_google_favicon(domain)
        if logo and min(logo.size) >= 48:
            print(f"    Logo fetched from Google Favicon ({domain})")
            return logo.convert("RGBA")

        return None

    @classmethod
    def _get_domain(cls, brand_name: str) -> str:
        if brand_name in cls.BRAND_DOMAINS:
            return cls.BRAND_DOMAINS[brand_name]
        clean = brand_name.lower().replace("_", "").replace(" ", "")
        for suffix in ("mobiles", "watches", "jewellary", "soaps", "shampoo",
                        "baby_products", "footwear"):
            if clean.endswith(suffix):
                clean = clean[: -len(suffix)]
                break
        return f"{clean}.com"

    @classmethod
    def _try_clearbit(cls, domain: str) -> Optional[Image.Image]:
        try:
            url = f"https://logo.clearbit.com/{domain}"
            resp = requests.get(url, timeout=10, allow_redirects=True)
            ct = resp.headers.get("content-type", "")
            if resp.status_code == 200 and "image" in ct:
                return Image.open(io.BytesIO(resp.content)).convert("RGBA")
        except Exception:
            pass
        return None

    @classmethod
    def _try_google_favicon(cls, domain: str) -> Optional[Image.Image]:
        try:
            url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
            resp = requests.get(url, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                img = Image.open(io.BytesIO(resp.content))
                if img.size[0] >= 32:
                    return img.convert("RGBA")
        except Exception:
            pass
        return None

    @staticmethod
    def _ensure_transparency(img: Image.Image) -> Optional[Image.Image]:
        """Ensure logo has transparent background by removing dominant corner color."""
        img = img.convert("RGBA")
        arr = np.array(img)

        # Already has meaningful transparency?
        if arr.shape[2] >= 4:
            transparent_ratio = (arr[:, :, 3] < 128).sum() / (
                arr.shape[0] * arr.shape[1]
            )
            if transparent_ratio > 0.03:
                return img

        h, w = arr.shape[:2]
        if h < 10 or w < 10:
            return img

        # Sample corner pixels to detect background color
        border = max(3, min(h, w) // 15)
        corner_pixels: List[tuple] = []
        for cy_s, cx_s in [
            (0, 0), (0, w - border), (h - border, 0), (h - border, w - border)
        ]:
            for dy in range(border):
                for dx in range(border):
                    cy = min(cy_s + dy, h - 1)
                    cx = min(cx_s + dx, w - 1)
                    corner_pixels.append(tuple(arr[cy, cx, :3].tolist()))

        bg_color = np.array(Counter(corner_pixels).most_common(1)[0][0])

        # Remove background pixels
        diff = np.abs(arr[:, :, :3].astype(int) - bg_color.astype(int))
        bg_mask = np.all(diff <= 30, axis=2)

        result = arr.copy()
        result[bg_mask, 3] = 0

        # Safety: if we removed >95% of pixels, background detection failed
        remaining = (result[:, :, 3] > 128).sum() / (h * w)
        if remaining < 0.05:
            return img

        return Image.fromarray(result)




# ---------------------------------------------------------------------------
# CLIP Content Extractor (RAG-based)
# ---------------------------------------------------------------------------

class CLIPContentExtractor:
    """RAG-based content generation: rank text descriptions against actual retrieved ad images using CLIP."""

    FEATURE_POOL = [
        # General product qualities
        "Premium quality craftsmanship",
        "Trusted by millions worldwide",
        "Award-winning innovative design",
        "Unbeatable value for money",
        "Industry-leading performance",
        "Eco-friendly sustainable choice",
        # Footwear & Apparel
        "Lightweight comfortable fit",
        "Breathable mesh upper design",
        "Advanced cushioning technology",
        "Durable long-lasting build",
        "Moisture-wicking performance fabric",
        "Athletic performance engineered",
        "Trendy modern street style",
        "All-day support and comfort",
        # Beverages & Water
        "Pure refreshing hydration",
        "Mineral-enriched natural water",
        "Natural fruit ingredients",
        "Zero artificial preservatives",
        "Clinically tested purity",
        "Refreshing clean taste",
        # Electronics & Tech
        "Cutting-edge processor power",
        "Stunning high-resolution display",
        "All-day battery performance",
        "Professional camera system",
        "5G ultra-fast connectivity",
        "Sleek premium metal build",
        # Beauty & Cosmetics
        "Dermatologically tested formula",
        "Long-lasting natural radiance",
        "Gentle on all skin types",
        "Clinically proven visible results",
        "Nourishing botanical ingredients",
        "Salon-quality results at home",
        # Food & Snacks
        "Made with natural ingredients",
        "Irresistible authentic flavor",
        "Fresh quality guaranteed daily",
        "Nutrition-packed healthy goodness",
        "Loved by food enthusiasts",
        # Automotive
        "Powerful engine performance",
        "Advanced safety technology",
        "Fuel-efficient smart engineering",
        "Spacious luxury interior cabin",
        "Connected smart driving tech",
        # Jewellery & Watches
        "Exquisite handcrafted artistry",
        "Certified precious fine materials",
        "Timeless elegant classic design",
        "Precision Swiss-quality movement",
        "Heritage craftsmanship tradition",
        # Financial & Insurance
        "Secure trusted digital banking",
        "Digital-first modern convenience",
        "Quick hassle-free processing",
        "Comprehensive financial solutions",
        # Travel & Lifestyle
        "Best price guaranteed always",
        "Seamless easy booking experience",
        "World-class premium travel service",
        "Memorable journey adventure awaits",
    ]

    STYLE_POOL = [
        "dramatic studio lighting on dark background",
        "bright outdoor natural sunlight setting",
        "warm golden hour sunset glow",
        "clean minimalist white background",
        "urban city street environment",
        "lush green nature backdrop",
        "modern gym fitness environment",
        "luxurious upscale premium interior",
        "casual home lifestyle setting",
        "professional office workspace",
        "vibrant colorful dynamic style",
        "soft focus bokeh background",
        "high-contrast dramatic photography",
        "close-up detailed product shot",
        "wide-angle scenic composition",
        "sleek futuristic tech aesthetic",
        "warm cozy family atmosphere",
        "fresh outdoor adventure setting",
        "elegant fashion editorial style",
        "dynamic sports action moment",
    ]

    MOOD_POOL = [
        "energetic dynamic movement",
        "calm peaceful serenity",
        "luxurious premium elegance",
        "fun playful joyful energy",
        "professional confident trust",
        "fresh refreshing vitality",
        "warm family togetherness",
        "bold powerful strength",
        "sophisticated refined taste",
        "adventurous exciting freedom",
    ]

    SUBJECT_POOL = [
        "person wearing the product proudly",
        "athlete in dynamic action pose",
        "model showcasing fashion outfit",
        "person drinking refreshing beverage",
        "close-up of product in use",
        "family enjoying product together",
        "professional using tech device",
        "person with radiant glowing skin",
        "driver in luxury vehicle interior",
        "person wearing elegant fine jewelry",
        "fitness enthusiast working out intensely",
        "traveler at beautiful scenic destination",
        "person savoring delicious food",
        "rider on motorcycle open road",
        "person with beautiful flowing hair",
    ]

    def __init__(self, clip_model, clip_processor, device: str):
        self.model = clip_model
        self.processor = clip_processor
        self.device = device

    def _encode_texts(self, texts):
        """Encode text descriptions using CLIP text encoder."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            return F.normalize(features, p=2, dim=-1)

    def _encode_images(self, image_paths):
        """Encode images using CLIP image encoder."""
        images = []
        for path in image_paths[:5]:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception:
                continue
        if not images:
            return torch.zeros(1, 512).to(self.device)
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=pixel_values)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            return F.normalize(features, p=2, dim=-1)

    def _rank_pool(self, image_features, pool, top_n):
        """Rank text pool by CLIP similarity to averaged image features."""
        if image_features.sum() == 0:
            return pool[:top_n]
        avg_feat = F.normalize(image_features.mean(dim=0, keepdim=True), p=2, dim=-1)
        text_feat = self._encode_texts(pool)
        similarities = (avg_feat @ text_feat.T).squeeze(0)
        top_indices = similarities.argsort(descending=True)[:top_n].tolist()
        return [pool[i] for i in top_indices]

    def extract_features(self, image_paths, n=6):
        """RAG: rank feature descriptions against actual retrieved ad images."""
        print("    [CLIP RAG] Ranking features against dataset images...")
        img_feats = self._encode_images(image_paths)
        features = self._rank_pool(img_feats, self.FEATURE_POOL, n)
        print(f"    Top features: {features[:3]}")
        return features

    def generate_diffusion_prompt(self, image_paths, brand, category, subcategory):
        """RAG: build diffusion prompt from CLIP-ranked style/mood/subject descriptions."""
        print("    [CLIP RAG] Building diffusion prompt from dataset images...")
        img_feats = self._encode_images(image_paths)

        top_styles = self._rank_pool(img_feats, self.STYLE_POOL, 2)
        top_moods = self._rank_pool(img_feats, self.MOOD_POOL, 1)
        top_subjects = self._rank_pool(img_feats, self.SUBJECT_POOL, 1)

        brand_clean = brand.replace("_", " ")
        sub_clean = subcategory.replace("_", " ") if subcategory else category.replace("_", " ")

        prompt = (
            f"professional advertisement photography for {brand_clean} {sub_clean}, "
            f"{top_subjects[0]}, {top_styles[0]}, {top_styles[1]}, "
            f"{top_moods[0]} mood, high quality 4k, sharp focus, commercial advertisement"
        )
        print(f"    Built prompt: {prompt[:80]}...")
        return prompt


# ---------------------------------------------------------------------------
# Gemini Client (copied from 08, handles vision + text)
# ---------------------------------------------------------------------------

class GeminiClient:
    MAX_IMAGE_DIMENSION = 1024
    MAX_RETRIES = 2
    MODEL_CHAIN = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]

    def __init__(self, api_key: str):
        try:
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=api_key)
            self.types = types
            self.model_name = self.MODEL_CHAIN[0]
        except ImportError:
            raise ImportError("google-genai package required: pip install google-genai")
        self.api_available = True

    @staticmethod
    def _is_daily_quota_exhausted(error_msg: str) -> bool:
        return "PerDay" in error_msg or "limit: 0" in error_msg

    @staticmethod
    def _parse_retry_delay(error_msg: str) -> Optional[float]:
        match = re.search(r'retry in ([\d.]+)s', error_msg, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def _call_with_retry(self, content_parts, system_instruction=None):
        if not self.api_available:
            raise RuntimeError("Gemini daily quota exhausted")

        last_error = None
        for model_name in self.MODEL_CHAIN:
            for attempt in range(self.MAX_RETRIES):
                try:
                    config = self.types.GenerateContentConfig(
                        max_output_tokens=2048, temperature=0.7,
                    )
                    if system_instruction:
                        config.system_instruction = system_instruction
                    response = self.client.models.generate_content(
                        model=model_name, contents=content_parts, config=config,
                    )
                    if model_name != self.model_name:
                        print(f"    (using model: {model_name})")
                        self.model_name = model_name
                    return response.text
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                    if not is_rate_limit:
                        break
                    if self._is_daily_quota_exhausted(err_str):
                        self.api_available = False
                        print(f"    {model_name}: daily quota exhausted")
                        break
                    if attempt < self.MAX_RETRIES - 1:
                        api_delay = self._parse_retry_delay(err_str)
                        delay = min(api_delay or 15, 20)
                        print(f"    {model_name}: rate limited, waiting {delay:.0f}s")
                        time.sleep(delay)
                    else:
                        break
            else:
                continue
            if not self.api_available:
                break

        self.api_available = False
        raise last_error or RuntimeError("Gemini API unavailable")

    def extract_ad_text(self, image_paths: List[str]) -> Dict[str, Any]:
        """Use Gemini Vision to extract text content from ad images."""
        if not self.api_available:
            raise RuntimeError("Gemini daily quota exhausted")

        content_parts = []
        valid_count = 0
        for path in image_paths[:3]:
            try:
                img = Image.open(path).convert("RGB")
                w, h = img.size
                if max(w, h) > self.MAX_IMAGE_DIMENSION:
                    scale = self.MAX_IMAGE_DIMENSION / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                img_part = self.types.Part.from_bytes(
                    data=buf.getvalue(), mime_type="image/jpeg"
                )
                content_parts.append(f"--- Ad Image #{valid_count + 1} ---")
                content_parts.append(img_part)
                valid_count += 1
            except Exception:
                continue

        if valid_count == 0:
            raise RuntimeError("No valid images to analyze")

        content_parts.append(
            """Extract ALL visible text from these advertisement images. Respond in JSON:
{
  "tagline": "The main headline/tagline text visible in the ads",
  "features": ["Feature 1 from the ad", "Feature 2", "Feature 3", "Feature 4"],
  "brand_text": "Brand name as displayed",
  "other_text": ["Any other text visible in the ads"]
}
Only include text you can ACTUALLY READ from the images. Do not make up text."""
        )

        raw = self._call_with_retry(
            content_parts,
            system_instruction="Extract visible text from ads. Respond with JSON only.",
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        return json.loads(cleaned)

    def extract_full_ad_content(self, image_paths, brand, category, subcategory):
        """Gemini Vision RAG: analyze actual dataset ad images to extract ALL content.

        Returns taglines, features, visual style analysis, AND a diffusion prompt
        that matches the style of the real ads in the dataset.
        """
        if not self.api_available:
            raise RuntimeError("Gemini daily quota exhausted")

        content_parts = []
        valid_count = 0
        for path in image_paths[:3]:
            try:
                img = Image.open(path).convert("RGB")
                w, h = img.size
                if max(w, h) > self.MAX_IMAGE_DIMENSION:
                    scale = self.MAX_IMAGE_DIMENSION / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                img_part = self.types.Part.from_bytes(
                    data=buf.getvalue(), mime_type="image/jpeg"
                )
                content_parts.append(f"--- Reference Ad Image #{valid_count + 1} ---")
                content_parts.append(img_part)
                valid_count += 1
            except Exception:
                continue

        if valid_count == 0:
            raise RuntimeError("No valid images to analyze")

        brand_clean = brand.replace("_", " ")
        sub_clean = subcategory.replace("_", " ") if subcategory else category.replace("_", " ")

        content_parts.append(
            f"""You are analyzing REAL advertisement images for the brand "{brand_clean}" ({sub_clean}).

Study these reference ads carefully and respond in JSON:
{{
  "tagline": "The main headline/tagline visible in the ads, or a compelling one matching the brand style",
  "features": ["Feature 1 (3-5 words)", "Feature 2", "Feature 3", "Feature 4", "Feature 5", "Feature 6"],
  "visual_style": "Describe the visual style: lighting, colors, composition, mood of these ads",
  "diffusion_prompt": "Write a detailed text-to-image prompt that recreates an ad in the SAME visual style as these reference images. Must include: a person using/wearing/holding the {sub_clean} product, matching setting, lighting, mood. Do NOT include text, logos, or words - only visual elements.",
  "negative_prompt": "Things to avoid in the generated image"
}}

IMPORTANT:
- Extract text you can ACTUALLY READ from the images
- Features should be short (3-5 words each), inspired by what the ads communicate
- The diffusion prompt must describe a PERSON using the {sub_clean} product
- Match the visual style and energy of these real ads"""
        )

        raw = self._call_with_retry(
            content_parts,
            system_instruction="Expert ad analyst. Extract content from ads and generate matching creative briefs. JSON only.",
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        return json.loads(cleaned)



# ---------------------------------------------------------------------------
# Image Generator (copied from 08)
# ---------------------------------------------------------------------------

class ImageGenerator:
    POLLINATIONS_TIMEOUT = 120
    POLLINATIONS_RETRIES = 2
    HF_INFERENCE_URL = (
        "https://router.huggingface.co/hf-inference/models/"
        "black-forest-labs/FLUX.1-schnell"
    )

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token

    def generate(
        self, prompt: str, negative_prompt: str = "",
        width: int = 1024, height: int = 768,
    ) -> Tuple[Optional[Image.Image], str]:
        img = self._try_pollinations(prompt, width, height)
        if img:
            return img, "pollinations_flux"
        img = self._try_hf_inference(prompt)
        if img:
            return img, "hf_inference_flux"
        img = self._make_gradient(width, height)
        return img, "gradient_fallback"

    def _try_pollinations(self, prompt, width, height) -> Optional[Image.Image]:
        print("\n  Trying Pollinations.ai (FLUX)...")
        encoded = urllib.parse.quote(prompt, safe="")
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={width}&height={height}&model=flux&nologo=true"
        )
        for attempt in range(self.POLLINATIONS_RETRIES):
            try:
                label = f" (attempt {attempt + 1})" if attempt > 0 else ""
                print(f"    Requesting image{label} (may take 30-90s)...")
                resp = requests.get(url, timeout=self.POLLINATIONS_TIMEOUT)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    print(f"    Success! Image size: {img.size}")
                    return img
                elif resp.status_code >= 500:
                    print(f"    Server error {resp.status_code}, retrying...")
                    time.sleep(5)
                else:
                    print(f"    Failed: status={resp.status_code}")
                    return None
            except requests.Timeout:
                print(f"    Timeout after {self.POLLINATIONS_TIMEOUT}s")
            except Exception as e:
                print(f"    Error: {e}")
        return None

    def _try_hf_inference(self, prompt) -> Optional[Image.Image]:
        if not self.hf_token:
            print("\n  Skipping HF Inference (no token)")
            return None
        print("\n  Trying HuggingFace Inference API (FLUX.1-schnell)...")
        try:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            resp = requests.post(
                self.HF_INFERENCE_URL, headers=headers,
                json={"inputs": prompt}, timeout=120,
            )
            if resp.status_code == 200 and len(resp.content) > 1000:
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                print(f"    Success! Image size: {img.size}")
                return img
            else:
                print(f"    Failed: status={resp.status_code}")
                if resp.status_code != 200:
                    print(f"    Response: {resp.text[:200]}")
        except Exception as e:
            print(f"    Error: {e}")
        return None

    def _make_gradient(
        self, width=1024, height=768, colors=None
    ) -> Image.Image:
        print("\n  Creating gradient fallback background...")
        if not colors or len(colors) < 2:
            colors = ["#1a1a2e", "#16213e", "#0f3460", "#533483"]
        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)

        def hex_to_rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

        c1 = hex_to_rgb(colors[0])
        c2 = hex_to_rgb(colors[1])
        for y in range(height):
            t = y / height
            r = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            b = int(c1[2] + (c2[2] - c1[2]) * t)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        return img


# ---------------------------------------------------------------------------
# Pamphlet Composer
# ---------------------------------------------------------------------------

class PamphletComposer:
    """AI-driven dynamic pamphlet composer. Analyzes product image to determine
    layout, places text on the quiet side, logo in free space, and varies the
    design so no two brands look the same."""

    WIDTH = 1080
    HEIGHT = 1080
    MARGIN = 55

    CTA_TEXTS = {
        "footwear": "SHOP NOW", "clothing_brands": "SHOP NOW",
        "mobile_devices": "BUY NOW", "computers": "BUY NOW",
        "four_wheelers": "BOOK A TEST DRIVE", "two_wheelers": "BOOK A TEST DRIVE",
        "water": "ORDER NOW", "soft_drinks": "GRAB YOURS",
        "juice": "TRY NOW", "banks": "OPEN ACCOUNT",
        "insurance": "GET QUOTE", "jewellery": "EXPLORE COLLECTION",
        "watches": "EXPLORE COLLECTION", "skincare_and_makeup": "SHOP NOW",
    }

    # Banner color themes (varied per brand)
    BANNER_THEMES = [
        ((140, 30, 140), (180, 50, 180)),   # purple/magenta
        ((30, 100, 180), (50, 130, 210)),    # blue
        ((180, 60, 30), (210, 90, 50)),      # orange/red
        ((30, 140, 100), (50, 180, 130)),    # teal/green
        ((160, 120, 30), (200, 160, 50)),    # gold
    ]

    def __init__(self):
        self._font_cache: Dict[str, ImageFont.FreeTypeFont] = {}

    def _font(self, path: str, size: int) -> ImageFont.FreeTypeFont:
        key = f"{path}_{size}"
        if key not in self._font_cache:
            try:
                self._font_cache[key] = ImageFont.truetype(path, size)
            except Exception:
                try:
                    self._font_cache[key] = ImageFont.truetype("arial.ttf", size)
                except Exception:
                    self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def _text_size(self, draw: ImageDraw.Draw, text: str, font) -> Tuple[int, int]:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    @staticmethod
    def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
        h = h.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _round_corners(self, img: Image.Image, radius: int) -> Image.Image:
        mask = Image.new("L", img.size, 0)
        d = ImageDraw.Draw(mask)
        d.rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius, fill=255)
        result = img.convert("RGBA")
        result.putalpha(mask)
        return result

    # ------------------------------------------------------------------
    # AI: Analyze product image to decide layout
    # ------------------------------------------------------------------
    def _analyze_image(self, img: Image.Image) -> Dict[str, Any]:
        """Divide image into a 3x3 grid, compute complexity per cell.
        Returns: which side is busy, where free space is, avg brightness."""
        arr = np.array(img.convert("RGB")).astype(float)
        h, w = arr.shape[:2]
        grid_h, grid_w = h // 3, w // 3

        complexity = np.zeros((3, 3))
        brightness = np.zeros((3, 3))
        for r in range(3):
            for c in range(3):
                cell = arr[r * grid_h:(r + 1) * grid_h, c * grid_w:(c + 1) * grid_w]
                # Complexity = edge density (gradient magnitude)
                dx = np.abs(np.diff(cell, axis=1)).mean()
                dy = np.abs(np.diff(cell, axis=0)).mean()
                complexity[r, c] = dx + dy
                brightness[r, c] = cell.mean()

        # Determine busy side: left vs right
        left_complexity = complexity[:, 0].mean()
        right_complexity = complexity[:, 2].mean()
        center_complexity = complexity[:, 1].mean()

        if left_complexity > right_complexity * 1.15:
            text_side = "right"
        elif right_complexity > left_complexity * 1.15:
            text_side = "left"
        else:
            text_side = "left"  # default

        # Find free space cell (lowest complexity) for logo
        flat_idx = int(np.argmin(complexity))
        free_row, free_col = flat_idx // 3, flat_idx % 3
        # Map to position name
        pos_names = [
            ["top-left", "top-center", "top-right"],
            ["mid-left", "mid-center", "mid-right"],
            ["bot-left", "bot-center", "bot-right"],
        ]
        free_space = pos_names[free_row][free_col]

        # Average brightness
        avg_brightness = brightness.mean()
        is_dark = avg_brightness < 120

        # Top vs bottom complexity for vertical text placement
        top_busy = complexity[0, :].mean()
        bot_busy = complexity[2, :].mean()
        brand_position = "top" if bot_busy > top_busy * 1.2 else "top"

        return {
            "text_side": text_side,
            "free_space": free_space,
            "free_row": free_row,
            "free_col": free_col,
            "avg_brightness": avg_brightness,
            "is_dark": is_dark,
            "complexity_grid": complexity,
            "brand_position": brand_position,
        }

    def _get_layout_config(self, content: PamphletContent, analysis: Dict) -> Dict:
        """Generate unique layout config based on image analysis + brand hash."""
        # Brand-based seed for deterministic variety
        brand_hash = sum(ord(c) for c in content.brand_name)

        text_side = analysis["text_side"]
        # Feature styles cycle based on brand
        feat_styles = ["bullets", "pills", "lines"]
        feat_style = feat_styles[brand_hash % len(feat_styles)]

        # Thumbnail styles cycle
        thumb_styles = ["tilted_cascade", "horizontal_strip", "stacked"]
        thumb_style = thumb_styles[(brand_hash // 3) % len(thumb_styles)]

        # Banner theme
        banner_theme = self.BANNER_THEMES[brand_hash % len(self.BANNER_THEMES)]

        # CTA button color variations
        cta_colors = [
            (240, 190, 50),   # gold
            (220, 60, 60),    # red
            (50, 180, 120),   # green
            (60, 140, 220),   # blue
            (220, 120, 50),   # orange
        ]
        cta_color = cta_colors[(brand_hash // 5) % len(cta_colors)]

        # Thumbnail position: opposite side of text
        thumb_side = "left" if text_side == "right" else "right"

        # Brand name size variation
        brand_sizes = [52, 58, 48, 54]
        brand_size = brand_sizes[brand_hash % len(brand_sizes)]

        return {
            "text_side": text_side,
            "feat_style": feat_style,
            "thumb_style": thumb_style,
            "thumb_side": thumb_side,
            "banner_theme": banner_theme,
            "cta_color": cta_color,
            "brand_size": brand_size,
            "free_space": analysis["free_space"],
            "free_row": analysis["free_row"],
            "free_col": analysis["free_col"],
        }

    # ------------------------------------------------------------------
    # Main compose
    # ------------------------------------------------------------------
    def compose(self, content: PamphletContent) -> Image.Image:
        """AI-driven layout: analyzes image, places elements dynamically."""
        W, H = self.WIDTH, self.HEIGHT
        accent_rgb = self._hex_to_rgb(content.accent_color)

        # ===== STEP 1: Full-bleed product image =====
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        if content.product_image:
            prod = content.product_image.copy().convert("RGB")
            pw, ph = prod.size
            scale = max(W / pw, H / ph)
            nw, nh = int(pw * scale), int(ph * scale)
            prod_resized = prod.resize((nw, nh), Image.LANCZOS)
            lx, ly = (nw - W) // 2, (nh - H) // 2
            prod_cropped = prod_resized.crop((lx, ly, lx + W, ly + H))
            canvas.paste(prod_cropped.convert("RGBA"), (0, 0))
        else:
            d1 = self._hex_to_rgb(ColorExtractor.darken_color(content.accent_color, 0.25))
            d2 = self._hex_to_rgb(ColorExtractor.darken_color(content.accent_color, 0.60))
            for row in range(H):
                t = row / H
                rgb = tuple(int(d1[i] + (d2[i] - d1[i]) * t) for i in range(3))
                ImageDraw.Draw(canvas).line([(0, row), (W, row)], fill=(*rgb, 255))

        # ===== STEP 2: AI analysis =====
        if content.product_image:
            analysis = self._analyze_image(content.product_image)
        else:
            analysis = {"text_side": "left", "free_space": "top-right",
                        "free_row": 0, "free_col": 2, "avg_brightness": 60,
                        "is_dark": True, "complexity_grid": np.zeros((3, 3)),
                        "brand_position": "top"}

        cfg = self._get_layout_config(content, analysis)
        text_side = cfg["text_side"]
        is_right = text_side == "right"
        M = self.MARGIN

        # ===== STEP 3: Gradient overlay (on text side) =====
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)

        # Top gradient for brand name area
        for row in range(int(H * 0.35)):
            t = row / (H * 0.35)
            alpha = int(185 * (1 - t) ** 1.3)
            od.line([(0, row), (W, row)], fill=(0, 0, 0, alpha))

        # Side gradient for features (on text side)
        side_band = Image.new("RGBA", (int(W * 0.60), H), (0, 0, 0, 0))
        sb_draw = ImageDraw.Draw(side_band)
        band_w = int(W * 0.60)
        for col in range(band_w):
            t = col / band_w
            alpha = int(150 * (1 - t) ** 2.2)
            sb_draw.line([(col, int(H * 0.18)), (col, int(H * 0.88))], fill=(0, 0, 0, alpha))
        side_full = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if is_right:
            side_band_flipped = side_band.transpose(Image.FLIP_LEFT_RIGHT)
            side_full.paste(side_band_flipped, (W - band_w, 0))
        else:
            side_full.paste(side_band, (0, 0))
        overlay = Image.alpha_composite(overlay, side_full)

        # Bottom gradient for banner/thumbnails
        bot_h = int(H * 0.28)
        bot_g = Image.new("RGBA", (W, bot_h), (0, 0, 0, 0))
        bgd = ImageDraw.Draw(bot_g)
        for row in range(bot_h):
            t = row / bot_h
            bgd.line([(0, row), (W, row)], fill=(0, 0, 0, int(200 * t ** 1.8)))
        bot_full = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bot_full.paste(bot_g, (0, H - bot_h))
        overlay = Image.alpha_composite(overlay, bot_full)

        canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)

        # ===== STEP 4: Logo in FREE SPACE (single placement, AI-detected) =====
        # Logo goes on the OPPOSITE side of the text, in the least complex area
        if content.logo_image:
            logo = content.logo_image.copy().convert("RGBA")
            lw, lh = logo.size
            logo_h = 90
            logo_scale = logo_h / lh
            logo_w = min(int(lw * logo_scale), 180)
            logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

            # Find best free space cell on the NON-text side
            grid = analysis.get("complexity_grid", np.zeros((3, 3)))
            best_score = float("inf")
            best_r, best_c = 1, (2 if not is_right else 0)  # default: opposite mid
            for r in range(3):
                for c in range(3):
                    # Skip text side columns (top rows are brand name area)
                    if is_right and c >= 1:
                        continue  # text is right, logo must be left (col 0)
                    if not is_right and c <= 1:
                        continue  # text is left, logo must be right (col 2)
                    # Skip bottom row (banner/thumbnails area)
                    if r == 2:
                        continue
                    if grid[r, c] < best_score:
                        best_score = grid[r, c]
                        best_r, best_c = r, c

            grid_h_cell, grid_w_cell = H // 3, W // 3
            logo_x = best_c * grid_w_cell + (grid_w_cell - logo_w) // 2
            logo_y = best_r * grid_h_cell + (grid_h_cell - logo_h) // 2
            logo_x = max(M, min(logo_x, W - logo_w - M))
            logo_y = max(25, min(logo_y, H - logo_h - 100))

            # Frosted glass backing for logo visibility
            back_pad = 14
            back_w, back_h = logo_w + 2 * back_pad, logo_h + 2 * back_pad
            backing = Image.new("RGBA", (back_w, back_h), (0, 0, 0, 0))
            backing_d = ImageDraw.Draw(backing)
            backing_d.rounded_rectangle(
                [0, 0, back_w - 1, back_h - 1], 16,
                fill=(255, 255, 255, 45),
            )
            # Outer glow
            glow = backing.copy().filter(ImageFilter.GaussianBlur(8))

            bx = logo_x - back_pad
            by = logo_y - back_pad
            bx = max(0, bx)
            by = max(0, by)
            # Paste glow
            if bx + glow.width <= W and by + glow.height <= H:
                rg = canvas.crop((bx, by, bx + glow.width, by + glow.height)).convert("RGBA")
                canvas.paste(Image.alpha_composite(rg, glow), (bx, by))
            # Paste backing
            if bx + backing.width <= W and by + backing.height <= H:
                rg = canvas.crop((bx, by, bx + backing.width, by + backing.height)).convert("RGBA")
                canvas.paste(Image.alpha_composite(rg, backing), (bx, by))
            # Paste logo
            canvas.paste(logo, (logo_x, logo_y), logo)
            draw = ImageDraw.Draw(canvas)

        # ===== STEP 5: Brand name (on text side) =====
        brand_font = self._font("C:/Windows/Fonts/segoeuib.ttf", cfg["brand_size"])
        brand_display = content.brand_name.replace("_", " ").upper()
        bw_text, bh_text = self._text_size(draw, brand_display, brand_font)

        if is_right:
            bx = W - M - bw_text
        else:
            bx = M
        by = 30

        # Shadow + text
        draw.text((bx + 3, by + 3), brand_display, fill=(0, 0, 0, 180), font=brand_font)
        draw.text((bx, by), brand_display, fill=(255, 255, 255, 255), font=brand_font)

        # Tagline
        tag_y = by + bh_text + 8
        if content.tagline:
            tag_font = self._font("C:/Windows/Fonts/georgiai.ttf", 21)
            tw_tag, _ = self._text_size(draw, content.tagline, tag_font)
            tx = (W - M - tw_tag) if is_right else M
            draw.text((tx + 2, tag_y + 2), content.tagline, fill=(0, 0, 0, 130), font=tag_font)
            draw.text((tx, tag_y), content.tagline, fill=(255, 255, 255, 220), font=tag_font)
            tag_y += 34

        # ===== STEP 6: Features (style varies per brand) =====
        features = content.features[:6]
        feat_end_y = tag_y + 20
        if features:
            if cfg["feat_style"] == "bullets":
                feat_end_y = self._draw_features_bullets(
                    canvas, draw, features, tag_y + 20, M, W, is_right, accent_rgb
                )
            elif cfg["feat_style"] == "pills":
                feat_end_y = self._draw_features_pills(
                    canvas, draw, features, tag_y + 20, M, W, is_right, accent_rgb
                )
            else:  # lines
                feat_end_y = self._draw_features_lines(
                    canvas, draw, features, tag_y + 20, M, W, is_right, accent_rgb
                )
            draw = ImageDraw.Draw(canvas)

        # ===== STEP 7: CTA Button =====
        cta_text = self.CTA_TEXTS.get(
            content.subcategory, self.CTA_TEXTS.get(content.category, "EXPLORE NOW")
        )
        cta_font = self._font("C:/Windows/Fonts/segoeuib.ttf", 20)
        cta_tw, cta_th = self._text_size(draw, cta_text, cta_font)
        cta_px, cta_py = 28, 12
        cta_w, cta_h = cta_tw + 2 * cta_px, cta_th + 2 * cta_py
        cta_y = feat_end_y + 14

        btn = Image.new("RGBA", (cta_w, cta_h), (0, 0, 0, 0))
        btn_d = ImageDraw.Draw(btn)
        cc = cfg["cta_color"]
        btn_d.rounded_rectangle([0, 0, cta_w - 1, cta_h - 1], cta_h // 2, fill=(*cc, 240))
        # Text color: dark on light buttons, white on dark
        txt_lum = cc[0] * 0.299 + cc[1] * 0.587 + cc[2] * 0.114
        txt_col = (20, 20, 20, 255) if txt_lum > 140 else (255, 255, 255, 255)
        btn_d.text((cta_px, cta_py - 2), cta_text, fill=txt_col, font=cta_font)
        cta_x = (W - M - cta_w) if is_right else M
        canvas.paste(btn, (cta_x, cta_y), btn)
        draw = ImageDraw.Draw(canvas)

        # ===== STEP 8: Info banner (bottom, on text side) =====
        banner_h = 65
        banner_w = int(W * 0.48)
        banner_y = H - banner_h - 14
        banner_x = (W - banner_w) if is_right else 0

        bt1, bt2 = cfg["banner_theme"]
        banner = Image.new("RGBA", (banner_w, banner_h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(banner)
        for col in range(banner_w):
            t = col / banner_w
            rgb = tuple(int(bt1[i] + (bt2[i] - bt1[i]) * t) for i in range(3))
            alpha = int(220 * (1 - t * 0.25))
            bd.line([(col, 0), (col, banner_h)], fill=(*rgb, alpha))
        # Taper the outer edge
        taper_start = banner_w - 35
        for col in range(taper_start, banner_w):
            t = (col - taper_start) / 35
            for row in range(banner_h):
                px = banner.getpixel((col, row))
                banner.putpixel((col, row), (px[0], px[1], px[2], max(0, int(px[3] * (1 - t)))))
        if is_right:
            banner = banner.transpose(Image.FLIP_LEFT_RIGHT)

        canvas.paste(banner, (banner_x, banner_y), banner)
        draw = ImageDraw.Draw(canvas)

        info_font = self._font("C:/Windows/Fonts/segoeuib.ttf", 15)
        contact_font = self._font("C:/Windows/Fonts/segoeuib.ttf", 22)
        brand_clean = content.brand_name.replace("_", " ").lower().replace(" ", "")
        info_x = (W - M - 10) if is_right else (M + 10)

        if is_right:
            iw1, _ = self._text_size(draw, "FOR MORE INFO", info_font)
            draw.text((W - M - iw1, banner_y + 8), "FOR MORE INFO",
                       fill=(255, 255, 255, 240), font=info_font)
            url_text = f"www.{brand_clean}.com"
            iw2, _ = self._text_size(draw, url_text, contact_font)
            draw.text((W - M - iw2, banner_y + 30), url_text,
                       fill=(255, 255, 0, 255), font=contact_font)
        else:
            draw.text((M + 10, banner_y + 8), "FOR MORE INFO",
                       fill=(255, 255, 255, 240), font=info_font)
            draw.text((M + 10, banner_y + 30), f"www.{brand_clean}.com",
                       fill=(255, 255, 0, 255), font=contact_font)

        # ===== STEP 9: Thumbnails (opposite side of text) =====
        if content.thumbnail_images:
            thumbs = content.thumbnail_images[:3]
            if cfg["thumb_style"] == "tilted_cascade":
                self._draw_thumbs_tilted(canvas, thumbs, W, H, M, cfg["thumb_side"])
            elif cfg["thumb_style"] == "horizontal_strip":
                self._draw_thumbs_strip(canvas, thumbs, W, H, M, cfg["thumb_side"])
            else:
                self._draw_thumbs_stacked(canvas, thumbs, W, H, M, cfg["thumb_side"])
            draw = ImageDraw.Draw(canvas)

        # ===== STEP 10: Decorative accent =====
        sparkle_font = self._font("C:/Windows/Fonts/segoeui.ttf", 26)
        sx = M if is_right else (W - M - 20)
        draw.text((sx, H - 44), "\u2726", fill=(*accent_rgb, 160), font=sparkle_font)

        return canvas.convert("RGB")

    # ------------------------------------------------------------------
    # Feature drawing variants
    # ------------------------------------------------------------------
    def _draw_features_bullets(self, canvas, draw, features, y, margin, W, is_right, accent):
        """Vertical bullet list with dot prefix."""
        font = self._font("C:/Windows/Fonts/segoeui.ttf", 21)
        for feat in features:
            text = f"\u2022  {feat}"
            tw, _ = self._text_size(draw, text, font)
            x = (W - margin - tw) if is_right else margin
            draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 140), font=font)
            draw.text((x, y), text, fill=(255, 255, 255, 240), font=font)
            y += 34
        return y + 8

    def _draw_features_pills(self, canvas, draw, features, y, margin, W, is_right, accent):
        """Semi-transparent pill capsules, 2 columns."""
        font = self._font("C:/Windows/Fonts/segoeui.ttf", 16)
        cols = 2
        gap = 10
        pill_w = (W // 2 - margin - gap) // cols * cols  # approximate
        pill_w = min(260, (W - 2 * margin - gap) // 2)
        pill_h = 38
        base_x = (W - margin - 2 * pill_w - gap) if is_right else margin

        for idx, feat in enumerate(features):
            col = idx % cols
            row = idx // cols
            px = base_x + col * (pill_w + gap)
            py = y + row * (pill_h + gap)

            pill = Image.new("RGBA", (pill_w, pill_h), (0, 0, 0, 0))
            pd = ImageDraw.Draw(pill)
            pd.rounded_rectangle([0, 0, pill_w - 1, pill_h - 1], 19,
                                 fill=(0, 0, 0, 100), outline=(*accent, 120), width=1)
            # Accent dot
            pd.ellipse([10, pill_h // 2 - 3, 16, pill_h // 2 + 3], fill=(*accent, 220))
            # Text
            ft = feat
            tw, _ = self._text_size(pd, ft, font)
            max_tw = pill_w - 30
            while tw > max_tw and len(ft) > 8:
                ft = ft[:-2]
                tw, _ = self._text_size(pd, ft, font)
            if ft != feat:
                ft += "\u2026"
            pd.text((22, (pill_h - 16) // 2), ft, fill=(255, 255, 255, 240), font=font)

            region = canvas.crop((px, py, px + pill_w, py + pill_h)).convert("RGBA")
            canvas.paste(Image.alpha_composite(region, pill), (px, py))

        n_rows = (len(features) + cols - 1) // cols
        return y + n_rows * (pill_h + gap) + 8

    def _draw_features_lines(self, canvas, draw, features, y, margin, W, is_right, accent):
        """Underlined feature lines with accent dash prefix."""
        font = self._font("C:/Windows/Fonts/segoeuil.ttf", 20)
        for feat in features:
            text = f"\u2014  {feat}"
            tw, th = self._text_size(draw, text, font)
            x = (W - margin - tw) if is_right else margin
            draw.text((x + 1, y + 1), text, fill=(0, 0, 0, 120), font=font)
            draw.text((x, y), text, fill=(255, 255, 255, 230), font=font)
            # Subtle underline
            draw.line([(x, y + th + 3), (x + tw, y + th + 3)],
                      fill=(*accent, 60), width=1)
            y += th + 14
        return y + 6

    # ------------------------------------------------------------------
    # Thumbnail drawing variants
    # ------------------------------------------------------------------
    def _draw_thumbs_tilted(self, canvas, thumbs, W, H, M, side):
        """Tilted/cascading thumbnails with white borders, slight rotation."""
        thumb_size = 145
        angles = [-7, 4, -4]
        if side == "right":
            base_x = W - 50 - int(thumb_size * 1.8)
        else:
            base_x = 30
        base_y = H - 55 - thumb_size

        for i, thumb in enumerate(thumbs):
            t = thumb.copy().convert("RGB")
            tw, th = t.size
            sc = min(thumb_size / tw, thumb_size / th)
            tnw, tnh = int(tw * sc), int(th * sc)
            t = t.resize((tnw, tnh), Image.LANCZOS)

            bdr = 5
            framed = Image.new("RGBA", (tnw + 2 * bdr, tnh + 2 * bdr), (255, 255, 255, 220))
            framed.paste(t.convert("RGBA"), (bdr, bdr))
            framed = self._round_corners(framed, 10)

            # Drop shadow
            shadow = Image.new("RGBA", (framed.width + 8, framed.height + 8), (0, 0, 0, 0))
            sd = ImageDraw.Draw(shadow)
            sd.rounded_rectangle([4, 4, shadow.width - 1, shadow.height - 1], 12,
                                 fill=(0, 0, 0, 50))
            shadow = shadow.filter(ImageFilter.GaussianBlur(4))
            shadow.paste(framed, (0, 0), framed)
            framed = shadow

            rotated = framed.rotate(angles[i % 3], expand=True, resample=Image.BICUBIC)
            tx = int(base_x + i * (thumb_size * 0.52))
            ty = int(base_y - i * 10)
            pw = min(rotated.width, W - tx)
            ph = min(rotated.height, H - ty)
            if pw > 0 and ph > 0 and tx >= 0 and ty >= 0:
                cr = rotated.crop((0, 0, pw, ph))
                rg = canvas.crop((tx, ty, tx + pw, ty + ph)).convert("RGBA")
                canvas.paste(Image.alpha_composite(rg, cr), (tx, ty))

    def _draw_thumbs_strip(self, canvas, thumbs, W, H, M, side):
        """Horizontal strip of thumbnails with rounded corners and shadow."""
        thumb_w, thumb_h = 130, 90
        gap = 10
        total = len(thumbs) * thumb_w + (len(thumbs) - 1) * gap
        if side == "right":
            sx = W - M - total
        else:
            sx = M
        sy = H - thumb_h - 30

        for i, thumb in enumerate(thumbs):
            t = thumb.copy().convert("RGB")
            tw, th = t.size
            sc = min(thumb_w / tw, thumb_h / th)
            tnw, tnh = int(tw * sc), int(th * sc)
            t = t.resize((tnw, tnh), Image.LANCZOS)

            bdr = 4
            framed = Image.new("RGBA", (tnw + 2 * bdr, tnh + 2 * bdr), (255, 255, 255, 200))
            framed.paste(t.convert("RGBA"), (bdr, bdr))
            framed = self._round_corners(framed, 8)

            tx = sx + i * (thumb_w + gap) + (thumb_w - framed.width) // 2
            ty = sy + (thumb_h - framed.height) // 2
            if 0 <= tx and tx + framed.width <= W and 0 <= ty and ty + framed.height <= H:
                rg = canvas.crop((tx, ty, tx + framed.width, ty + framed.height)).convert("RGBA")
                canvas.paste(Image.alpha_composite(rg, framed), (tx, ty))

    def _draw_thumbs_stacked(self, canvas, thumbs, W, H, M, side):
        """Vertically stacked overlapping thumbnails with shadow."""
        thumb_size = 130
        if side == "right":
            base_x = W - M - thumb_size - 20
        else:
            base_x = M + 10
        base_y = H - 30 - int(thumb_size * 1.6)

        for i, thumb in enumerate(thumbs):
            t = thumb.copy().convert("RGB")
            tw, th = t.size
            sc = min(thumb_size / tw, (thumb_size * 0.8) / th)
            tnw, tnh = int(tw * sc), int(th * sc)
            t = t.resize((tnw, tnh), Image.LANCZOS)

            bdr = 5
            framed = Image.new("RGBA", (tnw + 2 * bdr, tnh + 2 * bdr), (255, 255, 255, 210))
            framed.paste(t.convert("RGBA"), (bdr, bdr))
            framed = self._round_corners(framed, 10)

            tx = base_x + i * 18
            ty = base_y + i * (tnh * 0.35)
            tx, ty = int(tx), int(ty)
            pw = min(framed.width, W - tx)
            ph = min(framed.height, H - ty)
            if pw > 0 and ph > 0 and tx >= 0 and ty >= 0:
                cr = framed.crop((0, 0, pw, ph))
                rg = canvas.crop((tx, ty, tx + pw, ty + ph)).convert("RGBA")
                canvas.paste(Image.alpha_composite(rg, cr), (tx, ty))


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class MAdVersePamphletPipeline:
    """Full pamphlet pipeline: Match -> Retrieve -> Colors -> Content -> Generate -> Compose."""

    def __init__(self):
        print("\n" + "=" * 80)
        print("MAdVerse Pamphlet Pipeline v5")
        print("=" * 80)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = Path("./outputs")
        self.output_dir.mkdir(exist_ok=True)

        self._init_retrieval()

        print("    3/3 Brand index...", end=" ", flush=True)
        self.brand_matcher = BrandMatcher(self.id_to_metadata)
        print(f"done ({len(self.brand_matcher.brand_names)} brands)")

        self.color_extractor = ColorExtractor()
        self.pamphlet_composer = PamphletComposer()

        self.gemini_key = os.getenv("GOOGLE_API_KEY", "")
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.llm = GeminiClient(self.gemini_key) if self.gemini_key else None
        self.image_gen = ImageGenerator(hf_token=self.hf_token or None)

        self.clip_content_extractor = CLIPContentExtractor(
            self.clip_model, self.clip_processor, self.device
        )

        print("\n  Pipeline ready!")
        print("=" * 80)

    def _init_retrieval(self):
        print("\n  Loading models...")
        print("    1/3 CLIP...", end=" ", flush=True)
        clip_name = "openai/clip-vit-base-patch32"
        try:
            self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
        except OSError:
            print("network error, using cache...", end=" ", flush=True)
            self.clip_model = CLIPModel.from_pretrained(
                clip_name, local_files_only=True
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                clip_name, local_files_only=True
            )
        self.clip_model.eval()
        print("done")

        print("    2/3 FAISS...", end=" ", flush=True)
        index_path = Path("./embeddings/faiss_indexes/madverse_index.faiss")
        metadata_path = Path("./embeddings/faiss_indexes/id_to_metadata.pkl")
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self.id_to_metadata = pickle.load(f)
        print(f"done ({self.index.ntotal:,} vectors)")

    # ------------------------------------------------------------------
    # Stage 0: Brand Matching
    # ------------------------------------------------------------------
    def stage0_match_brand(self, query: str) -> BrandMatch:
        print(f"\n{'=' * 80}")
        print("STAGE 0: BRAND MATCHING")
        print(f"{'=' * 80}")

        match = self.brand_matcher.match(query)
        if match.matched_brand:
            print(f"  Matched: '{match.query_token}' -> {match.matched_brand}")
            print(f"  Confidence: {match.confidence:.0%}")
            print(f"  Category: {match.category}/{match.subcategory}")
            print(f"  Dataset images: {match.image_count}")
        else:
            print("  No specific brand identified in query")
            print("  Will use general FAISS retrieval")
        return match

    # ------------------------------------------------------------------
    # Stage 1: Hybrid FAISS Retrieval
    # ------------------------------------------------------------------
    def stage1_retrieve(
        self, query: str, brand_match: BrandMatch, k: int = 5
    ) -> List[RetrievedAd]:
        print(f"\n{'=' * 80}")
        print("STAGE 1: RETRIEVAL")
        print(f"{'=' * 80}")

        # Encode query with CLIP
        inputs = self.clip_processor(
            text=query, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k_: v.to(self.device) for k_, v in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = F.normalize(features, p=2, dim=-1)
        query_vec = features.cpu().numpy().flatten().astype(np.float32)

        if brand_match.matched_brand:
            print(f"  Brand-filtered retrieval for: {brand_match.matched_brand}")
            brand_indices = self.brand_matcher.get_brand_indices(
                brand_match.matched_brand
            )
            # Reconstruct vectors for this brand from FAISS index
            vectors = np.array(
                [self.index.reconstruct(int(i)) for i in brand_indices],
                dtype=np.float32,
            )
            sub_index = faiss.IndexFlatL2(self.index.d)
            sub_index.add(vectors)
            actual_k = min(k, len(brand_indices))
            distances, sub_indices = sub_index.search(
                np.array([query_vec]), actual_k
            )

            results = []
            for rank, (dist, sub_idx) in enumerate(
                zip(distances[0], sub_indices[0]), 1
            ):
                if sub_idx < 0:
                    continue
                original_idx = brand_indices[int(sub_idx)]
                meta = self.id_to_metadata[original_idx]
                ad = RetrievedAd(
                    rank=rank,
                    similarity=float(1 / (1 + float(dist))),
                    distance=float(dist),
                    image_path=meta.get("image_path", ""),
                    image_id=meta.get("image_id", ""),
                    brand=meta.get("brand", ""),
                    category=meta.get("category", ""),
                    subcategory=meta.get("subcategory", ""),
                    language=meta.get("language", ""),
                    ad_type=meta.get("ad_type", ""),
                    source=meta.get("source", ""),
                )
                results.append(ad)
                print(f"  #{rank} {ad.brand} ({ad.subcategory}) - {ad.similarity:.1%}")
            return results
        else:
            print("  Standard FAISS retrieval")
            distances, indices = self.index.search(np.array([query_vec]), k)
            results = []
            for rank, (dist, idx) in enumerate(
                zip(distances[0], indices[0]), 1
            ):
                if 0 <= idx < len(self.id_to_metadata):
                    meta = self.id_to_metadata[idx]
                    ad = RetrievedAd(
                        rank=rank,
                        similarity=float(1 / (1 + float(dist))),
                        distance=float(dist),
                        image_path=meta.get("image_path", ""),
                        image_id=meta.get("image_id", ""),
                        brand=meta.get("brand", ""),
                        category=meta.get("category", ""),
                        subcategory=meta.get("subcategory", ""),
                        language=meta.get("language", ""),
                        ad_type=meta.get("ad_type", ""),
                        source=meta.get("source", ""),
                    )
                    results.append(ad)
                    print(
                        f"  #{rank} {ad.brand} ({ad.category}) - {ad.similarity:.1%}"
                    )
            return results

    # ------------------------------------------------------------------
    # Stage 2: Color Extraction
    # ------------------------------------------------------------------
    def stage2_extract_colors(self, retrieved_ads: List[RetrievedAd]) -> List[str]:
        print(f"\n{'=' * 80}")
        print("STAGE 2: COLOR EXTRACTION (from dataset images)")
        print(f"{'=' * 80}")

        image_paths = [
            ad.image_path for ad in retrieved_ads
            if Path(ad.image_path).exists()
        ]
        if not image_paths:
            print("  No valid images found, using default colors")
            return ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560"]

        print(f"  Analyzing {len(image_paths)} images with KMeans clustering...")
        colors = self.color_extractor.extract_from_images(image_paths)
        accent = ColorExtractor.get_accent_color(colors)
        print(f"  Dominant colors: {colors}")
        print(f"  Selected accent: {accent}")
        return colors

    # ------------------------------------------------------------------
    # Stage 3: Content Generation (RAG-based)
    # ------------------------------------------------------------------
    def stage3_generate_content(
        self,
        query: str,
        brand_match: BrandMatch,
        retrieved_ads: List[RetrievedAd],
        colors: List[str],
    ) -> Tuple[PamphletContent, bool]:
        print(f"\n{'=' * 80}")
        print("STAGE 3: CONTENT GENERATION (RAG-based)")
        print(f"{'=' * 80}")

        brand = brand_match.matched_brand or (
            retrieved_ads[0].brand if retrieved_ads else "Product"
        )
        category = brand_match.category or (
            retrieved_ads[0].category if retrieved_ads else "product"
        )
        subcategory = brand_match.subcategory or (
            retrieved_ads[0].subcategory if retrieved_ads else ""
        )

        # Colors
        accent = ColorExtractor.get_accent_color(colors)
        secondary = colors[1] if len(colors) > 1 else accent
        bg_tint = ColorExtractor.lighten_color(accent, factor=0.92)

        # Collect image paths for analysis
        ad_image_paths = [
            ad.image_path for ad in retrieved_ads
            if Path(ad.image_path).exists()
        ]

        # --- Logo: web only, no dataset extraction fallback ---
        logo_image = None
        print("  Fetching brand logo from web...")
        try:
            logo_image = WebLogoFetcher.fetch(brand)
        except Exception as e:
            print(f"    Web logo fetch failed: {type(e).__name__}")
        if logo_image:
            print(f"  Web logo: {logo_image.size}, mode={logo_image.mode}")
        else:
            print("  Web logo not found, skipping logo placement.")

        # --- RAG Content Extraction ---
        tagline = ""
        features = []
        diffusion_prompt = ""
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
        gemini_used = False

        # Priority 1: Gemini Vision RAG - analyze actual dataset ad images
        if self.llm and self.llm.api_available and ad_image_paths:
            print("  [RAG] Gemini Vision: analyzing dataset ad images...")
            try:
                ad_content = self.llm.extract_full_ad_content(
                    ad_image_paths, brand, category, subcategory
                )
                if ad_content.get("tagline"):
                    tagline = ad_content["tagline"]
                    print(f"    Tagline: {tagline}")
                if ad_content.get("features"):
                    features = [
                        f for f in ad_content["features"] if f and len(f) > 2
                    ][:6]
                    print(f"    Features: {features[:3]}...")
                if ad_content.get("diffusion_prompt"):
                    diffusion_prompt = ad_content["diffusion_prompt"]
                    print(f"    Prompt: {diffusion_prompt[:80]}...")
                if ad_content.get("negative_prompt"):
                    negative_prompt = ad_content["negative_prompt"]
                gemini_used = True
                print("    Gemini Vision RAG successful!")
            except Exception as e:
                print(f"    Gemini Vision failed: {type(e).__name__}: {e}")

        # Priority 2: CLIP Descriptor RAG - rank descriptions against dataset images
        if not gemini_used and ad_image_paths:
            print("  [RAG] CLIP: ranking content against dataset images...")
            if not features:
                features = self.clip_content_extractor.extract_features(
                    ad_image_paths, n=6
                )
            if not diffusion_prompt:
                diffusion_prompt = self.clip_content_extractor.generate_diffusion_prompt(
                    ad_image_paths, brand, category, subcategory
                )

        # Tagline: brand data fallback (factual, not hardcoded category text)
        if not tagline:
            tagline = BRAND_TAGLINES.get(brand, "")

        # Minimal final fallbacks (brand-specific, no category dicts)
        if not tagline:
            brand_display = brand.replace("_", " ")
            tagline = f"Experience {brand_display}"

        if not features:
            features = [
                "Premium Quality", "Trusted Brand", "Excellent Value",
                "Customer Favorite", "Wide Availability", "Best in Class",
            ]

        if not diffusion_prompt:
            brand_clean = brand.replace("_", " ")
            sub_clean = (
                subcategory.replace("_", " ")
                if subcategory
                else category.replace("_", " ")
            )
            diffusion_prompt = (
                f"professional advertisement photography of {brand_clean} "
                f"{sub_clean}, person using the product, premium lighting, "
                f"high quality 4k, commercial advertisement"
            )

        # Load thumbnails from dataset
        thumbnails = []
        thumb_paths = []
        for ad in retrieved_ads[:3]:
            if Path(ad.image_path).exists():
                try:
                    thumb = Image.open(ad.image_path).convert("RGB")
                    thumbnails.append(thumb)
                    thumb_paths.append(ad.image_path)
                except Exception:
                    pass

        source = "Gemini Vision RAG" if gemini_used else "CLIP Descriptor RAG"
        print(f"\n  Brand: {brand}")
        print(f"  Tagline: {tagline}")
        print(f"  Features ({len(features)}): {features[:3]}...")
        print(f"  Accent color: {accent}")
        print(f"  Logo: {'web' if logo_image else 'none'}")
        print(f"  Thumbnails: {len(thumbnails)}")
        print(f"  Diffusion prompt: {diffusion_prompt[:80]}...")
        print(f"  Content source: {source}")

        pamphlet_content = PamphletContent(
            brand_name=brand,
            tagline=tagline,
            features=features[:6],
            accent_color=accent,
            secondary_color=secondary,
            background_tint=bg_tint,
            product_image=None,
            logo_image=logo_image,
            thumbnail_images=thumbnails,
            thumbnail_paths=thumb_paths,
            diffusion_prompt=diffusion_prompt,
            negative_prompt=negative_prompt,
            category=category,
            subcategory=subcategory,
        )
        return pamphlet_content, gemini_used

    # ------------------------------------------------------------------
    # Stage 4: Image Generation
    # ------------------------------------------------------------------
    def stage4_generate_image(
        self, content: PamphletContent, colors: List[str]
    ) -> Tuple[Optional[Image.Image], str]:
        print(f"\n{'=' * 80}")
        print("STAGE 4: IMAGE GENERATION")
        print(f"{'=' * 80}")

        prompt = content.diffusion_prompt
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Negative: {content.negative_prompt[:80]}...")

        img, method = self.image_gen.generate(
            prompt=prompt,
            negative_prompt=content.negative_prompt,
            width=1024,
            height=768,
        )

        # Use brand colors for gradient fallback
        if method == "gradient_fallback" and colors:
            img = self.image_gen._make_gradient(1024, 768, colors)

        if img:
            print(f"\n  Generated via: {method}")
            print(f"  Size: {img.size}")
        else:
            print("\n  All generation methods failed!")

        return img, method

    # ------------------------------------------------------------------
    # Stage 5: Pamphlet Composition
    # ------------------------------------------------------------------
    def stage5_compose(self, content: PamphletContent) -> Image.Image:
        print(f"\n{'=' * 80}")
        print("STAGE 5: PAMPHLET COMPOSITION")
        print(f"{'=' * 80}")

        pamphlet = self.pamphlet_composer.compose(content)
        print(f"  Pamphlet size: {pamphlet.size}")
        print(f"  Layout: full-bleed product image with all text overlaid")
        return pamphlet

    # ------------------------------------------------------------------
    # Logo Overlay on Product Image
    # ------------------------------------------------------------------
    @staticmethod
    def _overlay_logo(
        product_img: Image.Image, logo: Image.Image
    ) -> Image.Image:
        """Composite brand logo onto the generated product image."""
        product_img = product_img.convert("RGBA")
        logo = logo.copy().convert("RGBA")

        # Resize logo to ~14% of image width
        target_w = max(60, int(product_img.width * 0.14))
        scale = target_w / logo.width
        new_h = max(20, int(logo.height * scale))
        logo = logo.resize((target_w, new_h), Image.LANCZOS)

        # Position: bottom-right with padding
        pad = 20
        x = product_img.width - target_w - pad
        y = product_img.height - new_h - pad

        # Semi-transparent white backing for visibility
        backing = Image.new(
            "RGBA", (target_w + 16, new_h + 16), (255, 255, 255, 160)
        )
        bx, by = x - 8, y - 8
        # Clamp to image bounds
        bx = max(0, bx)
        by = max(0, by)
        bw = min(backing.width, product_img.width - bx)
        bh = min(backing.height, product_img.height - by)
        backing = backing.crop((0, 0, bw, bh))

        product_img.paste(
            Image.alpha_composite(
                product_img.crop(
                    (bx, by, bx + bw, by + bh)
                ).convert("RGBA"),
                backing,
            ),
            (bx, by),
        )

        # Paste logo with transparency
        product_img.paste(logo, (x, y), logo)
        return product_img.convert("RGB")

    # ------------------------------------------------------------------
    # Run Full Pipeline
    # ------------------------------------------------------------------
    def run(self, query: str, k: int = 5) -> PamphletResult:
        print(f"\n{'=' * 80}")
        print(f"QUERY: {query}")
        print(f"{'=' * 80}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = PamphletResult(
            query=query,
            timestamp=datetime.now().isoformat(),
        )
        total_start = time.time()

        # Stage 0: Brand matching
        t0 = time.time()
        brand_match = self.stage0_match_brand(query)
        result.stage_timings["brand_matching"] = time.time() - t0
        if brand_match.matched_brand:
            result.brand_match = {
                "matched_brand": brand_match.matched_brand,
                "query_token": brand_match.query_token,
                "confidence": brand_match.confidence,
                "category": brand_match.category,
                "subcategory": brand_match.subcategory,
                "image_count": brand_match.image_count,
            }

        # Stage 1: Retrieval
        t0 = time.time()
        retrieved_ads = self.stage1_retrieve(query, brand_match, k=k)
        result.stage_timings["retrieval"] = time.time() - t0
        result.retrieved_ads = [
            {
                "rank": ad.rank,
                "similarity": ad.similarity,
                "brand": ad.brand,
                "category": ad.category,
                "subcategory": ad.subcategory,
                "image_path": ad.image_path,
            }
            for ad in retrieved_ads
        ]

        if not retrieved_ads:
            result.errors.append("No ads retrieved")
            return result

        # Stage 2: Color extraction
        t0 = time.time()
        colors = self.stage2_extract_colors(retrieved_ads)
        result.stage_timings["color_extraction"] = time.time() - t0
        result.extracted_colors = colors

        # Stage 3: Content generation
        t0 = time.time()
        content, gemini_used = self.stage3_generate_content(
            query, brand_match, retrieved_ads, colors
        )
        result.stage_timings["content_generation"] = time.time() - t0
        result.gemini_used = gemini_used
        result.content = {
            "brand": content.brand_name,
            "tagline": content.tagline,
            "features": content.features,
            "accent_color": content.accent_color,
            "logo_extracted": content.logo_image is not None,
            "diffusion_prompt": content.diffusion_prompt,
        }

        # Stage 4: Image generation
        t0 = time.time()
        product_img, method = self.stage4_generate_image(content, colors)
        result.stage_timings["image_generation"] = time.time() - t0
        result.image_generator_used = method

        if product_img:
            # Logo is placed by PamphletComposer in AI-detected free space only
            content.product_image = product_img
            prod_path = self.output_dir / f"product_{timestamp}.png"
            product_img.save(prod_path)
            result.product_image_path = str(prod_path)
            print(f"  Saved product image: {prod_path.name}")

        # Stage 5: Pamphlet composition
        t0 = time.time()
        pamphlet = self.stage5_compose(content)
        result.stage_timings["composition"] = time.time() - t0

        pamphlet_path = self.output_dir / f"pamphlet_{timestamp}.png"
        pamphlet.save(pamphlet_path, quality=95)
        result.pamphlet_path = str(pamphlet_path)

        total_elapsed = time.time() - total_start
        result.stage_timings["total"] = total_elapsed

        # Save JSON result
        json_path = self.output_dir / f"pamphlet_result_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # Summary
        print(f"\n{'=' * 80}")
        print("PIPELINE COMPLETE")
        print(f"{'=' * 80}")
        print(f"\n  Timings:")
        for stage, elapsed in result.stage_timings.items():
            print(f"    {stage:20s}: {elapsed:6.1f}s")
        print(f"\n  Brand: {content.brand_name}")
        print(f"  Tagline: {content.tagline}")
        print(f"  Logo extracted: {content.logo_image is not None}")
        print(f"  Gemini used: {gemini_used}")
        print(f"  Image generator: {method}")
        print(f"\n  Outputs:")
        if result.product_image_path:
            print(f"    Product image: {result.product_image_path}")
        print(f"    Pamphlet:      {result.pamphlet_path}")
        print(f"    JSON result:   {json_path}")
        if result.errors:
            print(f"\n  Errors:")
            for err in result.errors:
                print(f"    - {err}")
        print(f"{'=' * 80}")

        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    pipeline = MAdVersePamphletPipeline()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("\nEnter your ad query (e.g. 'Nike running shoes advertisement'):")
        query = input("> ").strip()
        if not query:
            print("No query provided. Exiting.")
            sys.exit(0)

    result = pipeline.run(query, k=5)

    print(f"\nPamphlet: {result.pamphlet_path}")
    if result.errors:
        print(f"Errors: {result.errors}")
