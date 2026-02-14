"""
AdCraft AI Pipeline - RAG-Based Ad Generation (Fully Local)

Uses:
  - CLIP (local) for embeddings and content extraction
  - FAISS (local) for vector search
  - LocalAdImageGenerator (PIL + numpy) for category-specific ad backgrounds
  - Smart templates for marketing copy generation
  - deep_translator for multi-language support
  - PIL for professional ad composition and enhancement
"""

import difflib
import faiss
import io
import json
import math
import numpy as np
import os
import pickle
import random
import re
import sqlite3
import time
import traceback
import urllib.parse
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from sklearn.cluster import KMeans
from transformers import CLIPModel, CLIPProcessor

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
FAISS_DIR = EMBEDDINGS_DIR / "faiss_indexes"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_DIR = BASE_DIR / "products_db"

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ar": "Arabic",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-CN": "Chinese (Simplified)",
    "ru": "Russian",
    "it": "Italian",
    "nl": "Dutch",
    "tr": "Turkish",
}

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
class AdContent:
    brand_name: str = ""
    headline: str = ""
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
    cta_text: str = ""
    brand_domain: str = ""
    theme_name: str = ""
    # Bilingual support: secondary language text (usually English when primary is non-English)
    headline_secondary: str = ""
    features_secondary: List[str] = field(default_factory=list)
    cta_secondary: str = ""


PamphletContent = AdContent  # backward compatibility


@dataclass
class GenerationResult:
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
    errors: List[str] = field(default_factory=list)
    # New fields for AdCraft
    product_title: str = ""
    product_description: str = ""
    instagram_caption: str = ""
    whatsapp_copy: str = ""
    hashtags: List[str] = field(default_factory=list)
    translations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    product_id: Optional[str] = None
    languages_generated: List[str] = field(default_factory=list)


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

DEFAULT_NEGATIVE_PROMPT = (
    "text, words, letters, numbers, typography, watermark, logo, label, stamp, "
    "blurry, low quality, pixelated, distorted, deformed, ugly, oversaturated, "
    "extra fingers, extra limbs, disfigured face, bad anatomy, bad proportions, "
    "poorly drawn hands, poorly drawn face, mutation, amateur, clip art, cartoon, "
    "frame, border, collage, split image, multiple views"
)


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


# ---------------------------------------------------------------------------
# Color Extractor
# ---------------------------------------------------------------------------

class ColorExtractor:
    RESIZE_DIM = 100

    def extract_from_images(self, image_paths: List[str], n_colors: int = 5) -> List[str]:
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
        best = hex_colors[0] if hex_colors else "#1a1a2e"
        best_score = -1
        for h in hex_colors:
            r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            sat = (max_c - min_c) / max_c if max_c > 0 else 0
            brightness = (r + g + b) / 3
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
# Web Logo Fetcher (no API keys - uses free services)
# ---------------------------------------------------------------------------

class ProLogoFetcher:
    """Multi-source logo fetcher with working free APIs."""

    BRAND_DOMAINS = {
        "Nike": "nike.com", "Adidas": "adidas.com", "Puma": "puma.com",
        "Reebok": "reebok.com", "Skechers": "skechers.com", "FILA": "fila.com",
        "Bata": "bata.in", "HRX": "hrx.in", "Bisleri": "bisleri.com",
        "Coca_Cola": "coca-cola.com",
        "Pepsi": "pepsi.com", "Amul": "amul.com",
        "Patanjali": "patanjaliayurved.net", "Samsung_mobiles": "samsung.com",
        "Samsung": "samsung.com", "Apple_mobiles": "apple.com", "Apple": "apple.com",
        "Motorola_mobiles": "motorola.com", "One-Plus_mobiles": "oneplus.com",
        "OnePlus": "oneplus.com", "maruti_suzuki": "marutisuzuki.com",
        "tata": "tata.com", "mahindra": "mahindra.com", "hyundai": "hyundai.co.in",
        "honda": "honda.com", "toyota": "toyota.com", "tvs": "tvsmotor.com",
        "hero_motocorp": "heromotocorp.com", "bajaj": "bajajauto.com",
        "ICICI_Bank": "icicibank.com", "State_Bank_Of_India": "sbi.co.in",
        "LIC": "licindia.in", "Tanishq_jewellary": "tanishq.co.in",
        "Tanishq": "tanishq.co.in", "Titan_watches": "titan.co.in",
        "Titan": "titan.co.in", "Allen_Solly": "allensolly.com",
        "Louis_Philippe": "louisphilippe.com", "Peter_England": "peterengland.com",
        "Raymonds": "raymond.in", "Lux_soaps": "lux.com",
        "Pantene_shampoo": "pantene.com", "Air_India": "airindia.com",
        "Make_MyTrip": "makemytrip.com", "Zara": "zara.com",
        "H&M": "hm.com", "Rolex": "rolex.com", "BMW": "bmw.com",
        "Mercedes": "mercedes-benz.com", "Audi": "audi.com",
    }

    _HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    @classmethod
    def fetch(cls, brand_name: str) -> Optional[Image.Image]:
        domain = cls._get_domain(brand_name)

        # Source 1: Scrape website for apple-touch-icon (best: up to 512px)
        logo = cls._try_website_icon(domain)
        if logo and min(logo.size) >= 64:
            cleaned = cls._remove_background(logo)
            if cleaned:
                return cls._polish_logo(cleaned)

        # Source 2: Google faviconV2 (reliable, up to 256px)
        logo = cls._try_google_favicon(domain)
        if logo and min(logo.size) >= 48:
            cleaned = cls._remove_background(logo)
            if cleaned:
                return cls._polish_logo(cleaned)

        # Source 3: icon.horse (good backup)
        logo = cls._try_icon_horse(domain)
        if logo and min(logo.size) >= 48:
            cleaned = cls._remove_background(logo)
            if cleaned:
                return cls._polish_logo(cleaned)

        # Source 4: DuckDuckGo icon
        logo = cls._try_duckduckgo_icon(domain)
        if logo and min(logo.size) >= 48:
            cleaned = cls._remove_background(logo)
            if cleaned:
                return cls._polish_logo(cleaned)

        # Final fallback: text-based logo
        return cls._generate_text_logo(brand_name)

    @classmethod
    def _get_domain(cls, brand_name: str) -> str:
        # Exact match
        if brand_name in cls.BRAND_DOMAINS:
            return cls.BRAND_DOMAINS[brand_name]
        # Case-insensitive match
        for key, val in cls.BRAND_DOMAINS.items():
            if key.lower().replace("_", "") == brand_name.lower().replace("_", "").replace(" ", ""):
                return val
        # Heuristic: strip common suffixes
        clean = brand_name.lower().replace("_", "").replace(" ", "")
        for suffix in ("mobiles", "watches", "jewellary", "jewellery", "soaps",
                        "shampoo", "baby_products", "footwear", "bank"):
            if clean.endswith(suffix):
                clean = clean[:-len(suffix)]
                break
        return f"{clean}.com"

    @classmethod
    def _try_website_icon(cls, domain: str) -> Optional[Image.Image]:
        """Scrape the actual website for apple-touch-icon (often 180-512px)."""
        try:
            resp = requests.get(f"https://{domain}", timeout=10,
                                headers=cls._HEADERS, allow_redirects=True)
            if resp.status_code >= 400:
                resp = requests.get(f"https://www.{domain}", timeout=10,
                                    headers=cls._HEADERS, allow_redirects=True)
            if resp.status_code >= 400:
                return None
            html = resp.text
            base_url = resp.url

            # Find apple-touch-icon URLs (usually largest icons)
            icon_urls = []
            for pattern in [
                r'<link[^>]+rel=["\']apple-touch-icon["\'][^>]*href=["\']([^"\']*)["\']',
                r'<link[^>]+href=["\']([^"\']*)["\'][^>]*rel=["\']apple-touch-icon',
            ]:
                icon_urls.extend(re.findall(pattern, html, re.IGNORECASE))

            # Try each found icon, pick the largest
            best_img = None
            best_size = 0
            for icon_url in icon_urls[:6]:
                try:
                    from urllib.parse import urljoin
                    full_url = urljoin(base_url, icon_url)
                    ir = requests.get(full_url, timeout=8, headers=cls._HEADERS)
                    if ir.status_code == 200 and len(ir.content) > 200:
                        img = Image.open(io.BytesIO(ir.content)).convert("RGBA")
                        if min(img.size) > best_size:
                            best_img = img
                            best_size = min(img.size)
                except Exception:
                    continue
            return best_img
        except Exception:
            return None

    @classmethod
    def _try_google_favicon(cls, domain: str) -> Optional[Image.Image]:
        """Google faviconV2 API — most reliable, up to 256px."""
        try:
            url = (f"https://t1.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON"
                   f"&fallback_opts=TYPE,SIZE,URL&url=https://{domain}&size=256")
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                img = Image.open(io.BytesIO(resp.content))
                if img.size[0] >= 32:
                    return img.convert("RGBA")
        except Exception:
            pass
        return None

    @classmethod
    def _try_icon_horse(cls, domain: str) -> Optional[Image.Image]:
        """icon.horse — free, good quality backup."""
        try:
            url = f"https://icon.horse/icon/{domain}"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200 and len(resp.content) > 500:
                img = Image.open(io.BytesIO(resp.content))
                if img.size[0] >= 32:
                    return img.convert("RGBA")
        except Exception:
            pass
        return None

    @classmethod
    def _try_duckduckgo_icon(cls, domain: str) -> Optional[Image.Image]:
        try:
            url = f"https://icons.duckduckgo.com/ip3/{domain}.ico"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200 and len(resp.content) > 500:
                img = Image.open(io.BytesIO(resp.content))
                if img.size[0] >= 32:
                    return img.convert("RGBA")
        except Exception:
            pass
        return None

    @classmethod
    def _try_ai_generated_logo(cls, brand_name: str) -> Optional[Image.Image]:
        """Generate a clean brand logo using Pollinations AI."""
        try:
            clean = brand_name.replace("_", " ").strip()
            prompt = (
                f"minimalist professional brand logo for '{clean}', "
                f"clean vector style, single icon, transparent background, "
                f"simple flat design, corporate branding, "
                f"centered on white background, high resolution"
            )
            encoded = urllib.parse.quote(prompt, safe="")
            url = (
                f"https://image.pollinations.ai/prompt/{encoded}"
                f"?width=512&height=512&model=flux&nologo=true&enhance=true"
            )
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and len(resp.content) > 1000:
                img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                return img
        except Exception:
            pass
        return None

    @staticmethod
    def _remove_background(img: Image.Image, tolerance: int = 35) -> Optional[Image.Image]:
        """Smart background removal using border pixel analysis."""
        img = img.convert("RGBA")
        arr = np.array(img)
        h, w = arr.shape[:2]
        if h < 10 or w < 10:
            return img

        # Already has good transparency?
        if arr.shape[2] >= 4:
            transparent_ratio = (arr[:, :, 3] < 128).sum() / (h * w)
            if transparent_ratio > 0.05:
                return img

        # Sample border pixels (all 4 edges, multiple pixels deep)
        border_depth = max(3, min(h, w) // 12)
        border_pixels: List[tuple] = []
        for y in range(border_depth):
            for x in range(w):
                border_pixels.append(tuple(arr[y, x, :3].tolist()))
                border_pixels.append(tuple(arr[h - 1 - y, x, :3].tolist()))
        for x in range(border_depth):
            for y in range(border_depth, h - border_depth):
                border_pixels.append(tuple(arr[y, x, :3].tolist()))
                border_pixels.append(tuple(arr[y, w - 1 - x, :3].tolist()))

        bg_color = np.array(Counter(border_pixels).most_common(1)[0][0])

        # Create distance-based mask
        diff = np.sqrt(np.sum((arr[:, :, :3].astype(float) - bg_color.astype(float)) ** 2, axis=2))
        bg_mask = diff < tolerance

        result = arr.copy()
        result[bg_mask, 3] = 0

        # Feather edges for smooth anti-aliased border
        alpha_channel = Image.fromarray(result[:, :, 3])
        alpha_feathered = alpha_channel.filter(ImageFilter.GaussianBlur(1.5))
        feathered_arr = np.array(alpha_feathered)
        feathered_arr[bg_mask] = 0
        result[:, :, 3] = feathered_arr

        # Validate content ratio
        content_ratio = (result[:, :, 3] > 128).sum() / (h * w)
        if content_ratio < 0.05 or content_ratio > 0.98:
            return img
        return Image.fromarray(result)

    @classmethod
    def _polish_logo(cls, logo: Image.Image) -> Image.Image:
        """Crop to content bounding box and standardize height."""
        logo = logo.convert("RGBA")
        alpha = np.array(logo)[:, :, 3]
        rows = np.any(alpha > 30, axis=1)
        cols = np.any(alpha > 30, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            pad = 4
            rmin = max(0, rmin - pad)
            rmax = min(logo.height - 1, rmax + pad)
            cmin = max(0, cmin - pad)
            cmax = min(logo.width - 1, cmax + pad)
            logo = logo.crop((cmin, rmin, cmax + 1, rmax + 1))
        # Standardize to 200px height, upscale small icons with LANCZOS
        target_h = 200
        if logo.height > 0:
            scale = target_h / logo.height
            new_w = max(1, int(logo.width * scale))
            logo = logo.resize((new_w, target_h), Image.LANCZOS)
        return logo

    @classmethod
    def _generate_text_logo(cls, brand_name: str) -> Image.Image:
        """Generate a clean text-based logo as fallback."""
        import colorsys
        clean = brand_name.replace("_", " ").strip()
        initials = "".join(word[0].upper() for word in clean.split()[:2]) or "?"

        size = 200
        render_size = size * 4
        canvas = Image.new("RGBA", (render_size, render_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        hue = sum(ord(c) for c in brand_name) % 360
        r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.65, 0.85)
        color = (int(r * 255), int(g * 255), int(b * 255))

        margin = render_size // 8
        draw.rounded_rectangle(
            [margin, margin, render_size - margin, render_size - margin],
            radius=render_size // 4, fill=(*color, 240)
        )

        try:
            font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", render_size // 3)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), initials, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (render_size - tw) // 2
        ty = (render_size - th) // 2 - bbox[1]
        draw.text((tx, ty), initials, fill=(255, 255, 255, 255), font=font)

        return canvas.resize((size, size), Image.LANCZOS)


WebLogoFetcher = ProLogoFetcher  # backward compatibility


# ---------------------------------------------------------------------------
# CLIP Content Extractor (RAG-based, fully local)
# ---------------------------------------------------------------------------

class CLIPContentExtractor:
    FEATURE_POOL = [
        "Premium quality craftsmanship", "Trusted by millions worldwide",
        "Award-winning innovative design", "Unbeatable value for money",
        "Industry-leading performance", "Eco-friendly sustainable choice",
        "Lightweight comfortable fit", "Breathable mesh upper design",
        "Advanced cushioning technology", "Durable long-lasting build",
        "Moisture-wicking performance fabric", "Athletic performance engineered",
        "Trendy modern street style", "All-day support and comfort",
        "Pure refreshing hydration", "Mineral-enriched natural water",
        "Natural fruit ingredients", "Zero artificial preservatives",
        "Clinically tested purity", "Refreshing clean taste",
        "Cutting-edge processor power", "Stunning high-resolution display",
        "All-day battery performance", "Professional camera system",
        "5G ultra-fast connectivity", "Sleek premium metal build",
        "Dermatologically tested formula", "Long-lasting natural radiance",
        "Gentle on all skin types", "Clinically proven visible results",
        "Nourishing botanical ingredients", "Salon-quality results at home",
        "Made with natural ingredients", "Irresistible authentic flavor",
        "Fresh quality guaranteed daily", "Nutrition-packed healthy goodness",
        "Loved by food enthusiasts",
        "Powerful engine performance", "Advanced safety technology",
        "Fuel-efficient smart engineering", "Spacious luxury interior cabin",
        "Exquisite handcrafted artistry", "Certified precious fine materials",
        "Timeless elegant classic design", "Precision Swiss-quality movement",
        "Secure trusted digital banking", "Digital-first modern convenience",
        "Best price guaranteed always", "Seamless easy booking experience",
    ]

    # Category-specific features to avoid cross-category contamination
    CATEGORY_FEATURES = {
        "jewellery": [
            "Exquisite handcrafted artistry", "Certified precious fine materials",
            "Timeless elegant classic design", "Premium quality craftsmanship",
            "Trusted by millions worldwide", "Award-winning innovative design",
        ],
        "watches": [
            "Precision Swiss-quality movement", "Timeless elegant classic design",
            "Premium quality craftsmanship", "Sleek premium metal build",
            "Award-winning innovative design", "Trusted by millions worldwide",
        ],
        "footwear": [
            "Lightweight comfortable fit", "Breathable mesh upper design",
            "Advanced cushioning technology", "Durable long-lasting build",
            "Athletic performance engineered", "Trendy modern street style",
        ],
        "clothing_brands": [
            "Premium quality craftsmanship", "Trendy modern street style",
            "Lightweight comfortable fit", "Award-winning innovative design",
            "All-day support and comfort", "Trusted by millions worldwide",
        ],
        "mobile_devices": [
            "Cutting-edge processor power", "Stunning high-resolution display",
            "All-day battery performance", "Professional camera system",
            "5G ultra-fast connectivity", "Sleek premium metal build",
        ],
        "computers": [
            "Cutting-edge processor power", "Stunning high-resolution display",
            "All-day battery performance", "Industry-leading performance",
            "Sleek premium metal build", "Award-winning innovative design",
        ],
        "four_wheelers": [
            "Powerful engine performance", "Advanced safety technology",
            "Fuel-efficient smart engineering", "Spacious luxury interior cabin",
            "Industry-leading performance", "Award-winning innovative design",
        ],
        "two_wheelers": [
            "Powerful engine performance", "Fuel-efficient smart engineering",
            "Durable long-lasting build", "Industry-leading performance",
            "Award-winning innovative design", "Trendy modern street style",
        ],
        "water": [
            "Pure refreshing hydration", "Mineral-enriched natural water",
            "Clinically tested purity", "Refreshing clean taste",
            "Eco-friendly sustainable choice", "Trusted by millions worldwide",
        ],
        "soft_drinks": [
            "Refreshing clean taste", "Natural fruit ingredients",
            "Irresistible authentic flavor", "Zero artificial preservatives",
            "Loved by food enthusiasts", "Trusted by millions worldwide",
        ],
        "juice": [
            "Natural fruit ingredients", "Zero artificial preservatives",
            "Refreshing clean taste", "Nutrition-packed healthy goodness",
            "Fresh quality guaranteed daily", "Trusted by millions worldwide",
        ],
        "skincare_and_makeup": [
            "Dermatologically tested formula", "Long-lasting natural radiance",
            "Gentle on all skin types", "Clinically proven visible results",
            "Nourishing botanical ingredients", "Salon-quality results at home",
        ],
        "banks": [
            "Secure trusted digital banking", "Digital-first modern convenience",
            "Trusted by millions worldwide", "Industry-leading performance",
            "Seamless easy booking experience", "Award-winning innovative design",
        ],
        "insurance": [
            "Secure trusted digital banking", "Trusted by millions worldwide",
            "Best price guaranteed always", "Industry-leading performance",
            "Seamless easy booking experience", "Award-winning innovative design",
        ],
    }

    STYLE_POOL = [
        "dramatic studio lighting on dark background",
        "bright outdoor natural sunlight setting",
        "warm golden hour sunset glow",
        "clean minimalist white background",
        "urban city street environment",
        "lush green nature backdrop",
        "modern gym fitness environment",
        "luxurious upscale premium interior",
        "vibrant colorful dynamic style",
        "sleek futuristic tech aesthetic",
        "elegant fashion editorial style",
        "dynamic sports action moment",
    ]

    MOOD_POOL = [
        "energetic dynamic movement", "calm peaceful serenity",
        "luxurious premium elegance", "fun playful joyful energy",
        "professional confident trust", "fresh refreshing vitality",
        "warm family togetherness", "bold powerful strength",
        "sophisticated refined taste", "adventurous exciting freedom",
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
        "fitness enthusiast working out intensely",
        "traveler at beautiful scenic destination",
        "person savoring delicious food",
    ]

    def __init__(self, clip_model, clip_processor, device: str):
        self.model = clip_model
        self.processor = clip_processor
        self.device = device

    @staticmethod
    def _to_tensor(features):
        """Convert model output to tensor, handling different transformers versions."""
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, 'pooler_output'):
            return features.pooler_output
        if hasattr(features, 'last_hidden_state'):
            return features.last_hidden_state[:, 0, :]
        return features[0] if hasattr(features, '__getitem__') else features

    def _encode_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in inputs.items()
                       if k in ('input_ids', 'attention_mask')}
        with torch.no_grad():
            text_out = self.model.text_model(**text_inputs)
            pooled = text_out[1]  # pooler_output
            features = self.model.text_projection(pooled)
            return F.normalize(features, p=2, dim=-1)

    def _encode_images(self, image_paths):
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
            vision_out = self.model.vision_model(pixel_values=pixel_values)
            pooled = vision_out[1]  # pooler_output
            features = self.model.visual_projection(pooled)
            return F.normalize(features, p=2, dim=-1)

    def _rank_pool(self, image_features, pool, top_n):
        if image_features.sum() == 0:
            return pool[:top_n]
        avg_feat = F.normalize(image_features.mean(dim=0, keepdim=True), p=2, dim=-1)
        text_feat = self._encode_texts(pool)
        similarities = (avg_feat @ text_feat.T).squeeze(0)
        top_indices = similarities.argsort(descending=True)[:top_n].tolist()
        return [pool[i] for i in top_indices]

    def extract_features(self, image_paths, n=6, category=""):
        """Extract features, preferring category-specific pool when available."""
        if category and category in self.CATEGORY_FEATURES:
            return self.CATEGORY_FEATURES[category][:n]
        img_feats = self._encode_images(image_paths)
        return self._rank_pool(img_feats, self.FEATURE_POOL, n)

    # Rich category-specific prompt enrichments for photorealistic ad images
    CATEGORY_PROMPT_CONTEXT = {
        "jewellery": (
            "beautiful Indian woman wearing ornate gold jewelry, traditional silk saree, "
            "intricate gold necklace and bangles, luxury jewelry showroom background, "
            "warm golden lighting, rich cultural Indian aesthetic, elegant bridal style"
        ),
        "watches": (
            "luxury timepiece on elegant wrist, polished metal watch face close-up, "
            "rich leather strap detail, premium lifestyle setting, studio lighting, "
            "reflective surfaces, sophisticated gentleman or lady"
        ),
        "footwear": (
            "stylish shoes on model, dynamic urban setting, clean product showcase, "
            "athletic lifestyle, trendy street style, vibrant colors, action pose"
        ),
        "clothing_brands": (
            "fashion model wearing designer outfit, studio fashion photography, "
            "clean backdrop, professional styling, elegant pose, runway quality, "
            "fabric texture detail, seasonal collection feel"
        ),
        "mobile_devices": (
            "sleek smartphone in hand, futuristic tech aesthetic, glowing screen, "
            "modern minimalist setting, premium device showcase, reflections, "
            "cutting-edge technology feel"
        ),
        "computers": (
            "premium laptop or computer on modern desk, clean workspace, "
            "tech-forward aesthetic, screen glow, professional environment, "
            "productivity lifestyle, sharp angles"
        ),
        "four_wheelers": (
            "luxury car on scenic road, dramatic automotive photography, "
            "gleaming paint finish, dynamic angle, motion blur background, "
            "premium interior detail, powerful stance"
        ),
        "two_wheelers": (
            "motorcycle on open road, rider in gear, dynamic speed shot, "
            "powerful engine detail, dramatic lighting, adventure spirit"
        ),
        "water": (
            "crystal clear water splash, refreshing water bottle, pure droplets, "
            "clean blue tones, hydration lifestyle, nature background, "
            "mountain spring freshness"
        ),
        "soft_drinks": (
            "ice-cold beverage with condensation droplets, refreshing pour shot, "
            "vibrant colors, summer vibes, social gathering, energetic mood"
        ),
        "juice": (
            "fresh fruit juice with natural fruits around, colorful healthy drink, "
            "tropical fruits, organic feel, splash photography, vibrant natural colors"
        ),
        "skincare_and_makeup": (
            "beautiful woman with flawless radiant skin, beauty product close-up, "
            "soft natural lighting, clean beauty aesthetic, dewy skin glow, "
            "luxury cosmetics, spa-like serenity"
        ),
        "banks": (
            "modern banking lifestyle, professional setting, digital finance, "
            "trust and security feel, clean corporate aesthetic, happy customer"
        ),
        "insurance": (
            "happy family protected together, warm home setting, security and trust, "
            "life insurance peace of mind, caring professional atmosphere"
        ),
    }

    def _extract_query_subject(self, user_query, brand_clean):
        """Extract meaningful product description from user query."""
        if not user_query:
            return ""
        query_desc = user_query.lower()
        for word in brand_clean.lower().split():
            query_desc = query_desc.replace(word, "")
        filler = {"ad", "ads", "advertisement", "pamphlet", "poster", "banner",
                  "create", "make", "generate", "please", "want", "need", "i",
                  "a", "an", "the", "for", "with", "and", "of", "in", "on",
                  "me", "my", "give", "show", "by", "it", "that", "this"}
        words = [w for w in query_desc.split() if w not in filler]
        return " ".join(words).strip()

    def generate_diffusion_prompt(self, image_paths, brand, category, subcategory, user_query=""):
        """Generate a rich, detailed image prompt for photorealistic ad images.
        Uses user query + category context to produce highly specific prompts."""
        img_feats = self._encode_images(image_paths)
        top_styles = self._rank_pool(img_feats, self.STYLE_POOL, 1)
        top_moods = self._rank_pool(img_feats, self.MOOD_POOL, 1)

        brand_clean = brand.replace("_", " ")
        query_subject = self._extract_query_subject(user_query, brand_clean)

        # Get category-specific rich context
        cat_context = self.CATEGORY_PROMPT_CONTEXT.get(
            category, self.CATEGORY_PROMPT_CONTEXT.get(subcategory, "")
        )

        # Build a detailed prompt combining user intent + category knowledge
        parts = [f"professional commercial advertisement photography for {brand_clean}"]

        if query_subject:
            parts.append(query_subject)

        if cat_context:
            parts.append(cat_context)

        parts.append(f"{top_styles[0]}, {top_moods[0]} mood")
        parts.append("ultra realistic, photorealistic, 8k, sharp focus, professional lighting, commercial quality")

        return ", ".join(parts)

    def analyze_uploaded_image(self, image: Image.Image, category: str = ""):
        """Analyze an uploaded product image using CLIP to extract descriptors."""
        # If category is known, use category-specific features directly
        if category and category in self.CATEGORY_FEATURES:
            return {
                "features": self.CATEGORY_FEATURES[category][:6],
                "styles": self.STYLE_POOL[:2],
                "mood": "professional",
            }

        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=pixel_values)
            pooled = vision_out[1]
            img_feats = self.model.visual_projection(pooled)
            img_feats = F.normalize(img_feats, p=2, dim=-1)

        features = self._rank_pool(img_feats, self.FEATURE_POOL, 6)
        styles = self._rank_pool(img_feats, self.STYLE_POOL, 2)
        moods = self._rank_pool(img_feats, self.MOOD_POOL, 1)

        return {
            "features": features,
            "styles": styles,
            "mood": moods[0] if moods else "professional",
        }


# ---------------------------------------------------------------------------
# Image Generator (Pollinations → HF Inference → gradient — same as RAGPipeline.py)
# ---------------------------------------------------------------------------

class ImageGenerator:
    """Image generator — exact same as RAGPipeline.py.
    Tries: Pollinations FLUX → HuggingFace Inference (FLUX.1-schnell) → gradient fallback."""

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
        """Try all sources in order — matches RAGPipeline.py exactly."""
        img = self._try_pollinations(prompt, width, height)
        if img:
            return img, "pollinations_flux"
        img = self._try_hf_inference(prompt)
        if img:
            return img, "hf_inference_flux"
        img = self._make_gradient(width, height)
        return img, "gradient_fallback"

    def _try_pollinations(self, prompt, width, height) -> Optional[Image.Image]:
        print("\n    Trying Pollinations.ai (FLUX)...")
        encoded = urllib.parse.quote(prompt, safe="")
        seed = random.randint(1, 999999)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={width}&height={height}&model=flux&nologo=true&seed={seed}"
        )
        for attempt in range(self.POLLINATIONS_RETRIES):
            try:
                label = f" (attempt {attempt + 1})" if attempt > 0 else ""
                print(f"      Requesting image{label} (may take 30-90s)...")
                resp = requests.get(url, timeout=self.POLLINATIONS_TIMEOUT)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    print(f"      Success! Image size: {img.size}")
                    return img
                elif resp.status_code >= 500:
                    print(f"      Server error {resp.status_code}, retrying...")
                    time.sleep(5)
                else:
                    print(f"      Failed: status={resp.status_code}")
                    return None
            except requests.Timeout:
                print(f"      Timeout after {self.POLLINATIONS_TIMEOUT}s")
            except Exception as e:
                print(f"      Error: {e}")
        return None

    def _try_hf_inference(self, prompt) -> Optional[Image.Image]:
        if not self.hf_token:
            print("\n    Skipping HF Inference (no token)")
            return None
        print("\n    Trying HuggingFace Inference API (FLUX.1-schnell)...")
        try:
            seed = random.randint(1, 999999)
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            resp = requests.post(
                self.HF_INFERENCE_URL, headers=headers,
                json={"inputs": prompt, "parameters": {"seed": seed}},
                timeout=120,
            )
            if resp.status_code == 200 and len(resp.content) > 1000:
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                print(f"      Success! Image size: {img.size}")
                return img
            else:
                print(f"      Failed: status={resp.status_code}")
                if resp.status_code != 200:
                    print(f"      Response: {resp.text[:200]}")
        except Exception as e:
            print(f"      Error: {e}")
        return None

    def _make_gradient(self, width=1024, height=768, colors=None) -> Image.Image:
        print("\n    Creating gradient fallback background...")
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


# Keep backward-compatible name
PollinationsImageGenerator = ImageGenerator


# ---------------------------------------------------------------------------
# Local Ad Image Generator (PIL + numpy only, no external APIs)
# ---------------------------------------------------------------------------

class LocalAdImageGenerator:
    """Generates professional ad background images locally using PIL + numpy.
    No external APIs required. Creates category-specific abstract visuals."""

    WIDTH = 1080
    HEIGHT = 1080

    CATEGORY_THEMES = {
        "water": "aqua", "soft_drinks": "aqua", "juice": "aqua",
        "footwear": "dynamic", "clothing_brands": "dynamic",
        "mobile_devices": "tech", "computers": "tech",
        "jewellery": "luxury", "watches": "luxury",
        "four_wheelers": "power", "two_wheelers": "power",
        "skincare_and_makeup": "elegant",
        "banks": "corporate", "insurance": "corporate",
    }

    @staticmethod
    def _hex_to_rgb(h):
        h = h.lstrip("#")
        if len(h) < 6:
            h = h.ljust(6, "0")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def generate(self, category, subcategory, accent_hex, secondary_hex, brand_name=""):
        accent = self._hex_to_rgb(accent_hex) if isinstance(accent_hex, str) else accent_hex
        secondary = self._hex_to_rgb(secondary_hex) if isinstance(secondary_hex, str) else secondary_hex
        theme = self.CATEGORY_THEMES.get(
            subcategory, self.CATEGORY_THEMES.get(category, "modern"))

        renderers = {
            "aqua": self._render_aqua,
            "dynamic": self._render_dynamic,
            "tech": self._render_tech,
            "luxury": self._render_luxury,
            "power": self._render_power,
            "elegant": self._render_elegant,
            "corporate": self._render_corporate,
            "modern": self._render_modern,
        }
        img = renderers.get(theme, self._render_modern)(accent, secondary)
        img = ImageEnhance.Sharpness(img).enhance(1.15)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        return img

    # ── Shared drawing utilities ──

    def _linear_gradient(self, W, H, color_top, color_bottom):
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(3):
            arr[:, :, i] = np.linspace(color_top[i], color_bottom[i], H
                                        ).astype(np.uint8)[:, np.newaxis]
        return Image.fromarray(arr, "RGB")

    def _radial_gradient(self, W, H, cx, cy, radius, color, max_alpha=80):
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        falloff = np.clip(1 - dist / radius, 0, 1) ** 2
        arr = np.zeros((H, W, 4), dtype=np.float64)
        for i in range(3):
            arr[:, :, i] = color[i] * falloff
        arr[:, :, 3] = max_alpha * falloff
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")

    def _noise_layer(self, W, H, intensity=6):
        noise = np.random.randint(0, intensity, (H, W), dtype=np.uint8)
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        layer.putalpha(Image.fromarray(noise, "L"))
        return layer

    def _draw_droplets(self, canvas, count, color, size_range=(3, 20)):
        W, H = canvas.size
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        rng = random.Random(42)
        for _ in range(count):
            x, y = rng.randint(0, W), rng.randint(0, H)
            r = rng.randint(size_range[0], size_range[1])
            alpha = rng.randint(30, 90)
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=(*color[:3], alpha),
                         outline=(*color[:3], min(alpha + 30, 120)))
            if r > 5:
                hs = max(2, r // 3)
                hx, hy = x - r // 3, y - r // 3
                draw.ellipse([hx - hs, hy - hs, hx + hs, hy + hs],
                             fill=(255, 255, 255, min(alpha + 20, 80)))
        return Image.alpha_composite(canvas, layer)

    def _draw_sparkles(self, canvas, count, color):
        W, H = canvas.size
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        rng = random.Random(123)
        for _ in range(count):
            x, y = rng.randint(0, W), rng.randint(0, H)
            size = rng.randint(4, 25)
            alpha = rng.randint(60, 200)
            c = (*color[:3], alpha)
            draw.line([(x - size, y), (x + size, y)], fill=c, width=1)
            draw.line([(x, y - size), (x, y + size)], fill=c, width=1)
            ds = size // 2
            draw.line([(x - ds, y - ds), (x + ds, y + ds)], fill=c, width=1)
            draw.line([(x + ds, y - ds), (x - ds, y + ds)], fill=c, width=1)
            draw.ellipse([x - 2, y - 2, x + 2, y + 2],
                         fill=(255, 255, 255, min(alpha, 180)))
        return Image.alpha_composite(canvas, layer)

    def _draw_wave_lines(self, canvas, y_base, amplitude, color, count=3, thickness=2):
        W, H = canvas.size
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for i in range(count):
            offset_y = y_base + i * (amplitude * 2 + 15)
            alpha = max(20, 80 - i * 20)
            freq = 1.5 + i * 0.3
            points = []
            for x in range(0, W + 1, 2):
                t = x / W
                y = offset_y + int(amplitude * math.sin(t * math.pi * freq * 2))
                points.append((x, y))
            for j in range(len(points) - 1):
                draw.line([points[j], points[j + 1]],
                          fill=(*color[:3], alpha), width=thickness)
        return Image.alpha_composite(canvas, layer)

    def _draw_light_rays(self, canvas, origin, color, count=8, max_alpha=40):
        W, H = canvas.size
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        rng = random.Random(77)
        ox, oy = origin
        for _ in range(count):
            angle = rng.uniform(0, math.pi * 2)
            spread = rng.uniform(0.05, 0.15)
            length = rng.randint(300, max(W, H))
            alpha = rng.randint(10, max_alpha)
            dx1 = length * math.cos(angle - spread)
            dy1 = length * math.sin(angle - spread)
            dx2 = length * math.cos(angle + spread)
            dy2 = length * math.sin(angle + spread)
            pts = [(ox, oy), (int(ox + dx1), int(oy + dy1)),
                   (int(ox + dx2), int(oy + dy2))]
            draw.polygon(pts, fill=(*color[:3], alpha))
        layer = layer.filter(ImageFilter.GaussianBlur(3))
        return Image.alpha_composite(canvas, layer)

    def _draw_hexagon_grid(self, canvas, color, spacing=60, alpha=25):
        W, H = canvas.size
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        r = spacing // 2
        h = int(r * math.sqrt(3) / 2)
        for row in range(-1, H // max(h, 1) + 2):
            for col in range(-1, W // max(spacing, 1) + 2):
                cx = col * spacing + (spacing // 2 if row % 2 else 0)
                cy = row * h
                pts = []
                for i in range(6):
                    a = math.pi / 3 * i + math.pi / 6
                    px = cx + int(r * 0.9 * math.cos(a))
                    py = cy + int(r * 0.9 * math.sin(a))
                    pts.append((px, py))
                draw.polygon(pts, outline=(*color[:3], alpha))
        return Image.alpha_composite(canvas, layer)

    # ── AQUA: Water / Beverages ──
    def _render_aqua(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        canvas = self._linear_gradient(W, H, (0, 50, 120), (0, 120, 200)).convert("RGBA")

        # Central underwater caustic glow
        glow = self._radial_gradient(W, H, W // 2, H // 3, int(W * 0.6),
                                     (100, 200, 255), 50)
        canvas = Image.alpha_composite(canvas, glow)
        glow2 = self._radial_gradient(W, H, W // 3, int(H * 0.7),
                                      int(W * 0.4), accent, 30)
        canvas = Image.alpha_composite(canvas, glow2)

        # Wave lines
        canvas = self._draw_wave_lines(canvas, H // 4, 30, (150, 220, 255),
                                       count=4, thickness=2)
        canvas = self._draw_wave_lines(canvas, int(H * 0.65), 20,
                                       (100, 180, 240), count=3, thickness=1)

        # Light rays from top
        canvas = self._draw_light_rays(canvas, (W // 2, -50),
                                       (180, 220, 255), count=10, max_alpha=25)

        # Water droplets (small + large bubbles)
        canvas = self._draw_droplets(canvas, 60, (180, 230, 255), size_range=(2, 15))
        canvas = self._draw_droplets(canvas, 15, (200, 240, 255), size_range=(15, 35))

        # Ice cube shapes
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        rng = random.Random(99)
        for _ in range(6):
            ix, iy = rng.randint(50, W - 100), rng.randint(50, H - 100)
            iw, ih = rng.randint(40, 90), rng.randint(40, 90)
            alpha = rng.randint(15, 35)
            draw.rounded_rectangle([ix, iy, ix + iw, iy + ih], radius=5,
                                   fill=(200, 230, 255, alpha),
                                   outline=(220, 240, 255, min(alpha + 15, 60)))
            draw.rounded_rectangle([ix + 3, iy + 3, ix + iw // 3, iy + ih // 3],
                                   radius=3, fill=(255, 255, 255, min(alpha + 10, 40)))
        canvas = Image.alpha_composite(canvas, layer)

        # Frost streak across middle
        frost = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        fd = ImageDraw.Draw(frost)
        for offset in range(-20, 20):
            y = H // 2 + offset
            alpha_f = max(0, 18 - abs(offset))
            fd.line([(0, y), (W, y)], fill=(220, 240, 255, alpha_f))
        frost = frost.filter(ImageFilter.GaussianBlur(6))
        canvas = Image.alpha_composite(canvas, frost)

        canvas = Image.alpha_composite(canvas, self._noise_layer(W, H, 6))
        return canvas.convert("RGB")

    # ── DYNAMIC: Sports / Fashion / Footwear ──
    def _render_dynamic(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        dark = tuple(max(0, c - 80) for c in accent)
        canvas = self._linear_gradient(W, H, (20, 20, 30), dark).convert("RGBA")

        # Diagonal accent bands
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.polygon([(0, int(H * 0.3)), (W, 0), (W, int(H * 0.15)),
                       (0, int(H * 0.45))], fill=(*accent, 40))
        draw.polygon([(0, int(H * 0.55)), (W, int(H * 0.25)),
                       (W, int(H * 0.35)), (0, int(H * 0.65))],
                     fill=(*secondary, 25))
        canvas = Image.alpha_composite(canvas, layer)

        # Speed lines
        layer2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(layer2)
        rng = random.Random(42)
        for _ in range(30):
            y = rng.randint(0, H)
            x_start = rng.randint(0, W // 3)
            x_end = x_start + rng.randint(100, 500)
            d2.line([(x_start, y), (min(x_end, W), y)],
                    fill=(*accent, rng.randint(15, 50)), width=rng.randint(1, 3))
        canvas = Image.alpha_composite(canvas, layer2)

        # Central glow
        glow = self._radial_gradient(W, H, int(W * 0.6), int(H * 0.4),
                                     int(W * 0.5), accent, 40)
        canvas = Image.alpha_composite(canvas, glow)

        # Geometric triangles
        layer3 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d3 = ImageDraw.Draw(layer3)
        for _ in range(8):
            cx, cy = rng.randint(0, W), rng.randint(0, H)
            size = rng.randint(50, 200)
            d3.polygon([(cx, cy - size), (cx - size, cy + size // 2),
                        (cx + size, cy + size // 2)],
                       outline=(*accent, rng.randint(10, 30)))
        canvas = Image.alpha_composite(canvas, layer3)

        # Halftone dots in corner
        layer4 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d4 = ImageDraw.Draw(layer4)
        for x in range(0, W // 3, 15):
            for y in range(0, H // 3, 15):
                dist = math.sqrt((x / (W / 3)) ** 2 + (y / (H / 3)) ** 2)
                if dist < 1:
                    r = max(1, int(4 * (1 - dist)))
                    d4.ellipse([x - r, y - r, x + r, y + r],
                               fill=(*accent, int(30 * (1 - dist))))
        canvas = Image.alpha_composite(canvas, layer4)

        # Burst lines from center
        burst = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bd = ImageDraw.Draw(burst)
        bcx, bcy = int(W * 0.55), int(H * 0.45)
        for _ in range(20):
            angle = rng.uniform(0, math.pi * 2)
            length = rng.randint(200, 500)
            x2 = int(bcx + length * math.cos(angle))
            y2 = int(bcy + length * math.sin(angle))
            bd.line([(bcx, bcy), (x2, y2)],
                    fill=(*accent, rng.randint(8, 22)), width=2)
        canvas = Image.alpha_composite(canvas, burst)

        canvas = Image.alpha_composite(canvas, self._noise_layer(W, H, 5))
        return canvas.convert("RGB")

    # ── TECH: Mobile / Computers ──
    def _render_tech(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        canvas = self._linear_gradient(W, H, (8, 8, 20), (15, 15, 35)).convert("RGBA")

        # Hexagonal grid
        canvas = self._draw_hexagon_grid(canvas, accent, spacing=70, alpha=18)

        # Neon glows
        glow = self._radial_gradient(W, H, W // 2, H // 2, int(W * 0.45),
                                     accent, 35)
        canvas = Image.alpha_composite(canvas, glow)
        glow2 = self._radial_gradient(W, H, int(W * 0.2), int(H * 0.7),
                                      int(W * 0.3), secondary, 20)
        canvas = Image.alpha_composite(canvas, glow2)

        # Circuit-like lines
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        rng = random.Random(55)
        for _ in range(15):
            x, y = rng.randint(0, W), rng.randint(0, H)
            alpha = rng.randint(20, 50)
            for _ in range(rng.randint(3, 7)):
                if rng.random() > 0.5:
                    dx = rng.randint(30, 120) * rng.choice([-1, 1])
                    draw.line([(x, y), (x + dx, y)],
                              fill=(*accent, alpha), width=1)
                    x += dx
                else:
                    dy = rng.randint(30, 120) * rng.choice([-1, 1])
                    draw.line([(x, y), (x, y + dy)],
                              fill=(*accent, alpha), width=1)
                    y += dy
                draw.ellipse([x - 3, y - 3, x + 3, y + 3],
                             fill=(*accent, min(alpha + 30, 80)))
        canvas = Image.alpha_composite(canvas, layer)

        # Floating particles
        layer2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(layer2)
        for _ in range(50):
            px, py = rng.randint(0, W), rng.randint(0, H)
            pr = rng.randint(1, 3)
            d2.ellipse([px - pr, py - pr, px + pr, py + pr],
                       fill=(*accent, rng.randint(30, 100)))
        canvas = Image.alpha_composite(canvas, layer2)

        # Data stream lines (vertical, faint)
        stream = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(stream)
        for _ in range(12):
            sx = rng.randint(0, W)
            sy_start = rng.randint(0, H // 2)
            sy_end = sy_start + rng.randint(100, 400)
            for sy in range(sy_start, min(sy_end, H), 8):
                alpha_s = rng.randint(15, 40)
                sd.rectangle([sx - 1, sy, sx + 1, sy + 4],
                             fill=(*accent, alpha_s))
        canvas = Image.alpha_composite(canvas, stream)

        # Scanline effect
        layer3 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d3 = ImageDraw.Draw(layer3)
        for y in range(0, H, 3):
            d3.line([(0, y), (W, y)], fill=(0, 0, 0, 8))
        canvas = Image.alpha_composite(canvas, layer3)

        return canvas.convert("RGB")

    # ── LUXURY: Jewelry / Watches ──
    def _render_luxury(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        canvas = self._linear_gradient(W, H, (10, 5, 15), (25, 15, 35)).convert("RGBA")

        gold = accent if sum(accent) > 200 else (212, 175, 55)

        # Center glow
        glow = self._radial_gradient(W, H, W // 2, H // 2, int(W * 0.5),
                                     gold, 25)
        canvas = Image.alpha_composite(canvas, glow)

        # Elegant curved lines
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for i in range(5):
            points = []
            offset = i * 80 + 100
            for x in range(0, W + 1, 3):
                t = x / W
                y = offset + int(60 * math.sin(t * math.pi * (1.5 + i * 0.2)))
                points.append((x, y))
            alpha = max(10, 40 - i * 8)
            for j in range(len(points) - 1):
                draw.line([points[j], points[j + 1]],
                          fill=(*gold, alpha), width=1)
        canvas = Image.alpha_composite(canvas, layer)

        # Sparkles
        canvas = self._draw_sparkles(canvas, 45, gold)

        # Vignette
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt(((X - W / 2) / (W * 0.6)) ** 2
                       + ((Y - H / 2) / (H * 0.6)) ** 2)
        alpha_vig = np.clip((dist - 0.5) * 200, 0, 150).astype(np.uint8)
        vig = np.zeros((H, W, 4), dtype=np.uint8)
        vig[:, :, 3] = alpha_vig
        canvas = Image.alpha_composite(canvas,
                                       Image.fromarray(vig, "RGBA"))

        # Diamond facets
        layer2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(layer2)
        rng = random.Random(88)
        for _ in range(12):
            cx, cy = rng.randint(W // 4, 3 * W // 4), rng.randint(H // 4, 3 * H // 4)
            size = rng.randint(30, 80)
            d2.polygon([(cx, cy - size), (cx + size, cy),
                        (cx, cy + size), (cx - size, cy)],
                       outline=(*gold, rng.randint(5, 18)))
        canvas = Image.alpha_composite(canvas, layer2)

        # Golden dust band across middle
        dust = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dd = ImageDraw.Draw(dust)
        for _ in range(200):
            dx = rng.randint(0, W)
            dy = rng.randint(int(H * 0.3), int(H * 0.7))
            dr = rng.randint(1, 3)
            dd.ellipse([dx - dr, dy - dr, dx + dr, dy + dr],
                       fill=(*gold, rng.randint(20, 70)))
        dust = dust.filter(ImageFilter.GaussianBlur(1))
        canvas = Image.alpha_composite(canvas, dust)

        return canvas.convert("RGB")

    # ── POWER: Automotive ──
    def _render_power(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        canvas = self._linear_gradient(W, H, (20, 20, 25), (40, 40, 50)).convert("RGBA")

        # Metallic sheen band
        sheen = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sheen)
        for y in range(int(H * 0.3), int(H * 0.5)):
            t = (y - int(H * 0.3)) / (H * 0.2)
            b = int(40 * math.sin(t * math.pi))
            sd.line([(0, y), (W, y)], fill=(b, b, b + 5, 25))
        canvas = Image.alpha_composite(canvas, sheen)

        # Accent glow
        glow = self._radial_gradient(W, H, int(W * 0.6), int(H * 0.3),
                                     int(W * 0.4), accent, 35)
        canvas = Image.alpha_composite(canvas, glow)

        # Speed lines (nearly horizontal)
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        rng = random.Random(66)
        for _ in range(30):
            x = rng.randint(-200, W)
            y = rng.randint(0, H)
            length = rng.randint(200, 600)
            rad = math.radians(rng.uniform(-10, 10))
            x2 = int(x + length * math.cos(rad))
            y2 = int(y + length * math.sin(rad))
            draw.line([(x, y), (x2, y2)],
                      fill=(*accent, rng.randint(10, 35)),
                      width=rng.randint(1, 3))
        canvas = Image.alpha_composite(canvas, layer)

        # Bold geometric shapes
        layer2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(layer2)
        d2.polygon([(int(W * 0.6), 0), (W, 0), (W, int(H * 0.4)),
                    (int(W * 0.7), int(H * 0.3))], fill=(*accent, 18))
        d2.polygon([(0, int(H * 0.7)), (int(W * 0.3), H), (0, H)],
                   fill=(*secondary, 14))
        canvas = Image.alpha_composite(canvas, layer2)

        # Tire-tread/grid pattern in corner
        grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(grid)
        for x in range(int(W * 0.7), W, 20):
            for y in range(int(H * 0.65), H, 20):
                gd.rectangle([x, y, x + 8, y + 3], fill=(*accent, 12))
        canvas = Image.alpha_composite(canvas, grid)

        # Lens flare spot
        flare = self._radial_gradient(W, H, int(W * 0.7), int(H * 0.25),
                                      120, (255, 255, 240), 50)
        canvas = Image.alpha_composite(canvas, flare)

        canvas = Image.alpha_composite(canvas, self._noise_layer(W, H, 5))
        return canvas.convert("RGB")

    # ── ELEGANT: Beauty / Skincare ──
    def _render_elegant(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        light = tuple(min(255, c + 120) for c in accent)
        canvas = self._linear_gradient(W, H, (255, 240, 245), light).convert("RGBA")

        # Soft radial glow
        glow = self._radial_gradient(W, H, W // 2, int(H * 0.4),
                                     int(W * 0.6), (255, 200, 220), 40)
        canvas = Image.alpha_composite(canvas, glow)

        # Flowing curves
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for i in range(6):
            base_y = int(H * (0.15 + i * 0.13))
            alpha = max(15, 50 - i * 8)
            points = []
            for x in range(0, W + 1, 2):
                t = x / W
                y = base_y + int(50 * math.sin(t * math.pi * (2 + i * 0.3) + i))
                points.append((x, y))
            for j in range(len(points) - 1):
                draw.line([points[j], points[j + 1]],
                          fill=(*accent, alpha), width=2)
        canvas = Image.alpha_composite(canvas, layer)

        # Bokeh circles
        layer2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(layer2)
        rng = random.Random(77)
        for _ in range(20):
            bx, by = rng.randint(0, W), rng.randint(0, H)
            br = rng.randint(30, 100)
            alpha = rng.randint(8, 25)
            d2.ellipse([bx - br, by - br, bx + br, by + br],
                       fill=(*accent, alpha),
                       outline=(*accent, min(alpha + 10, 40)))
        layer2 = layer2.filter(ImageFilter.GaussianBlur(5))
        canvas = Image.alpha_composite(canvas, layer2)

        # Petal shapes
        petals = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        pd = ImageDraw.Draw(petals)
        for _ in range(8):
            px, py = rng.randint(W // 4, 3 * W // 4), rng.randint(H // 4, 3 * H // 4)
            pw, ph = rng.randint(20, 50), rng.randint(40, 80)
            angle = rng.randint(0, 360)
            petal = Image.new("RGBA", (pw * 2, ph * 2), (0, 0, 0, 0))
            pd2 = ImageDraw.Draw(petal)
            pd2.ellipse([0, 0, pw * 2 - 1, ph * 2 - 1],
                        fill=(*accent, rng.randint(8, 20)))
            petal = petal.rotate(angle, expand=True, resample=Image.BICUBIC)
            ppx = max(0, min(px - petal.width // 2, W - petal.width))
            ppy = max(0, min(py - petal.height // 2, H - petal.height))
            if ppx + petal.width <= W and ppy + petal.height <= H:
                region = canvas.crop((ppx, ppy, ppx + petal.width, ppy + petal.height))
                canvas.paste(Image.alpha_composite(region.convert("RGBA"), petal),
                             (ppx, ppy))
        canvas = self._draw_sparkles(canvas, 20, (255, 200, 220))
        return canvas.convert("RGB")

    # ── CORPORATE: Banks / Insurance ──
    def _render_corporate(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        canvas = self._linear_gradient(W, H, (240, 245, 255),
                                       (200, 215, 240)).convert("RGBA")

        # Large geometric circles
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.ellipse([int(W * 0.55), int(-H * 0.2),
                      int(W * 1.2), int(H * 0.5)],
                     fill=(*accent, 12), outline=(*accent, 20))
        draw.ellipse([int(-W * 0.1), int(H * 0.6),
                      int(W * 0.3), int(H * 1.1)],
                     fill=(*secondary, 10))
        canvas = Image.alpha_composite(canvas, layer)

        # Grid lines
        layer2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(layer2)
        for x in range(0, W, 80):
            d2.line([(x, 0), (x, H)], fill=(*accent, 8), width=1)
        for y in range(0, H, 80):
            d2.line([(0, y), (W, y)], fill=(*accent, 8), width=1)
        canvas = Image.alpha_composite(canvas, layer2)

        # Left accent bar
        bar = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(bar).rectangle([0, 0, int(W * 0.03), H],
                                       fill=(*accent, 60))
        canvas = Image.alpha_composite(canvas, bar)

        # Upward arrow motif (subtle)
        arrows = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ad = ImageDraw.Draw(arrows)
        for i in range(3):
            ax = int(W * (0.6 + i * 0.12))
            ay = int(H * 0.4)
            size = 60 + i * 20
            ad.polygon([(ax, ay - size), (ax - size // 2, ay),
                        (ax + size // 2, ay)],
                       outline=(*accent, 15 - i * 3))
        canvas = Image.alpha_composite(canvas, arrows)

        return canvas.convert("RGB")

    # ── MODERN: Default ──
    def _render_modern(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        dark_accent = tuple(max(0, c - 60) for c in accent)
        canvas = self._linear_gradient(W, H, dark_accent, (20, 20, 30)).convert("RGBA")

        # Gradient mesh (overlapping radial gradients)
        for cx, cy, rf, c, a in [
            (0.2, 0.2, 0.5, accent, 40),
            (0.8, 0.3, 0.4, secondary, 30),
            (0.5, 0.8, 0.45, tuple(max(0, int(v * 0.7)) for v in accent), 25),
        ]:
            g = self._radial_gradient(W, H, int(W * cx), int(H * cy),
                                      int(max(W, H) * rf), c, a)
            canvas = Image.alpha_composite(canvas, g)

        # Corner brackets
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        m, s = 30, 80
        c = (*accent, 60)
        for cx, cy, dx, dy in [(m, m, 1, 1), (W - m, m, -1, 1),
                                (m, H - m, 1, -1), (W - m, H - m, -1, -1)]:
            draw.line([(cx, cy), (cx + s * dx, cy)], fill=c, width=2)
            draw.line([(cx, cy), (cx, cy + s * dy)], fill=c, width=2)
        canvas = Image.alpha_composite(canvas, layer)

        # Diagonal stripes
        layer2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(layer2)
        for offset in range(-H, W + H, 45):
            d2.line([(offset, 0), (offset + H, H)],
                    fill=(*accent, 12), width=1)
        canvas = Image.alpha_composite(canvas, layer2)

        # Dot pattern accent
        dots = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dd = ImageDraw.Draw(dots)
        for x in range(50, int(W * 0.3), 28):
            for y in range(50, int(H * 0.25), 28):
                dd.ellipse([x - 1, y - 1, x + 1, y + 1],
                           fill=(*accent, 25))
        canvas = Image.alpha_composite(canvas, dots)

        canvas = Image.alpha_composite(canvas, self._noise_layer(W, H, 5))
        return canvas.convert("RGB")


# ---------------------------------------------------------------------------
# Text Generator (Pollinations.ai - free, no API key)
# ---------------------------------------------------------------------------

class PollinationsTextGenerator:
    URL = "https://text.pollinations.ai/"
    TIMEOUT = 60

    def generate(self, system_prompt: str, user_prompt: str, model: str = "openai") -> Optional[str]:
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "model": model,
                "seed": int(time.time()) % 10000,
                "jsonMode": False,
            }
            resp = requests.post(self.URL, json=payload, timeout=self.TIMEOUT)
            if resp.status_code == 200 and len(resp.text) > 10:
                return resp.text.strip()
        except Exception:
            pass
        return None

    def generate_json(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "model": "openai",
                "seed": int(time.time()) % 10000,
                "jsonMode": True,
            }
            resp = requests.post(self.URL, json=payload, timeout=self.TIMEOUT)
            if resp.status_code == 200:
                text = resp.text.strip()
                if text.startswith("```"):
                    text = re.sub(r'^```(?:json)?\s*', '', text)
                    text = re.sub(r'\s*```$', '', text)
                return json.loads(text)
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Content Generator (combines RAG + Pollinations text + templates)
# ---------------------------------------------------------------------------

class ContentGenerator:
    def __init__(self, clip_extractor: CLIPContentExtractor):
        self.clip = clip_extractor

    def generate_product_content(
        self, brand: str, category: str, subcategory: str,
        features: List[str], tagline: str, query: str
    ) -> Dict[str, Any]:
        """Generate product title, description, captions using smart templates."""
        brand_clean = brand.replace("_", " ")
        sub_clean = subcategory.replace("_", " ") if subcategory else category.replace("_", " ")

        # Extract product type from user query (strip brand name, filler, and action words)
        product_desc = ""
        if query:
            desc = query.lower()
            for word in brand_clean.lower().split():
                desc = desc.replace(word, "")
            filler = {"ad", "ads", "advertisement", "pamphlet", "poster", "banner",
                      "create", "make", "generate", "please", "want", "need", "i",
                      "a", "an", "the", "for", "with", "and", "of", "in", "on",
                      "me", "my", "give", "show", "by", "it", "that", "this",
                      "wearing", "worn", "using", "holding", "showing", "displaying",
                      "person", "people", "man", "woman", "women", "men", "model",
                      "models", "girl", "boy", "lady", "ladies"}
            words = [w for w in desc.split() if w not in filler]
            product_desc = " ".join(words).strip()

        # Use query-derived product type for display, fall back to category
        display_product = product_desc if product_desc else sub_clean

        feat1 = features[0].lower() if features else "premium quality"
        feat2 = features[1].lower() if len(features) > 1 else "exceptional design"
        feat3 = features[2] if len(features) > 2 else "Premium Quality"

        title = f"{brand_clean} {display_product.title()} - {tagline}" if tagline else f"Discover {brand_clean} {display_product.title()}"
        if len(title) > 65:
            title = f"{brand_clean} - {tagline}" if tagline else f"{brand_clean} Collection"

        return {
            "product_title": title,
            "product_description": (
                f"Experience the excellence of {brand_clean} {display_product}. "
                f"Featuring {feat1} and {feat2}. "
                f"Trusted and designed for the discerning customer."
            ),
            "instagram_caption": (
                f"Elevate your style with {brand_clean} {display_product}! "
                f"{features[0] if features else 'Premium Quality'} | {feat3}. "
                f"Shop now and experience the difference!"
            ),
            "whatsapp_copy": (
                f"Check out {brand_clean} {display_product}! "
                f"{tagline if tagline else feat1.capitalize()}. Get yours today!"
            ),
            "hashtags": [
                brand_clean.replace(" ", ""),
                display_product.replace(" ", ""),
                "premium", "trending", "shopnow",
            ],
        }

    def generate_ad_content_for_language(
        self, brand: str, category: str, subcategory: str,
        features: List[str], tagline: str, query: str, language: str
    ) -> Optional[Dict[str, str]]:
        """Return None — translations handled by the Translator class."""
        return None


# ---------------------------------------------------------------------------
# Translator (free, no API key)
# ---------------------------------------------------------------------------

class Translator:
    def __init__(self):
        self._translator = None

    def _get_translator(self, source: str, target: str):
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source=source, target=target)
        except ImportError:
            return None

    def translate(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        if target_lang == source_lang or target_lang == "en":
            return text
        try:
            translator = self._get_translator(source_lang, target_lang)
            if translator:
                return translator.translate(text)
        except Exception:
            pass
        return text

    def translate_content(self, content: Dict[str, str], target_lang: str) -> Dict[str, str]:
        if target_lang == "en":
            return content
        translated = {}
        for key, value in content.items():
            if isinstance(value, str) and value:
                translated[key] = self.translate(value, target_lang)
            elif isinstance(value, list):
                translated[key] = [self.translate(v, target_lang) for v in value if isinstance(v, str)]
            else:
                translated[key] = value
        return translated


# ---------------------------------------------------------------------------
# Image Enhancer (fully local, PIL-based)
# ---------------------------------------------------------------------------

class ImageEnhancer:
    @staticmethod
    def enhance(image: Image.Image, brightness: float = 1.15,
                contrast: float = 1.2, sharpness: float = 1.3,
                color: float = 1.1) -> Image.Image:
        img = image.copy()
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        img = ImageEnhance.Color(img).enhance(color)
        return img

    @staticmethod
    def auto_enhance(image: Image.Image) -> Image.Image:
        """Smart auto-enhancement based on image analysis."""
        img = image.copy().convert("RGB")
        arr = np.array(img).astype(float)

        # Analyze current image
        mean_brightness = arr.mean() / 255.0
        std_val = arr.std() / 255.0

        # Adjust based on analysis
        brightness = 1.0
        if mean_brightness < 0.4:
            brightness = 1.3
        elif mean_brightness > 0.7:
            brightness = 0.9

        contrast = 1.0
        if std_val < 0.15:
            contrast = 1.4
        elif std_val > 0.35:
            contrast = 0.95

        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Sharpness(img).enhance(1.25)
        img = ImageEnhance.Color(img).enhance(1.1)

        return img


# ---------------------------------------------------------------------------
# Pro Ad Designer (1080x1080 HD - 6 professional themes)
# ---------------------------------------------------------------------------

class ProAdDesigner:
    """Professional ad designer with multiple theme templates."""

    WIDTH = 1080
    HEIGHT = 1080

    FONTS = {
        "headline": "C:/Windows/Fonts/segoeuib.ttf",
        "body": "C:/Windows/Fonts/segoeui.ttf",
        "accent": "C:/Windows/Fonts/georgiai.ttf",
        "display": "C:/Windows/Fonts/impact.ttf",
        "elegant": "C:/Windows/Fonts/times.ttf",
        "modern": "C:/Windows/Fonts/calibri.ttf",
    }

    # Fonts for non-Latin scripts — (path, ttc_index) tuples for .ttc collections
    INDIC_FONTS = {
        "headline": ("C:/Windows/Fonts/Nirmala.ttc", 1),   # Nirmala UI Bold
        "body": ("C:/Windows/Fonts/Nirmala.ttc", 0),        # Nirmala UI Regular
        "accent": ("C:/Windows/Fonts/Nirmala.ttc", 0),
        "display": ("C:/Windows/Fonts/Nirmala.ttc", 1),
        "elegant": ("C:/Windows/Fonts/Nirmala.ttc", 0),
        "modern": ("C:/Windows/Fonts/Nirmala.ttc", 0),
    }

    CJK_FONTS = {
        "headline": ("C:/Windows/Fonts/msyhbd.ttc", 0),   # Microsoft YaHei Bold
        "body": ("C:/Windows/Fonts/msyh.ttc", 0),          # Microsoft YaHei
        "accent": ("C:/Windows/Fonts/msyh.ttc", 0),
        "display": ("C:/Windows/Fonts/msyhbd.ttc", 0),
        "elegant": ("C:/Windows/Fonts/msyh.ttc", 0),
        "modern": ("C:/Windows/Fonts/msyh.ttc", 0),
    }

    ARABIC_FONTS = {
        "headline": "C:/Windows/Fonts/segoeuib.ttf",
        "body": "C:/Windows/Fonts/segoeui.ttf",
        "accent": "C:/Windows/Fonts/segoeui.ttf",
        "display": "C:/Windows/Fonts/segoeuib.ttf",
        "elegant": "C:/Windows/Fonts/segoeui.ttf",
        "modern": "C:/Windows/Fonts/segoeui.ttf",
    }

    CTA_TEXTS = {
        "footwear": "SHOP NOW", "clothing_brands": "SHOP NOW",
        "mobile_devices": "BUY NOW", "computers": "BUY NOW",
        "four_wheelers": "BOOK A TEST DRIVE", "two_wheelers": "BOOK A TEST DRIVE",
        "water": "ORDER NOW", "soft_drinks": "GRAB YOURS",
        "juice": "TRY NOW", "banks": "OPEN ACCOUNT",
        "insurance": "GET QUOTE", "jewellery": "EXPLORE COLLECTION",
        "watches": "EXPLORE COLLECTION", "skincare_and_makeup": "SHOP NOW",
    }

    THEMES = [
        "minimal_clean", "bold_hero", "premium_dark",
        "split_layout", "card_float", "gradient_mesh",
    ]

    CATEGORY_THEMES = {
        "jewellery": ["premium_dark", "minimal_clean"],
        "watches": ["premium_dark", "card_float"],
        "footwear": ["bold_hero", "split_layout"],
        "clothing_brands": ["split_layout", "minimal_clean"],
        "mobile_devices": ["gradient_mesh", "card_float"],
        "computers": ["gradient_mesh", "minimal_clean"],
        "four_wheelers": ["bold_hero", "premium_dark"],
        "two_wheelers": ["bold_hero", "gradient_mesh"],
        "soft_drinks": ["bold_hero", "gradient_mesh"],
        "water": ["minimal_clean", "card_float"],
        "juice": ["bold_hero", "gradient_mesh"],
        "skincare_and_makeup": ["minimal_clean", "premium_dark"],
        "banks": ["card_float", "split_layout"],
        "insurance": ["split_layout", "card_float"],
    }

    def __init__(self):
        self._font_cache: Dict[str, ImageFont.FreeTypeFont] = {}
        self._script = "latin"  # "latin", "indic", "cjk", "arabic"
        self._theme_renderers = {
            "minimal_clean": self._render_minimal_clean,
            "bold_hero": self._render_bold_hero,
            "premium_dark": self._render_premium_dark,
            "split_layout": self._render_split_layout,
            "card_float": self._render_card_float,
            "gradient_mesh": self._render_gradient_mesh,
        }

    # --- Shared utilities ---

    @staticmethod
    def _detect_script(text: str) -> str:
        """Detect the primary script of text: 'indic', 'cjk', 'arabic', or 'latin'."""
        if not text:
            return "latin"
        indic_count = 0
        cjk_count = 0
        arabic_count = 0
        total = 0
        for ch in text:
            cp = ord(ch)
            if cp < 0x20 or ch.isspace():
                continue
            total += 1
            # Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
            if 0x0900 <= cp <= 0x0D7F:
                indic_count += 1
            # CJK Unified, Hiragana, Katakana, Korean
            elif (0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF
                  or 0xAC00 <= cp <= 0xD7AF or 0x3400 <= cp <= 0x4DBF):
                cjk_count += 1
            # Arabic, Arabic Supplement, Arabic Extended
            elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
                arabic_count += 1
        if total == 0:
            return "latin"
        if indic_count / total > 0.15:
            return "indic"
        if cjk_count / total > 0.15:
            return "cjk"
        if arabic_count / total > 0.15:
            return "arabic"
        return "latin"

    def _font(self, role: str, size: int) -> ImageFont.FreeTypeFont:
        # Select font dict based on current script
        if self._script == "indic":
            font_dict = self.INDIC_FONTS
        elif self._script == "cjk":
            font_dict = self.CJK_FONTS
        elif self._script == "arabic":
            font_dict = self.ARABIC_FONTS
        else:
            font_dict = self.FONTS

        entry = font_dict.get(role, self.FONTS.get(role, role))

        # Handle (path, ttc_index) tuples for .ttc font collections
        if isinstance(entry, tuple):
            path, ttc_index = entry
        else:
            path, ttc_index = entry, 0

        key = f"{path}_{ttc_index}_{size}"
        if key not in self._font_cache:
            try:
                self._font_cache[key] = ImageFont.truetype(path, size, index=ttc_index)
            except Exception:
                # Fallback chain
                fallbacks = []
                if self._script == "indic":
                    fallbacks = [
                        ("C:/Windows/Fonts/Nirmala.ttc", 1),  # Bold
                        ("C:/Windows/Fonts/Nirmala.ttc", 0),  # Regular
                    ]
                elif self._script == "cjk":
                    fallbacks = [
                        ("C:/Windows/Fonts/msyh.ttc", 0),
                        ("C:/Windows/Fonts/simsun.ttc", 0),
                    ]
                fallbacks.append(("arial.ttf", 0))
                loaded = False
                for fb_path, fb_idx in fallbacks:
                    try:
                        self._font_cache[key] = ImageFont.truetype(fb_path, size, index=fb_idx)
                        loaded = True
                        break
                    except Exception:
                        continue
                if not loaded:
                    self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def _text_size(self, draw, text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    @staticmethod
    def _hex_to_rgb(h: str):
        h = h.lstrip("#")
        if len(h) < 6:
            h = h.ljust(6, "0")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _round_corners(self, img, radius):
        mask = Image.new("L", img.size, 0)
        d = ImageDraw.Draw(mask)
        d.rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius, fill=255)
        result = img.convert("RGBA")
        result.putalpha(mask)
        return result

    def _word_wrap(self, draw, text, font, max_width):
        words = text.split()
        lines, current = [], ""
        for word in words:
            test = f"{current} {word}".strip()
            tw, _ = self._text_size(draw, test, font)
            if tw <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [text]

    def _draw_text_shadow(self, draw, xy, text, font, fill=(255, 255, 255, 255),
                          shadow_color=(0, 0, 0, 120), offset=3):
        x, y = xy
        for dx, dy, a_mult in [(offset, offset, 0.3), (offset // 2, offset // 2, 0.6)]:
            sc = (*shadow_color[:3], int(shadow_color[3] * a_mult))
            draw.text((x + dx, y + dy), text, fill=sc, font=font)
        draw.text((x, y), text, fill=fill, font=font)

    def _draw_text_outline(self, draw, xy, text, font, fill=(255, 255, 255, 255),
                           outline_color=(0, 0, 0, 200), width=2):
        x, y = xy
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                if dx * dx + dy * dy <= width * width:
                    draw.text((x + dx, y + dy), text, fill=outline_color, font=font)
        draw.text((x, y), text, fill=fill, font=font)

    def _draw_spaced_text(self, canvas, xy, text, font, fill, spacing=4,
                          center_x=None, y_pos=None):
        draw = ImageDraw.Draw(canvas)
        if center_x is not None:
            total_w = sum(self._text_size(draw, c, font)[0] + spacing for c in text) - spacing
            x = center_x - total_w // 2
            y = y_pos if y_pos is not None else (xy[1] if xy else 0)
        else:
            x, y = xy
        for char in text:
            draw.text((x, y), char, fill=fill, font=font)
            cw, _ = self._text_size(draw, char, font)
            x += cw + spacing
        return canvas

    def _draw_secondary_text(self, canvas, content, y, margin, max_width,
                                light_text=True, centered=False):
        """Draw the secondary language headline + features below the primary if bilingual."""
        if not content.headline_secondary:
            return y
        draw = ImageDraw.Draw(canvas)
        # Temporarily switch to Latin font for the English secondary text
        saved_script = self._script
        self._script = "latin"

        # Secondary headline - smaller, muted color
        sec_font = self._font("body", 18)
        color = (200, 200, 210, 180) if light_text else (80, 80, 90, 180)
        lines = self._word_wrap(draw, content.headline_secondary, sec_font, max_width)
        for line in lines[:2]:
            if centered:
                tw, _ = self._text_size(draw, line, sec_font)
                draw.text(((canvas.width - tw) // 2, y), line, fill=color, font=sec_font)
            else:
                draw.text((margin, y), line, fill=color, font=sec_font)
            y += 24
        y += 6

        # Secondary features (first 3, smaller)
        if content.features_secondary:
            sec_feat_font = self._font("body", 14)
            feat_color = (170, 170, 180, 150) if light_text else (110, 110, 120, 150)
            for feat in content.features_secondary[:3]:
                text = f"  {feat}"
                if centered:
                    tw, _ = self._text_size(draw, text, sec_feat_font)
                    draw.text(((canvas.width - tw) // 2, y), text, fill=feat_color, font=sec_feat_font)
                else:
                    draw.text((margin, y), text, fill=feat_color, font=sec_feat_font)
                y += 20
            y += 6

        self._script = saved_script
        return y

    def _place_logo(self, canvas, logo, position="top-left", max_height=70, padding=30, backing=True):
        if logo is None:
            return canvas
        logo = logo.copy().convert("RGBA")
        lw, lh = logo.size
        if lh > 0:
            scale = max_height / lh
            logo_w = min(int(lw * scale), max_height * 3)
            logo = logo.resize((logo_w, max_height), Image.LANCZOS)
        else:
            return canvas

        W, H = canvas.size
        positions = {
            "top-left": (padding, padding),
            "top-right": (W - logo.width - padding, padding),
            "top-center": ((W - logo.width) // 2, padding),
            "bottom-left": (padding, H - max_height - padding),
            "bottom-right": (W - logo.width - padding, H - max_height - padding),
        }
        x, y = positions.get(position, positions["top-left"])

        if backing:
            # Only apply subtle frosted backing — lighter and more elegant
            back_pad = 10
            bw, bh = logo.width + 2 * back_pad, max_height + 2 * back_pad
            bx, by = x - back_pad, y - back_pad
            if 0 <= bx and bx + bw <= W and 0 <= by and by + bh <= H:
                region = canvas.crop((bx, by, bx + bw, by + bh)).convert("RGBA")
                blurred = region.filter(ImageFilter.GaussianBlur(12))
                frost = Image.new("RGBA", (bw, bh), (255, 255, 255, 35))
                backing_img = Image.alpha_composite(blurred, frost)
                mask = Image.new("L", (bw, bh), 0)
                ImageDraw.Draw(mask).rounded_rectangle([0, 0, bw - 1, bh - 1], radius=10, fill=255)
                backing_img.putalpha(mask)
                canvas.paste(Image.alpha_composite(
                    canvas.crop((bx, by, bx + bw, by + bh)).convert("RGBA"),
                    backing_img), (bx, by))

        # Draw logo with subtle drop shadow for visibility
        shadow_offset = 2
        if logo.width > 10 and logo.height > 10:
            shadow = Image.new("RGBA", logo.size, (0, 0, 0, 0))
            # Use logo alpha as shadow shape
            if logo.mode == "RGBA":
                alpha = logo.split()[3]
                from PIL import ImageFilter as IF2
                shadow_alpha = alpha.point(lambda p: min(p, 40))
                shadow.putalpha(shadow_alpha)
                shadow = shadow.filter(ImageFilter.GaussianBlur(2))
                sx, sy = x + shadow_offset, y + shadow_offset
                if sx + shadow.width <= W and sy + shadow.height <= H:
                    rg = canvas.crop((sx, sy, sx + shadow.width, sy + shadow.height)).convert("RGBA")
                    canvas.paste(Image.alpha_composite(rg, shadow), (sx, sy))

        canvas.paste(logo, (x, y), logo)
        return canvas

    def _draw_cta_button(self, canvas, text, position, color, style="rounded", font_size=22):
        font = self._font("headline", font_size)
        draw = ImageDraw.Draw(canvas)
        tw, th = self._text_size(draw, text, font)
        pad_x, pad_y = 32, 14
        btn_w = max(tw + 2 * pad_x, 180)
        btn_h = th + 2 * pad_y

        # Render at 2x for quality
        scale = 2
        btn = Image.new("RGBA", (btn_w * scale, btn_h * scale), (0, 0, 0, 0))
        bd = ImageDraw.Draw(btn)

        if style == "outline":
            bd.rounded_rectangle([0, 0, btn_w * scale - 1, btn_h * scale - 1],
                                 radius=btn_h * scale // 2, outline=(*color, 240), width=3 * scale)
        else:
            bd.rounded_rectangle([0, 0, btn_w * scale - 1, btn_h * scale - 1],
                                 radius=btn_h * scale // 2, fill=(*color, 240))

        lum = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
        if style == "outline":
            txt_col = (*color, 255)
        else:
            txt_col = (20, 20, 20, 255) if lum > 140 else (255, 255, 255, 255)

        big_font = self._font("headline", font_size * scale)
        btw, bth = self._text_size(bd, text, big_font)
        bd.text(((btn_w * scale - btw) // 2, (btn_h * scale - bth) // 2 - 2),
                text, fill=txt_col, font=big_font)

        btn = btn.resize((btn_w, btn_h), Image.LANCZOS)

        # Drop shadow
        x, y = position
        shadow = Image.new("RGBA", (btn_w + 10, btn_h + 10), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.rounded_rectangle([5, 5, btn_w + 4, btn_h + 4], radius=btn_h // 2, fill=(0, 0, 0, 35))
        shadow = shadow.filter(ImageFilter.GaussianBlur(4))
        sx, sy = max(0, x - 3), max(0, y - 3)
        sw, sh = min(shadow.width, canvas.width - sx), min(shadow.height, canvas.height - sy)
        if sw > 0 and sh > 0:
            sr = canvas.crop((sx, sy, sx + sw, sy + sh)).convert("RGBA")
            sc = shadow.crop((0, 0, sw, sh))
            canvas.paste(Image.alpha_composite(sr, sc), (sx, sy))

        if x + btn_w <= canvas.width and y + btn_h <= canvas.height:
            canvas.paste(btn, (x, y), btn)
        return canvas, btn_h

    def _draw_feature_list(self, canvas, features, start_xy, max_width,
                           style="icons", color=(255, 255, 255, 240),
                           accent=(255, 200, 50), font_size=18):
        draw = ImageDraw.Draw(canvas)
        font = self._font("body", font_size)
        x, y = start_xy
        icons = ["\u2713", "\u2605", "\u25CF", "\u25B6"]

        for i, feat in enumerate(features[:5]):
            if style == "numbered":
                num_font = self._font("headline", font_size + 4)
                num = f"{i + 1:02d}"
                draw.text((x, y - 2), num, fill=(*accent[:3], 160), font=num_font)
                draw.text((x + 38, y + 2), feat, fill=color, font=font)
                y += font_size + 18
            elif style == "dividers":
                draw.text((x, y), feat, fill=color, font=font)
                ftw, fth = self._text_size(draw, feat, font)
                y += fth + 6
                draw.line([(x, y), (x + min(ftw, int(max_width * 0.6)), y)],
                          fill=(*accent[:3], 50), width=1)
                y += 10
            else:  # icons
                icon = icons[i % len(icons)]
                icon_font = self._font("body", font_size + 2)
                draw.text((x, y), icon, fill=(*accent[:3], 200), font=icon_font)
                draw.text((x + 26, y), feat, fill=color, font=font)
                y += font_size + 14
        return y

    def _get_cta_text(self, content):
        if content.cta_text:
            return content.cta_text
        return self.CTA_TEXTS.get(content.subcategory,
               self.CTA_TEXTS.get(content.category, "EXPLORE NOW"))

    def _cover_fill(self, img, W, H):
        """Scale image to cover entire canvas (crop overflow)."""
        img = img.copy().convert("RGB")
        pw, ph = img.size
        scale = max(W / pw, H / ph)
        img = img.resize((int(pw * scale), int(ph * scale)), Image.LANCZOS)
        lx = (img.width - W) // 2
        ly = (img.height - H) // 2
        return img.crop((lx, ly, lx + W, ly + H))

    def _generate_subtle_noise(self, W, H, intensity=8):
        noise = np.random.randint(0, intensity, (H, W), dtype=np.uint8)
        noise_rgba = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        noise_rgba.putalpha(Image.fromarray(noise, "L"))
        return noise_rgba

    # --- Rich decorative element methods ---

    def _draw_wave_divider(self, canvas, y_pos, color, amplitude=40, thickness=3, fill_below=None):
        """Draw a smooth curved wave separator line across the canvas."""
        W, H = canvas.size
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        import math
        # Generate wave points
        points = []
        for x in range(0, W + 1, 2):
            t = x / W
            y = y_pos + int(amplitude * math.sin(t * math.pi * 2.0))
            points.append((x, y))
        # Fill below wave
        if fill_below:
            fill_points = list(points) + [(W, H), (0, H)]
            draw.polygon(fill_points, fill=fill_below)
        # Draw wave line
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=color, width=thickness)
        canvas = Image.alpha_composite(canvas, layer)
        return canvas

    def _draw_badge(self, canvas, position, text, bg_color, text_color=(255, 255, 255, 255),
                    size=80, style="circle"):
        """Draw a circular or ribbon badge with text."""
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        x, y = position
        if style == "circle":
            # Draw circle
            draw.ellipse([x, y, x + size, y + size], fill=bg_color,
                         outline=(255, 255, 255, 180), width=2)
            # Text inside
            font = self._font("headline", max(10, size // 5))
            lines = text.upper().split('\n') if '\n' in text else [text.upper()]
            ty = y + size // 2 - len(lines) * (size // 5 + 2) // 2
            for line in lines[:3]:
                tw, _ = self._text_size(draw, line, font)
                draw.text(((2 * x + size - tw) // 2, ty), line, fill=text_color, font=font)
                ty += size // 5 + 4
        elif style == "ribbon":
            # Draw ribbon shape
            rw, rh = size * 2, size // 2
            pts = [(x, y), (x + rw, y), (x + rw - 10, y + rh // 2),
                   (x + rw, y + rh), (x, y + rh), (x + 10, y + rh // 2)]
            draw.polygon(pts, fill=bg_color)
            font = self._font("headline", max(10, rh // 2 - 2))
            tw, _ = self._text_size(draw, text.upper(), font)
            draw.text(((2 * x + rw - tw) // 2, y + rh // 4), text.upper(),
                      fill=text_color, font=font)
        canvas = Image.alpha_composite(canvas, layer)
        return canvas

    def _draw_bubbles(self, canvas, accent_color, count=25, region=None):
        """Draw scattered translucent bubbles/droplets across the canvas."""
        W, H = canvas.size
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        import random
        rng = random.Random()  # random each time for variety
        x_min = region[0] if region else 0
        y_min = region[1] if region else 0
        x_max = region[2] if region else W
        y_max = region[3] if region else H
        for _ in range(count):
            bx = rng.randint(x_min, x_max)
            by = rng.randint(y_min, y_max)
            br = rng.randint(3, 18)
            alpha = rng.randint(15, 55)
            # Main bubble
            draw.ellipse([bx - br, by - br, bx + br, by + br],
                         fill=(*accent_color[:3], alpha),
                         outline=(*accent_color[:3], min(alpha + 20, 80)))
            # Highlight spot (top-left of bubble for 3D effect)
            if br > 6:
                hs = max(2, br // 4)
                hx, hy = bx - br // 3, by - br // 3
                draw.ellipse([hx - hs, hy - hs, hx + hs, hy + hs],
                             fill=(255, 255, 255, min(alpha + 10, 60)))
        canvas = Image.alpha_composite(canvas, layer)
        return canvas

    def _draw_glow_circle(self, canvas, center, radius, color, intensity=30):
        """Draw a soft glowing circle (radial gradient) at given position."""
        W, H = canvas.size
        cx, cy = center
        layer_arr = np.zeros((H, W, 4), dtype=np.uint8)
        Y_g, X_g = np.ogrid[:H, :W]
        dist = np.sqrt((X_g - cx) ** 2 + (Y_g - cy) ** 2)
        falloff = np.clip(1 - dist / radius, 0, 1) ** 2
        for c in range(3):
            layer_arr[:, :, c] = np.clip(color[c] * falloff, 0, 255).astype(np.uint8)
        layer_arr[:, :, 3] = np.clip(intensity * falloff, 0, 255).astype(np.uint8)
        layer = Image.fromarray(layer_arr, "RGBA")
        canvas = Image.alpha_composite(canvas, layer)
        return canvas

    def _draw_diagonal_stripes(self, canvas, color, stripe_width=2, gap=40, alpha=20):
        """Draw subtle diagonal stripes across the canvas."""
        W, H = canvas.size
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for offset in range(-H, W + H, gap):
            draw.line([(offset, 0), (offset + H, H)],
                      fill=(*color[:3], alpha), width=stripe_width)
        canvas = Image.alpha_composite(canvas, layer)
        return canvas

    def _draw_corner_accents(self, canvas, color, size=80, thickness=3):
        """Draw elegant corner bracket accents."""
        W, H = canvas.size
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        m = 25  # margin from edge
        c = (*color[:3], min(color[3] if len(color) > 3 else 255, 120))
        for cx, cy, dx, dy in [(m, m, 1, 1), (W - m, m, -1, 1),
                                (m, H - m, 1, -1), (W - m, H - m, -1, -1)]:
            draw.line([(cx, cy), (cx + size * dx, cy)], fill=c, width=thickness)
            draw.line([(cx, cy), (cx, cy + size * dy)], fill=c, width=thickness)
        canvas = Image.alpha_composite(canvas, layer)
        return canvas

    def _apply_post_processing(self, canvas):
        rgb = canvas.convert("RGB")
        # Upscale-sharpen-downscale for crispness
        w, h = rgb.size
        up = rgb.resize((w * 2, h * 2), Image.LANCZOS)
        up = up.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=3))
        rgb = up.resize((w, h), Image.LANCZOS)
        rgb = ImageEnhance.Sharpness(rgb).enhance(1.15)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.1)
        rgb = ImageEnhance.Color(rgb).enhance(1.05)
        return rgb

    def _real_ad_background(self, W, H, ad_images, accent, style="dark"):
        """Use a REAL ad from the dataset as blurred background for authentic look."""
        canvas = Image.new("RGBA", (W, H), (30, 30, 40, 255))
        if not ad_images:
            return canvas
        bg = self._cover_fill(ad_images[0], W, H).convert("RGBA")
        if style == "dark":
            bg = bg.filter(ImageFilter.GaussianBlur(18))
            dark = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            dd = ImageDraw.Draw(dark)
            for row in range(H):
                t = row / H
                a = int(90 + 130 * (t ** 1.2))
                tint = tuple(int(accent[i] * 0.08 * t) for i in range(3))
                dd.line([(0, row), (W, row)], fill=(*tint, min(a, 225)))
            bg = Image.alpha_composite(bg, dark)
        elif style == "light":
            bg = bg.filter(ImageFilter.GaussianBlur(25))
            light = Image.new("RGBA", (W, H), (255, 255, 255, 180))
            bg = Image.alpha_composite(bg, light)
        elif style == "tinted":
            bg = bg.filter(ImageFilter.GaussianBlur(20))
            tint = Image.new("RGBA", (W, H), (*accent, 100))
            bg = Image.alpha_composite(bg, tint)
            dark = Image.new("RGBA", (W, H), (0, 0, 0, 80))
            bg = Image.alpha_composite(bg, dark)
        return bg

    def _draw_ref_thumbnails(self, canvas, thumbs, position="bottom-right",
                              thumb_size=100, style="tilted"):
        """Draw reference ad thumbnails from the dataset on the ad."""
        if not thumbs:
            return canvas
        W, H = canvas.size
        thumbs = thumbs[:3]

        if style == "tilted":
            angles = [-6, 3, -3]
            if position == "bottom-right":
                base_x = W - 40 - int(thumb_size * 1.7)
                base_y = H - 40 - thumb_size
            else:
                base_x = 25
                base_y = H - 40 - thumb_size
            for i, thumb in enumerate(thumbs):
                t = thumb.copy().convert("RGB")
                tw, th = t.size
                sc = min(thumb_size / tw, thumb_size / th)
                t = t.resize((int(tw * sc), int(th * sc)), Image.LANCZOS)
                bdr = 4
                framed = Image.new("RGBA", (t.width + 2 * bdr, t.height + 2 * bdr), (255, 255, 255, 200))
                framed.paste(t.convert("RGBA"), (bdr, bdr))
                framed = self._round_corners(framed, 8)
                # Shadow
                shadow = Image.new("RGBA", (framed.width + 6, framed.height + 6), (0, 0, 0, 0))
                sd = ImageDraw.Draw(shadow)
                sd.rounded_rectangle([3, 3, shadow.width - 1, shadow.height - 1], 10, fill=(0, 0, 0, 40))
                shadow = shadow.filter(ImageFilter.GaussianBlur(3))
                shadow.paste(framed, (0, 0), framed)
                framed = shadow
                rotated = framed.rotate(angles[i % 3], expand=True, resample=Image.BICUBIC)
                tx = int(base_x + i * (thumb_size * 0.48))
                ty = int(base_y - i * 8)
                pw = min(rotated.width, W - tx)
                ph = min(rotated.height, H - ty)
                if pw > 0 and ph > 0 and 0 <= tx and 0 <= ty:
                    cr = rotated.crop((0, 0, pw, ph))
                    rg = canvas.crop((tx, ty, tx + pw, ty + ph)).convert("RGBA")
                    canvas.paste(Image.alpha_composite(rg, cr), (tx, ty))
        elif style == "strip":
            gap = 8
            total_w = len(thumbs) * (thumb_size + gap) - gap
            if position == "bottom-right":
                sx = W - total_w - 30
            else:
                sx = 30
            sy = H - int(thumb_size * 0.7) - 25
            for i, thumb in enumerate(thumbs):
                t = thumb.copy().convert("RGB")
                tw, th = t.size
                sc = min(thumb_size / tw, (thumb_size * 0.7) / th)
                t = t.resize((int(tw * sc), int(th * sc)), Image.LANCZOS)
                bdr = 3
                framed = Image.new("RGBA", (t.width + 2 * bdr, t.height + 2 * bdr), (255, 255, 255, 180))
                framed.paste(t.convert("RGBA"), (bdr, bdr))
                framed = self._round_corners(framed, 6)
                tx = sx + i * (thumb_size + gap)
                ty = sy
                if 0 <= tx and tx + framed.width <= W and 0 <= ty and ty + framed.height <= H:
                    rg = canvas.crop((tx, ty, tx + framed.width, ty + framed.height)).convert("RGBA")
                    canvas.paste(Image.alpha_composite(rg, framed), (tx, ty))
        return canvas

    def _draw_brand_url_bar(self, canvas, brand_name, domain, color_pair, position="bottom", side="left"):
        """Draw a professional brand info bar with gradient."""
        W, H = canvas.size
        bar_h = 55
        bar_w = int(W * 0.45)
        bar_y = H - bar_h - 10
        c1, c2 = color_pair

        bar = Image.new("RGBA", (bar_w, bar_h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bar)
        for col in range(bar_w):
            t = col / bar_w
            rgb = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
            alpha = int(210 * (1 - t * 0.3))
            bd.line([(col, 0), (col, bar_h)], fill=(*rgb, alpha))
        # Taper edge
        for col in range(bar_w - 30, bar_w):
            t = (col - (bar_w - 30)) / 30
            for row in range(bar_h):
                px = bar.getpixel((col, row))
                bar.putpixel((col, row), (px[0], px[1], px[2], max(0, int(px[3] * (1 - t)))))

        bar_x = 0 if side == "left" else (W - bar_w)
        if side == "right":
            bar = bar.transpose(Image.FLIP_LEFT_RIGHT)
        canvas.paste(bar, (bar_x, bar_y), bar)

        draw = ImageDraw.Draw(canvas)
        info_font = self._font("body", 12)
        url_font = self._font("headline", 18)
        brand_url = domain or f"www.{brand_name.lower().replace('_', '').replace(' ', '')}.com"
        margin = 20 if side == "left" else (W - bar_w + 20)
        draw.text((margin, bar_y + 6), "VISIT US", fill=(255, 255, 255, 220), font=info_font)
        draw.text((margin, bar_y + 24), brand_url, fill=(255, 255, 80, 255), font=url_font)
        return canvas

    # --- AI Image Analysis (from PamphletComposer) ---

    def _analyze_image(self, img):
        """Divide image into 3x3 grid, compute complexity per cell.
        Returns which side is busy, where free space is, avg brightness."""
        arr = np.array(img.convert("RGB")).astype(float)
        h, w = arr.shape[:2]
        grid_h, grid_w = h // 3, w // 3

        complexity = np.zeros((3, 3))
        brightness = np.zeros((3, 3))
        for r in range(3):
            for c in range(3):
                cell = arr[r * grid_h:(r + 1) * grid_h, c * grid_w:(c + 1) * grid_w]
                dx = np.abs(np.diff(cell, axis=1)).mean()
                dy = np.abs(np.diff(cell, axis=0)).mean()
                complexity[r, c] = dx + dy
                brightness[r, c] = cell.mean()

        left_complexity = complexity[:, 0].mean()
        right_complexity = complexity[:, 2].mean()

        if left_complexity > right_complexity * 1.15:
            text_side = "right"
        elif right_complexity > left_complexity * 1.15:
            text_side = "left"
        else:
            text_side = "left"

        flat_idx = int(np.argmin(complexity))
        free_row, free_col = flat_idx // 3, flat_idx % 3

        avg_brightness = brightness.mean()
        is_dark = avg_brightness < 120

        return {
            "text_side": text_side,
            "free_row": free_row,
            "free_col": free_col,
            "avg_brightness": avg_brightness,
            "is_dark": is_dark,
            "complexity_grid": complexity,
        }

    # --- Theme selection ---

    def _select_theme(self, content) -> str:
        if content.theme_name and content.theme_name in self.THEMES:
            return content.theme_name
        # Build a weighted pool: category-preferred themes appear more often,
        # but ALL themes are available — and a random pick ensures variety
        preferred = self.CATEGORY_THEMES.get(
            content.subcategory,
            self.CATEGORY_THEMES.get(content.category, [])
        )
        # Pool: preferred themes x3 weight + all remaining themes x1
        pool = list(preferred) * 3
        for t in self.THEMES:
            if t not in preferred:
                pool.append(t)
        theme = random.choice(pool)
        print(f"  Theme: {theme}")
        return theme

    def get_layout_hint(self, content) -> str:
        """Get layout hint for image generation based on theme."""
        # All themes now use full-bleed images with gradient overlays
        return "center"

    # --- Main compose entry point ---

    def compose(self, content) -> Image.Image:
        # Detect script from headline/features/CTA for correct font selection
        sample_text = " ".join([
            content.headline or "",
            content.tagline or "",
            " ".join(content.features[:3]),
            content.cta_text or "",
        ])
        self._script = self._detect_script(sample_text)

        # If no product image, use branded typography-focused ad
        if content.product_image is None:
            self._analysis = {"text_side": "left", "free_row": 0, "free_col": 2,
                              "avg_brightness": 60, "is_dark": True,
                              "complexity_grid": np.zeros((3, 3))}
            canvas = self._render_no_image_ad(content)
            return self._apply_post_processing(canvas)

        # AI analysis of product image for dynamic layout
        self._analysis = self._analyze_image(content.product_image)

        theme = self._select_theme(content)
        renderer = self._theme_renderers[theme]
        canvas = renderer(content)
        return self._apply_post_processing(canvas)

    # === NO-IMAGE AD: Professional branded typography ad ===
    def _render_no_image_ad(self, content) -> Image.Image:
        """When no product image is available, create a visually striking
        typography-focused ad that uses the ENTIRE canvas — no empty space."""
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)
        secondary = self._hex_to_rgb(content.secondary_color)
        dark_accent = tuple(max(0, c - 60) for c in accent)
        bright_accent = tuple(min(255, c + 60) for c in accent)

        # ── BACKGROUND: Rich diagonal gradient ──
        canvas = Image.new("RGBA", (W, H), (18, 18, 28, 255))
        draw = ImageDraw.Draw(canvas)
        for row in range(H):
            t = row / H
            r = int(dark_accent[0] * (1 - t) + accent[0] * 0.2 * t)
            g = int(dark_accent[1] * (1 - t) + accent[1] * 0.2 * t)
            b = int(dark_accent[2] * (1 - t) + accent[2] * 0.2 * t)
            draw.line([(0, row), (W, row)], fill=(max(0, min(255, r)),
                       max(0, min(255, g)), max(0, min(255, b)), 255))

        # ── DECORATIVE: Large geometric circle in upper-right ──
        circle_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        cd = ImageDraw.Draw(circle_layer)
        cr = int(W * 0.42)
        cx, cy = int(W * 0.78), int(H * 0.22)
        # Outer ring
        cd.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                   outline=(*accent, 35), width=2)
        # Filled circle (subtle)
        cd.ellipse([cx - cr + 30, cy - cr + 30, cx + cr - 30, cy + cr - 30],
                   fill=(*accent, 18))
        # Inner ring
        inner_r = int(cr * 0.55)
        cd.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
                   outline=(*bright_accent, 25), width=1)
        canvas = Image.alpha_composite(canvas, circle_layer)

        # ── DECORATIVE: Diagonal accent band across lower-right ──
        band_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bd = ImageDraw.Draw(band_layer)
        band_pts = [(W, int(H * 0.55)), (W, int(H * 0.65)),
                    (int(W * 0.35), H), (int(W * 0.25), H)]
        bd.polygon(band_pts, fill=(*accent, 22))
        # Second thinner band
        band_pts2 = [(W, int(H * 0.68)), (W, int(H * 0.72)),
                     (int(W * 0.50), H), (int(W * 0.45), H)]
        bd.polygon(band_pts2, fill=(*secondary, 18))
        canvas = Image.alpha_composite(canvas, band_layer)

        # ── DECORATIVE: Dot pattern in upper-left area ──
        dot_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dd = ImageDraw.Draw(dot_layer)
        dot_spacing = 30
        for dx in range(45, int(W * 0.3), dot_spacing):
            for dy in range(45, int(H * 0.25), dot_spacing):
                dd.ellipse([dx - 1, dy - 1, dx + 1, dy + 1], fill=(*accent, 30))
        canvas = Image.alpha_composite(canvas, dot_layer)

        # Subtle noise
        noise = self._generate_subtle_noise(W, H, intensity=5)
        canvas = Image.alpha_composite(canvas, noise)
        draw = ImageDraw.Draw(canvas)

        # ── LOGO: Top-left, clean, no backing ──
        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=60, padding=55, backing=False)
        draw = ImageDraw.Draw(canvas)

        margin_l = 65
        margin_r = 65
        text_w = W - margin_l - margin_r

        # ── BRAND NAME: Spaced uppercase label ──
        brand_display = content.brand_name.replace("_", " ").upper()
        brand_font = self._font("body", 14)
        self._draw_spaced_text(canvas, (margin_l, int(H * 0.20)), brand_display, brand_font,
                               fill=(*accent, 220), spacing=6)
        draw = ImageDraw.Draw(canvas)

        # Accent bar under brand name
        y = int(H * 0.24)
        draw.rounded_rectangle([margin_l, y, margin_l + 55, y + 4], radius=2,
                               fill=(*accent, 220))
        y += 28

        # ── HEADLINE: Large, bold, white — the main visual element ──
        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("headline", 54)
        lines = self._word_wrap(draw, headline, head_font, text_w)
        for line in lines[:3]:
            self._draw_text_shadow(draw, (margin_l, y), line, head_font,
                                   fill=(255, 255, 255, 255),
                                   shadow_color=(0, 0, 0, 80), offset=2)
            y += 64
        y += 12

        # ── TAGLINE: Accent-colored italic ──
        if content.tagline and content.tagline != content.headline:
            tag_font = self._font("accent", 21)
            draw.text((margin_l, y), content.tagline, fill=(*bright_accent, 220), font=tag_font)
            y += 34

        # ── FEATURES: Clean list with accent bullets, spread vertically ──
        feats = content.features[:5]
        if feats:
            y += 16
            feat_font = self._font("body", 18)
            for feat in feats:
                # Accent dot bullet
                draw.ellipse([margin_l, y + 7, margin_l + 8, y + 15],
                             fill=(*accent, 220))
                draw.text((margin_l + 22, y), feat,
                          fill=(225, 225, 235, 240), font=feat_font)
                y += 30

        # ── BILINGUAL: Secondary language text ──
        y += 8
        y = self._draw_secondary_text(canvas, content, y, margin_l, text_w,
                                      light_text=True, centered=False)
        draw = ImageDraw.Draw(canvas)

        # ── CTA BUTTON: Prominent, centered horizontally ──
        cta = self._get_cta_text(content)
        cta_font = self._font("headline", 22)
        ctw, _ = self._text_size(draw, cta, cta_font)
        btn_w = max(ctw + 70, 220)
        cta_y = max(y + 20, H - 160)
        cta_x = (W - btn_w) // 2
        canvas, btn_h = self._draw_cta_button(canvas, cta,
                                              (cta_x, cta_y), accent,
                                              style="rounded", font_size=22)

        # ── BOTTOM: Brand domain / website bar ──
        draw = ImageDraw.Draw(canvas)
        domain = content.brand_domain or f"www.{content.brand_name.lower().replace('_', '').replace(' ', '')}.com"
        url_font = self._font("body", 13)
        uw, _ = self._text_size(draw, domain, url_font)
        draw.text(((W - uw) // 2, H - 45), domain, fill=(*accent, 150), font=url_font)

        # (no decorative overlays — clean professional look)

        # ── BOTTOM ACCENT BAR ──
        draw = ImageDraw.Draw(canvas)
        for bx in range(W):
            t = bx / W
            c = tuple(int(accent[i] + (secondary[i] - accent[i]) * t) for i in range(3))
            draw.line([(bx, H - 5), (bx, H)], fill=(*c, 220))

        return canvas

    # === THEME 1: Clean Wave — Product hero top, wave divider, text bottom ===
    def _render_minimal_clean(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)
        darker = tuple(max(0, c - 40) for c in accent)

        # Solid dark background (visible below wave)
        canvas = Image.new("RGBA", (W, H), (*darker, 255))

        # Product image fills top 65%
        prod_h = int(H * 0.65)
        if content.product_image:
            prod = self._cover_fill(content.product_image, W, prod_h)
            canvas.paste(prod.convert("RGBA"), (0, 0))

        # Wave divider with solid filled bottom
        wave_y = prod_h - 20
        canvas = self._draw_wave_divider(canvas, wave_y, (255, 255, 255, 200),
                                          amplitude=25, thickness=3,
                                          fill_below=(*darker, 252))

        draw = ImageDraw.Draw(canvas)
        margin = 60
        text_w = W - 2 * margin

        # Logo top-left on product area
        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=55, padding=30, backing=False)
        draw = ImageDraw.Draw(canvas)

        # Headline — short tagline, centered, 1 line
        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("headline", 36)
        lines = self._word_wrap(draw, headline, head_font, text_w)
        y = wave_y + 50
        for line in lines[:2]:
            tw, _ = self._text_size(draw, line, head_font)
            draw.text(((W - tw) // 2, y), line, fill=(255, 255, 255, 255), font=head_font)
            y += 44

        # Features — single centered line
        y += 12
        feat_font = self._font("body", 14)
        feat_text = "  \u2022  ".join(content.features[:3])
        ftw, _ = self._text_size(draw, feat_text, feat_font)
        draw.text(((W - ftw) // 2, y), feat_text, fill=(200, 200, 210, 200), font=feat_font)
        y += 28

        # CTA button centered
        cta = self._get_cta_text(content)
        cta_font = self._font("headline", 18)
        ctw, _ = self._text_size(draw, cta, cta_font)
        btn_w = max(ctw + 50, 180)
        canvas, _ = self._draw_cta_button(canvas, cta,
                                          ((W - btn_w) // 2, min(y + 8, H - 65)),
                                          accent, style="rounded", font_size=18)

        return canvas

    # === THEME 2: Bold Hero — Product top, dark bottom, bold headline ===
    def _render_bold_hero(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)

        # Dark background
        canvas = Image.new("RGBA", (W, H), (15, 15, 22, 255))

        # Product fills top 65%
        prod_h = int(H * 0.65)
        if content.product_image:
            prod = self._cover_fill(content.product_image, W, prod_h)
            canvas.paste(prod.convert("RGBA"), (0, 0))

        # Wave divider — accent colored wave, dark bottom
        wave_y = prod_h - 20
        canvas = self._draw_wave_divider(canvas, wave_y, (*accent, 220),
                                          amplitude=30, thickness=3,
                                          fill_below=(15, 15, 22, 252))

        draw = ImageDraw.Draw(canvas)
        margin = 60

        # Logo top-left
        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=50, padding=30, backing=False)
        draw = ImageDraw.Draw(canvas)

        # Headline — bold uppercase, left-aligned
        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("display", 40)
        lines = self._word_wrap(draw, headline.upper(), head_font, W - 2 * margin)
        y = wave_y + 48
        for line in lines[:2]:
            draw.text((margin, y), line, fill=(255, 255, 255, 255), font=head_font)
            y += 48

        # Features — single line
        y += 10
        feat_font = self._font("body", 14)
        feat_text = "  \u2022  ".join(content.features[:3])
        draw.text((margin, y), feat_text, fill=(180, 180, 190, 200), font=feat_font)
        y += 28

        # CTA button
        canvas, _ = self._draw_cta_button(canvas, self._get_cta_text(content),
                                          (margin, min(y + 6, H - 65)), accent,
                                          style="rounded", font_size=18)

        # Bottom accent bar
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, H - 4, W, H], fill=(*accent, 220))

        return canvas

    # === THEME 3: Premium Dark — Product top with gradient fade, elegant text below ===
    def _render_premium_dark(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)
        if sum(accent) < 200:
            accent = (212, 175, 55)

        # Rich dark canvas
        canvas = Image.new("RGBA", (W, H), (12, 12, 18, 255))

        # Product image fills top 60%
        prod_h = int(H * 0.60)
        if content.product_image:
            prod = self._cover_fill(content.product_image, W, prod_h)
            # Gradient fade at bottom of product image — smooth blend into dark
            prod_rgba = prod.convert("RGBA")
            fade = Image.new("RGBA", (W, prod_h), (0, 0, 0, 0))
            fd = ImageDraw.Draw(fade)
            fade_start = int(prod_h * 0.65)
            for row in range(fade_start, prod_h):
                t = (row - fade_start) / (prod_h - fade_start)
                fd.line([(0, row), (W, row)], fill=(12, 12, 18, int(255 * t)))
            prod_rgba = Image.alpha_composite(prod_rgba, fade)
            canvas.paste(prod_rgba, (0, 0))

        draw = ImageDraw.Draw(canvas)

        # Thin accent line divider
        div_y = int(H * 0.62)
        line_w = int(W * 0.4)
        draw.line([((W - line_w) // 2, div_y), ((W + line_w) // 2, div_y)],
                  fill=(*accent, 160), width=1)

        # Logo top-left
        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=50, padding=30, backing=False)
        draw = ImageDraw.Draw(canvas)

        # Spaced brand name — small, elegant, centered
        brand_display = content.brand_name.replace("_", " ").upper()
        brand_font = self._font("body", 12)
        canvas = self._draw_spaced_text(canvas, None, brand_display, brand_font,
                                        fill=(*accent, 200), spacing=5,
                                        center_x=W // 2, y_pos=div_y + 14)
        draw = ImageDraw.Draw(canvas)

        # Headline — elegant, centered
        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("elegant", 36)
        lines = self._word_wrap(draw, headline, head_font, W - 140)
        y = div_y + 40
        for line in lines[:2]:
            tw, _ = self._text_size(draw, line, head_font)
            draw.text(((W - tw) // 2, y), line, fill=(240, 240, 245, 255), font=head_font)
            y += 44

        # Features — single centered line
        y += 10
        feat_font = self._font("body", 13)
        feat_text = "  \u2022  ".join(content.features[:3])
        ftw, _ = self._text_size(draw, feat_text, feat_font)
        draw.text(((W - ftw) // 2, y), feat_text, fill=(170, 170, 180, 190), font=feat_font)
        y += 26

        # CTA — outline style, centered
        cta = self._get_cta_text(content)
        cta_font = self._font("headline", 18)
        ctw, _ = self._text_size(draw, cta, cta_font)
        btn_w = max(ctw + 56, 180)
        canvas, _ = self._draw_cta_button(canvas, cta,
                                          ((W - btn_w) // 2, min(y + 10, H - 60)),
                                          accent, style="outline", font_size=18)

        return canvas

    # === THEME 4: Side-by-Side — Product left, clean text right ===
    def _render_split_layout(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)
        split_x = int(W * 0.52)

        # Right side — clean accent gradient
        darker = tuple(max(0, c - 50) for c in accent)
        canvas = Image.new("RGBA", (W, H), (*accent, 255))
        draw = ImageDraw.Draw(canvas)
        for row in range(H):
            t = row / H
            rgb = tuple(int(accent[i] + (darker[i] - accent[i]) * t) for i in range(3))
            draw.line([(split_x, row), (W, row)], fill=(*rgb, 255))

        # Left side — product image fills
        if content.product_image:
            prod = self._cover_fill(content.product_image, split_x, H)
            canvas.paste(prod.convert("RGBA"), (0, 0))
        else:
            lighter = tuple(min(255, c + 60) for c in accent)
            for row in range(H):
                t = row / H
                rgb = tuple(int(lighter[i] * (1 - t * 0.3)) for i in range(3))
                draw.line([(0, row), (split_x, row)], fill=(*rgb, 255))

        # Soft gradient blend at the split edge
        blend_w = 40
        blend = Image.new("RGBA", (blend_w, H), (0, 0, 0, 0))
        bd = ImageDraw.Draw(blend)
        for bx in range(blend_w):
            t = bx / blend_w
            alpha = int(255 * (t ** 0.8))
            bd.line([(bx, 0), (bx, H)], fill=(*accent, alpha))
        canvas.paste(blend, (split_x - blend_w, 0), blend)

        draw = ImageDraw.Draw(canvas)
        lum = accent[0] * 0.299 + accent[1] * 0.587 + accent[2] * 0.114
        text_color = (255, 255, 255, 255) if lum < 140 else (20, 20, 30, 255)
        margin_r = split_x + 35
        text_max_w = W - margin_r - 35

        # Logo top-left on product side
        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=45, padding=25, backing=False)
        draw = ImageDraw.Draw(canvas)

        # Headline on right — vertically centered
        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("headline", 32)
        lines = self._word_wrap(draw, headline, head_font, text_max_w)
        y = int(H * 0.28)
        for line in lines[:2]:
            draw.text((margin_r, y), line, fill=text_color, font=head_font)
            y += 40

        # Accent bar
        y += 10
        draw.rounded_rectangle([margin_r, y, margin_r + 45, y + 3], radius=2,
                               fill=(*text_color[:3], 180))
        y += 24

        # Features — stacked
        feat_font = self._font("body", 14)
        for feat in content.features[:3]:
            draw.text((margin_r, y), f"\u2022 {feat}",
                      fill=(*text_color[:3], 200), font=feat_font)
            y += 24

        # CTA button
        y += 14
        btn_color = (255, 255, 255) if lum < 140 else accent
        canvas, _ = self._draw_cta_button(canvas, self._get_cta_text(content),
                                          (margin_r, min(y, H - 80)), btn_color,
                                          style="rounded", font_size=16)

        return canvas

    # === THEME 5: Product Showcase — Product top with wave, accent bottom ===
    def _render_card_float(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)

        # Accent gradient background
        canvas = Image.new("RGBA", (W, H), (*accent, 255))
        draw = ImageDraw.Draw(canvas)
        darker = tuple(max(0, c - 50) for c in accent)
        for row in range(H):
            t = row / H
            rgb = tuple(int(accent[i] + (darker[i] - accent[i]) * t * 0.5) for i in range(3))
            draw.line([(0, row), (W, row)], fill=(*rgb, 255))

        # Product fills top 62%
        prod_h = int(H * 0.62)
        if content.product_image:
            prod = self._cover_fill(content.product_image, W, prod_h)
            canvas.paste(prod.convert("RGBA"), (0, 0))

        # Wave divider — white wave, accent fill below
        wave_y = prod_h - 18
        canvas = self._draw_wave_divider(canvas, wave_y, (255, 255, 255, 200),
                                          amplitude=22, thickness=3,
                                          fill_below=(*accent, 252))

        draw = ImageDraw.Draw(canvas)
        margin = 60

        # Logo top-left
        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=50, padding=30, backing=False)
        draw = ImageDraw.Draw(canvas)

        # Headline centered
        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("headline", 34)
        lum = accent[0] * 0.299 + accent[1] * 0.587 + accent[2] * 0.114
        head_color = (255, 255, 255, 255) if lum < 140 else (20, 20, 30, 255)
        text_w = W - 2 * margin
        lines = self._word_wrap(draw, headline, head_font, text_w)
        y = wave_y + 48
        for line in lines[:2]:
            tw, _ = self._text_size(draw, line, head_font)
            draw.text(((W - tw) // 2, y), line, fill=head_color, font=head_font)
            y += 42

        # Features — single centered line
        y += 10
        feat_font = self._font("body", 13)
        feat_color = (*head_color[:3], 190)
        feat_text = "  \u2022  ".join(content.features[:3])
        ftw, _ = self._text_size(draw, feat_text, feat_font)
        draw.text(((W - ftw) // 2, y), feat_text, fill=feat_color, font=feat_font)
        y += 26

        # CTA centered
        cta = self._get_cta_text(content)
        cta_font = self._font("headline", 17)
        ctw, _ = self._text_size(draw, cta, cta_font)
        btn_w = max(ctw + 50, 170)
        btn_color = (255, 255, 255) if lum < 140 else accent
        canvas, _ = self._draw_cta_button(canvas, cta,
                                          ((W - btn_w) // 2, min(y + 10, H - 60)),
                                          btn_color, style="rounded", font_size=17)

        return canvas

    # === THEME 6: Vibrant Mesh — Product top, colorful mesh bottom ===
    def _render_gradient_mesh(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)
        secondary = self._hex_to_rgb(content.secondary_color)

        # Dark canvas with vibrant mesh (visible below wave)
        canvas = Image.new("RGBA", (W, H), (20, 20, 35, 255))
        mesh_arr = np.zeros((H, W, 4), dtype=float)
        Y_grid, X_grid = np.ogrid[:H, :W]
        for cx, cy, color, rf in [
            (W * 0.2, H * 0.78, accent, 0.5),
            (W * 0.8, H * 0.85, secondary, 0.45),
            (W * 0.5, H * 0.92, tuple(max(0, int(c * 0.8)) for c in accent), 0.55),
        ]:
            radius = max(W, H) * rf
            dist = np.sqrt((X_grid - cx) ** 2 + (Y_grid - cy) ** 2)
            falloff = np.clip(1 - dist / radius, 0, 1) ** 2
            for c_idx in range(3):
                mesh_arr[:, :, c_idx] += color[c_idx] * falloff * 0.4
            mesh_arr[:, :, 3] += 120 * falloff * 0.5
        mesh_arr = np.clip(mesh_arr, 0, 255).astype(np.uint8)
        canvas = Image.alpha_composite(canvas, Image.fromarray(mesh_arr, "RGBA"))

        # Product fills top 65%
        prod_h = int(H * 0.65)
        if content.product_image:
            prod = self._cover_fill(content.product_image, W, prod_h)
            canvas.paste(prod.convert("RGBA"), (0, 0))

        # Wave divider with dark fill
        wave_y = prod_h - 18
        canvas = self._draw_wave_divider(canvas, wave_y, (*accent, 180),
                                          amplitude=25, thickness=3,
                                          fill_below=(20, 20, 35, 235))

        draw = ImageDraw.Draw(canvas)
        margin = 60

        # Logo top-left
        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=50, backing=False)
        draw = ImageDraw.Draw(canvas)

        # Headline — left-aligned with text shadow
        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("headline", 36)
        lines = self._word_wrap(draw, headline, head_font, W - 2 * margin)
        y = wave_y + 48
        for line in lines[:2]:
            self._draw_text_shadow(draw, (margin, y), line, head_font,
                                   fill=(255, 255, 255, 255))
            y += 44

        # Features — single line
        y += 10
        feat_font = self._font("body", 14)
        feat_text = "  \u2022  ".join(content.features[:3])
        draw.text((margin, y), feat_text, fill=(200, 200, 215, 200), font=feat_font)
        y += 28

        # CTA
        canvas, _ = self._draw_cta_button(canvas, self._get_cta_text(content),
                                          (margin, min(y + 6, H - 60)), accent,
                                          style="rounded", font_size=18)

        # Bottom accent bar
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, H - 4, W, H], fill=(*accent, 220))

        return canvas

PamphletComposer = ProAdDesigner  # backward compatibility


# ---------------------------------------------------------------------------
# Database (SQLite - no external dependencies)
# ---------------------------------------------------------------------------

class Database:
    def __init__(self):
        self.db_path = DB_DIR / "adcraft.db"
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                price TEXT,
                category TEXT,
                brand TEXT,
                image_paths TEXT,
                enhanced_image_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generated_content (
                id TEXT PRIMARY KEY,
                product_id TEXT,
                content_type TEXT,
                language TEXT,
                content_json TEXT,
                pamphlet_path TEXT,
                product_image_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clicks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT,
                platform TEXT,
                source TEXT,
                clicked_at TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        """)
        conn.commit()
        conn.close()

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def create_product(self, name: str, description: str = "",
                       price: str = "", category: str = "",
                       brand: str = "", image_paths: list = None) -> str:
        product_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        conn = self._conn()
        conn.execute(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (product_id, name, description, price, category, brand,
             json.dumps(image_paths or []), None, now, now)
        )
        conn.commit()
        conn.close()
        return product_id

    def get_product(self, product_id: str) -> Optional[dict]:
        conn = self._conn()
        row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
        conn.close()
        if row:
            d = dict(row)
            d["image_paths"] = json.loads(d["image_paths"])
            return d
        return None

    def list_products(self) -> List[dict]:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM products ORDER BY created_at DESC").fetchall()
        conn.close()
        products = []
        for row in rows:
            d = dict(row)
            d["image_paths"] = json.loads(d["image_paths"])
            products.append(d)
        return products

    def delete_product(self, product_id: str):
        conn = self._conn()
        conn.execute("DELETE FROM generated_content WHERE product_id = ?", (product_id,))
        conn.execute("DELETE FROM clicks WHERE product_id = ?", (product_id,))
        conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
        conn.commit()
        conn.close()

    def save_generated_content(self, product_id: str, content_type: str,
                                language: str, content: dict,
                                pamphlet_path: str = None,
                                product_image_path: str = None) -> str:
        content_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        conn = self._conn()
        conn.execute(
            "INSERT INTO generated_content VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (content_id, product_id, content_type, language,
             json.dumps(content, ensure_ascii=False), pamphlet_path,
             product_image_path, now)
        )
        conn.commit()
        conn.close()
        return content_id

    def get_product_content(self, product_id: str) -> List[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM generated_content WHERE product_id = ? ORDER BY created_at DESC",
            (product_id,)
        ).fetchall()
        conn.close()
        results = []
        for row in rows:
            d = dict(row)
            d["content_json"] = json.loads(d["content_json"])
            results.append(d)
        return results

    def track_click(self, product_id: str, platform: str, source: str = ""):
        now = datetime.now().isoformat()
        conn = self._conn()
        conn.execute(
            "INSERT INTO clicks (product_id, platform, source, clicked_at) VALUES (?, ?, ?, ?)",
            (product_id, platform, source, now)
        )
        conn.commit()
        conn.close()

    def get_analytics(self, product_id: str) -> dict:
        conn = self._conn()
        rows = conn.execute(
            "SELECT platform, COUNT(*) as count FROM clicks WHERE product_id = ? GROUP BY platform",
            (product_id,)
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM clicks WHERE product_id = ?", (product_id,)
        ).fetchone()[0]
        conn.close()
        platforms = {row["platform"]: row["count"] for row in rows}
        return {"total_clicks": total, "by_platform": platforms}


# ---------------------------------------------------------------------------
# Main Pipeline (orchestrates everything)
# ---------------------------------------------------------------------------

class AdCraftPipeline:
    _instance = None

    def __init__(self):
        print("\n" + "=" * 60)
        print("  AdCraft AI - Loading Pipeline")
        print("=" * 60)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {self.device}")

        self._load_models()

        # HF token for FLUX.1-schnell inference (same as RAGPipeline.py)
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.image_gen = ImageGenerator(hf_token=self.hf_token or None)

        self.color_extractor = ColorExtractor()
        self.ad_designer = ProAdDesigner()
        self.pamphlet_composer = self.ad_designer  # backward compat
        self.local_image_gen = LocalAdImageGenerator()
        self.content_gen = ContentGenerator(self.clip_content_extractor)
        self.translator = Translator()
        self.enhancer = ImageEnhancer()
        self.db = Database()

        print("\n  Pipeline ready!")
        print("=" * 60)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_models(self):
        clip_name = "openai/clip-vit-base-patch32"
        print("  Loading CLIP model...", end=" ", flush=True)
        try:
            self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
        except OSError:
            self.clip_model = CLIPModel.from_pretrained(clip_name, local_files_only=True).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_name, local_files_only=True)
        self.clip_model.eval()
        print("done")

        print("  Loading FAISS index...", end=" ", flush=True)
        index_path = FAISS_DIR / "madverse_index.faiss"
        metadata_path = FAISS_DIR / "id_to_metadata.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run BuildFAISS.py first.")
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self.id_to_metadata = pickle.load(f)
        print(f"done ({self.index.ntotal:,} vectors)")

        print("  Building brand index...", end=" ", flush=True)
        self.brand_matcher = BrandMatcher(self.id_to_metadata)
        print(f"done ({len(self.brand_matcher.brand_names)} brands)")

        self.clip_content_extractor = CLIPContentExtractor(
            self.clip_model, self.clip_processor, self.device
        )

    # ------------------------------------------------------------------
    # Stage 0: Brand Matching
    # ------------------------------------------------------------------
    def match_brand(self, query: str) -> BrandMatch:
        return self.brand_matcher.match(query)

    # ------------------------------------------------------------------
    # Stage 1: FAISS Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, brand_match: BrandMatch, k: int = 5) -> List[RetrievedAd]:
        inputs = self.clip_processor(text=query, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k_: v.to(self.device) for k_, v in inputs.items()
                       if k_ in ('input_ids', 'attention_mask')}
        with torch.no_grad():
            text_out = self.clip_model.text_model(**text_inputs)
            pooled = text_out[1]
            features = self.clip_model.text_projection(pooled)
            features = F.normalize(features, p=2, dim=-1)
        query_vec = features.cpu().numpy().flatten().astype(np.float32)

        if brand_match.matched_brand:
            brand_indices = self.brand_matcher.get_brand_indices(brand_match.matched_brand)
            if brand_indices:
                vectors = np.array(
                    [self.index.reconstruct(int(i)) for i in brand_indices],
                    dtype=np.float32,
                )
                sub_index = faiss.IndexFlatL2(self.index.d)
                sub_index.add(vectors)
                actual_k = min(k, len(brand_indices))
                distances, sub_indices = sub_index.search(np.array([query_vec]), actual_k)
                results = []
                for rank, (dist, sub_idx) in enumerate(zip(distances[0], sub_indices[0]), 1):
                    if sub_idx < 0:
                        continue
                    original_idx = brand_indices[int(sub_idx)]
                    meta = self.id_to_metadata[original_idx]
                    results.append(RetrievedAd(
                        rank=rank, similarity=float(1 / (1 + float(dist))),
                        distance=float(dist),
                        image_path=meta.get("image_path", ""),
                        image_id=meta.get("image_id", ""),
                        brand=meta.get("brand", ""),
                        category=meta.get("category", ""),
                        subcategory=meta.get("subcategory", ""),
                        language=meta.get("language", ""),
                        ad_type=meta.get("ad_type", ""),
                        source=meta.get("source", ""),
                    ))
                return results

        distances, indices = self.index.search(np.array([query_vec]), k)
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            if 0 <= idx < len(self.id_to_metadata):
                meta = self.id_to_metadata[idx]
                results.append(RetrievedAd(
                    rank=rank, similarity=float(1 / (1 + float(dist))),
                    distance=float(dist),
                    image_path=meta.get("image_path", ""),
                    image_id=meta.get("image_id", ""),
                    brand=meta.get("brand", ""),
                    category=meta.get("category", ""),
                    subcategory=meta.get("subcategory", ""),
                    language=meta.get("language", ""),
                    ad_type=meta.get("ad_type", ""),
                    source=meta.get("source", ""),
                ))
        return results

    # ------------------------------------------------------------------
    # Full Generation Pipeline
    # ------------------------------------------------------------------
    def generate(self, query: str, languages: List[str] = None,
                 uploaded_image: Image.Image = None,
                 product_id: str = None,
                 brand_override: str = None) -> GenerationResult:
        if languages is None:
            languages = ["en"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = GenerationResult(
            query=query,
            timestamp=datetime.now().isoformat(),
            product_id=product_id,
        )
        total_start = time.time()

        try:
            # Stage 0: Brand matching
            t0 = time.time()
            brand_match = self.match_brand(query)
            result.stage_timings["brand_matching"] = time.time() - t0
            if brand_match.matched_brand:
                result.brand_match = {
                    "matched_brand": brand_match.matched_brand,
                    "confidence": brand_match.confidence,
                    "category": brand_match.category,
                    "subcategory": brand_match.subcategory,
                    "image_count": brand_match.image_count,
                }

            # Stage 1: Retrieval
            t0 = time.time()
            retrieved_ads = self.retrieve(query, brand_match, k=5)
            result.stage_timings["retrieval"] = time.time() - t0
            result.retrieved_ads = [
                {"rank": ad.rank, "similarity": ad.similarity,
                 "brand": ad.brand, "category": ad.category,
                 "subcategory": ad.subcategory, "image_path": ad.image_path}
                for ad in retrieved_ads
            ]

            if not retrieved_ads:
                result.errors.append("No ads retrieved from dataset")
                return result

            # Stage 2: Color extraction
            t0 = time.time()
            ad_image_paths = [ad.image_path for ad in retrieved_ads if Path(ad.image_path).exists()]
            colors = self.color_extractor.extract_from_images(ad_image_paths)
            result.stage_timings["color_extraction"] = time.time() - t0
            result.extracted_colors = colors

            # Stage 3: Content generation (CLIP RAG + Pollinations text)
            t0 = time.time()
            # Use user-specified brand if provided, otherwise auto-detect
            if brand_override:
                brand = brand_override.replace(" ", "_")
            else:
                brand = brand_match.matched_brand or (retrieved_ads[0].brand if retrieved_ads else "Product")
            category = brand_match.category or (retrieved_ads[0].category if retrieved_ads else "product")
            subcategory = brand_match.subcategory or (retrieved_ads[0].subcategory if retrieved_ads else "")

            accent = ColorExtractor.get_accent_color(colors)
            secondary = colors[1] if len(colors) > 1 else accent
            bg_tint = ColorExtractor.lighten_color(accent, factor=0.92)

            # CLIP-based feature extraction (category-aware to avoid cross-category contamination)
            features = self.clip_content_extractor.extract_features(ad_image_paths, n=6, category=category)
            diffusion_prompt = self.clip_content_extractor.generate_diffusion_prompt(
                ad_image_paths, brand, category, subcategory, user_query=query
            )

            # Try AI-enhanced prompt via Pollinations text API for better image quality
            try:
                brand_clean = brand.replace("_", " ")
                ai_prompt = self.text_gen.generate(
                    system_prompt=(
                        "You are an expert advertising photographer prompt writer. "
                        "Given a brand and product description, write a single detailed "
                        "image generation prompt (2-3 sentences) describing a photorealistic "
                        "advertisement photo. Include specific details about the scene, "
                        "models, lighting, colors, setting, and mood. "
                        "Output ONLY the prompt text, nothing else."
                    ),
                    user_prompt=(
                        f"Brand: {brand_clean}\n"
                        f"Category: {category}\n"
                        f"User request: {query}\n"
                        f"Write a detailed photorealistic image prompt for this ad:"
                    ),
                )
                if ai_prompt and len(ai_prompt) > 30:
                    diffusion_prompt = (
                        f"{ai_prompt.strip()}, "
                        f"ultra realistic, photorealistic, 8k, sharp focus, "
                        f"professional commercial advertisement photography"
                    )
                    print(f"  AI-enhanced prompt: {diffusion_prompt[:120]}...")
            except Exception:
                pass  # Fall back to CLIP-based prompt

            # Tagline
            tagline = BRAND_TAGLINES.get(brand, f"Experience {brand.replace('_', ' ')}")

            # If user uploaded an image, analyze it too
            if uploaded_image:
                img_analysis = self.clip_content_extractor.analyze_uploaded_image(uploaded_image, category=category)
                features = img_analysis["features"]

            # Generate text content (descriptions, captions)
            text_content = self.content_gen.generate_product_content(
                brand, category, subcategory, features, tagline, query
            )
            result.product_title = text_content.get("product_title", "")
            result.product_description = text_content.get("product_description", "")
            result.instagram_caption = text_content.get("instagram_caption", "")
            result.whatsapp_copy = text_content.get("whatsapp_copy", "")
            result.hashtags = text_content.get("hashtags", [])

            # Logo (with transparent background)
            logo_image = None
            try:
                logo_image = ProLogoFetcher.fetch(brand)
            except Exception:
                pass

            # Get brand domain for URL display
            brand_domain = ""
            try:
                domain = ProLogoFetcher._get_domain(brand)
                brand_domain = f"www.{domain}"
            except Exception:
                pass

            # Build headline — use short tagline for the ad visual, not the verbose product_title
            headline = tagline or brand.replace("_", " ")

            # Thumbnails
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

            content = AdContent(
                brand_name=brand, headline=headline, tagline=tagline,
                features=features[:6],
                accent_color=accent, secondary_color=secondary,
                background_tint=bg_tint, logo_image=logo_image,
                thumbnail_images=thumbnails, thumbnail_paths=thumb_paths,
                diffusion_prompt=diffusion_prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                category=category, subcategory=subcategory,
                brand_domain=brand_domain,
            )

            result.content = {
                "brand": brand, "tagline": tagline, "features": features,
                "accent_color": accent, "diffusion_prompt": diffusion_prompt,
            }
            result.stage_timings["content_generation"] = time.time() - t0

            # Stage 4: Image Generation — exact same as RAGPipeline.py
            # Pollinations FLUX → HuggingFace Inference FLUX.1-schnell → gradient fallback
            t0 = time.time()
            if uploaded_image:
                product_img = self.enhancer.auto_enhance(uploaded_image)
                result.image_generator_used = "uploaded"
            else:
                print("\n  Stage 4: Image Generation")
                product_img, method = self.image_gen.generate(
                    prompt=diffusion_prompt,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    width=1024,
                    height=768,
                )

                # Use brand colors for gradient fallback
                if method == "gradient_fallback" and colors:
                    product_img = self.image_gen._make_gradient(1024, 768, colors)

                result.image_generator_used = method
                print(f"  Generated via: {method}")
                print(f"  Size: {product_img.size}")

            if product_img:
                content.product_image = product_img
                prod_path = str(OUTPUT_DIR / f"product_{timestamp}.png")
                product_img.save(prod_path, quality=95)
                result.product_image_path = prod_path

            result.stage_timings["image_generation"] = time.time() - t0

            # Stage 5: Multi-language translations FIRST (needed for ad text)
            t0 = time.time()
            result.languages_generated = ["en"]
            result.translations["en"] = {
                "product_title": result.product_title,
                "product_description": result.product_description,
                "instagram_caption": result.instagram_caption,
                "whatsapp_copy": result.whatsapp_copy,
                "tagline": tagline,
            }

            for lang in languages:
                if lang == "en":
                    continue
                try:
                    lang_content = self.content_gen.generate_ad_content_for_language(
                        brand, category, subcategory, features, tagline, query, lang
                    )
                    if lang_content:
                        result.translations[lang] = lang_content
                        result.languages_generated.append(lang)
                        continue
                except Exception:
                    pass

                try:
                    translated = {
                        "product_title": self.translator.translate(result.product_title, lang),
                        "product_description": self.translator.translate(result.product_description, lang),
                        "instagram_caption": self.translator.translate(result.instagram_caption, lang),
                        "whatsapp_copy": self.translator.translate(result.whatsapp_copy, lang),
                        "tagline": self.translator.translate(tagline, lang),
                    }
                    result.translations[lang] = translated
                    result.languages_generated.append(lang)
                except Exception as e:
                    result.errors.append(f"Translation to {lang} failed: {str(e)}")

            result.stage_timings["translations"] = time.time() - t0

            # Stage 6: Ad composition — BILINGUAL: show both languages on the ad
            t0 = time.time()
            # Determine the primary display language (first non-en if available, else en)
            primary_lang = "en"
            for lang in languages:
                if lang != "en" and lang in result.translations:
                    primary_lang = lang
                    break

            # Save English text as secondary BEFORE translating
            english_headline = content.headline
            english_features = list(content.features)
            english_cta = self.ad_designer._get_cta_text(content)

            # Update ad content text to use the primary language
            if primary_lang != "en" and primary_lang in result.translations:
                t = result.translations[primary_lang]
                # Handle both key formats:
                #   Pollinations returns: {tagline, description, caption, cta}
                #   Translator returns:  {product_title, product_description, ...}
                translated_headline = (
                    t.get("product_title", "")
                    or t.get("tagline", "")
                    or t.get("description", "")
                )
                if translated_headline:
                    content.headline = translated_headline
                content.tagline = t.get("tagline", "") or content.tagline

                # Translate CTA (check both key formats)
                translated_cta = t.get("cta", "")
                if translated_cta:
                    content.cta_text = translated_cta
                else:
                    try:
                        content.cta_text = self.translator.translate(english_cta, primary_lang) or english_cta
                    except Exception:
                        pass

                # Translate features
                translated_features = []
                for feat in content.features[:6]:
                    try:
                        tf = self.translator.translate(feat, primary_lang)
                        translated_features.append(tf if tf else feat)
                    except Exception:
                        translated_features.append(feat)
                if translated_features:
                    content.features = translated_features

                # Store English as SECONDARY language for bilingual rendering
                # Only if English is also in the selected languages
                if "en" in languages:
                    content.headline_secondary = english_headline
                    content.features_secondary = english_features[:4]
                    content.cta_secondary = english_cta

            pamphlet = self.ad_designer.compose(content)
            pamphlet_path = str(OUTPUT_DIR / f"pamphlet_{timestamp}.png")
            pamphlet.save(pamphlet_path, quality=95)
            result.pamphlet_path = pamphlet_path
            result.stage_timings["composition"] = time.time() - t0

            # Save to database
            if product_id:
                self.db.save_generated_content(
                    product_id, "full_generation", "multi",
                    {
                        "product_title": result.product_title,
                        "product_description": result.product_description,
                        "instagram_caption": result.instagram_caption,
                        "whatsapp_copy": result.whatsapp_copy,
                        "hashtags": result.hashtags,
                        "translations": result.translations,
                    },
                    pamphlet_path=result.pamphlet_path,
                    product_image_path=result.product_image_path,
                )

            result.stage_timings["total"] = time.time() - total_start

            # Save result JSON
            json_path = OUTPUT_DIR / f"result_{timestamp}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            result.errors.append(f"Pipeline error: {str(e)}")
            traceback.print_exc()

        return result

    # ------------------------------------------------------------------
    # Individual feature methods
    # ------------------------------------------------------------------
    def enhance_image(self, image: Image.Image) -> Image.Image:
        return self.enhancer.auto_enhance(image)

    def generate_description(self, query: str, image: Image.Image = None) -> dict:
        brand_match = self.match_brand(query)
        brand = brand_match.matched_brand or "Product"
        category = brand_match.category or "general"
        subcategory = brand_match.subcategory or ""
        tagline = BRAND_TAGLINES.get(brand, "")

        if image:
            analysis = self.clip_content_extractor.analyze_uploaded_image(image)
            features = analysis["features"]
        else:
            retrieved = self.retrieve(query, brand_match, k=3)
            paths = [ad.image_path for ad in retrieved if Path(ad.image_path).exists()]
            features = self.clip_content_extractor.extract_features(paths, n=6)

        return self.content_gen.generate_product_content(
            brand, category, subcategory, features, tagline, query
        )

    def generate_captions(self, query: str, languages: List[str] = None) -> dict:
        if languages is None:
            languages = ["en"]

        desc = self.generate_description(query)
        result = {"en": desc}

        for lang in languages:
            if lang == "en":
                continue
            try:
                translated = self.translator.translate_content(desc, lang)
                result[lang] = translated
            except Exception:
                result[lang] = desc

        return result
