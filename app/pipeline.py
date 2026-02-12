"""
AdCraft AI Pipeline - RAG-Based Ad Generation (No API Keys Required)

Uses:
  - CLIP (local) for embeddings and content extraction
  - FAISS (local) for vector search
  - Pollinations.ai (free, no API key) for image generation
  - Pollinations.ai (free, no API key) for text generation
  - deep_translator (free) for multi-language support
  - PIL (local) for image composition and enhancement
"""

import difflib
import faiss
import io
import json
import numpy as np
import os
import pickle
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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from sklearn.cluster import KMeans
from transformers import CLIPModel, CLIPProcessor

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
    "text, words, letters, numbers, typography, watermark, logo, label, "
    "blurry, low quality, distorted, deformed, ugly, oversaturated, "
    "extra fingers, extra limbs, disfigured face, bad anatomy, "
    "poorly drawn hands, poorly drawn face, mutation"
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

class WebLogoFetcher:
    BRAND_DOMAINS = {
        "Nike": "nike.com", "Adidas": "adidas.com", "Puma": "puma.com",
        "Reebok": "reebok.com", "Skechers": "skechers.com", "FILA": "fila.com",
        "Bata": "bata.in", "HRX": "hrx.in", "Bisleri": "bisleri.com",
        "Coca_Cola": "coca-cola.com", "Pepsi": "pepsi.com", "Amul": "amul.com",
        "Patanjali": "patanjaliayurved.net", "Samsung_mobiles": "samsung.com",
        "Apple_mobiles": "apple.com", "Motorola_mobiles": "motorola.com",
        "One-Plus_mobiles": "oneplus.com", "maruti_suzuki": "marutisuzuki.com",
        "tata": "tata.com", "mahindra": "mahindra.com", "hyundai": "hyundai.co.in",
        "honda": "honda.com", "toyota": "toyota.com", "tvs": "tvsmotor.com",
        "hero_motocorp": "heromotocorp.com", "bajaj": "bajajauto.com",
        "ICICI_Bank": "icicibank.com", "State_Bank_Of_India": "sbi.co.in",
        "LIC": "licindia.in", "Tanishq_jewellary": "tanishq.co.in",
        "Titan_watches": "titan.co.in", "Allen_Solly": "allensolly.com",
        "Louis_Philippe": "louisphilippe.com", "Peter_England": "peterengland.com",
        "Raymonds": "raymond.in", "Lux_soaps": "lux.com",
        "Pantene_shampoo": "pantene.com", "Air_India": "airindia.com",
        "Make_MyTrip": "makemytrip.com",
    }

    @classmethod
    def fetch(cls, brand_name: str) -> Optional[Image.Image]:
        domain = cls._get_domain(brand_name)
        logo = cls._try_clearbit(domain)
        if logo:
            return cls._ensure_transparency(logo)
        logo = cls._try_google_favicon(domain)
        if logo and min(logo.size) >= 48:
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
                clean = clean[:-len(suffix)]
                break
        return f"{clean}.com"

    @classmethod
    def _try_clearbit(cls, domain: str) -> Optional[Image.Image]:
        try:
            url = f"https://logo.clearbit.com/{domain}"
            resp = requests.get(url, timeout=8, allow_redirects=True)
            if resp.status_code == 200 and "image" in resp.headers.get("content-type", ""):
                return Image.open(io.BytesIO(resp.content)).convert("RGBA")
        except Exception:
            pass
        return None

    @classmethod
    def _try_google_favicon(cls, domain: str) -> Optional[Image.Image]:
        try:
            url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                img = Image.open(io.BytesIO(resp.content))
                if img.size[0] >= 32:
                    return img.convert("RGBA")
        except Exception:
            pass
        return None

    @staticmethod
    def _ensure_transparency(img: Image.Image) -> Optional[Image.Image]:
        img = img.convert("RGBA")
        arr = np.array(img)
        if arr.shape[2] >= 4:
            transparent_ratio = (arr[:, :, 3] < 128).sum() / (arr.shape[0] * arr.shape[1])
            if transparent_ratio > 0.03:
                return img
        h, w = arr.shape[:2]
        if h < 10 or w < 10:
            return img
        border = max(3, min(h, w) // 15)
        corner_pixels: List[tuple] = []
        for cy_s, cx_s in [(0, 0), (0, w - border), (h - border, 0), (h - border, w - border)]:
            for dy in range(border):
                for dx in range(border):
                    cy = min(cy_s + dy, h - 1)
                    cx = min(cx_s + dx, w - 1)
                    corner_pixels.append(tuple(arr[cy, cx, :3].tolist()))
        bg_color = np.array(Counter(corner_pixels).most_common(1)[0][0])
        diff = np.abs(arr[:, :, :3].astype(int) - bg_color.astype(int))
        bg_mask = np.all(diff <= 30, axis=2)
        result = arr.copy()
        result[bg_mask, 3] = 0
        remaining = (result[:, :, 3] > 128).sum() / (h * w)
        if remaining < 0.05:
            return img
        return Image.fromarray(result)


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

    def extract_features(self, image_paths, n=6):
        img_feats = self._encode_images(image_paths)
        return self._rank_pool(img_feats, self.FEATURE_POOL, n)

    def generate_diffusion_prompt(self, image_paths, brand, category, subcategory):
        img_feats = self._encode_images(image_paths)
        top_styles = self._rank_pool(img_feats, self.STYLE_POOL, 2)
        top_moods = self._rank_pool(img_feats, self.MOOD_POOL, 1)
        top_subjects = self._rank_pool(img_feats, self.SUBJECT_POOL, 1)

        brand_clean = brand.replace("_", " ")
        sub_clean = subcategory.replace("_", " ") if subcategory else category.replace("_", " ")

        return (
            f"professional advertisement photography for {brand_clean} {sub_clean}, "
            f"{top_subjects[0]}, {top_styles[0]}, {top_styles[1]}, "
            f"{top_moods[0]} mood, high quality 4k, sharp focus, commercial advertisement"
        )

    def analyze_uploaded_image(self, image: Image.Image):
        """Analyze an uploaded product image using CLIP to extract descriptors."""
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
# Image Generator (Pollinations.ai - free, no API key)
# ---------------------------------------------------------------------------

class PollinationsImageGenerator:
    TIMEOUT = 120
    RETRIES = 2

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Optional[Image.Image]:
        encoded = urllib.parse.quote(prompt, safe="")
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={width}&height={height}&model=flux&nologo=true&enhance=true"
        )
        for attempt in range(self.RETRIES):
            try:
                resp = requests.get(url, timeout=self.TIMEOUT)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    return img
                elif resp.status_code >= 500:
                    time.sleep(5)
                else:
                    return None
            except requests.Timeout:
                continue
            except Exception:
                continue
        return None

    def generate_hd(self, prompt: str) -> Optional[Image.Image]:
        """Generate HD 1080x1080 image."""
        return self.generate(prompt, width=1080, height=1080)


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
        self.text_gen = PollinationsTextGenerator()

    def generate_product_content(
        self, brand: str, category: str, subcategory: str,
        features: List[str], tagline: str, query: str
    ) -> Dict[str, Any]:
        """Generate product title, description, captions using Pollinations text API."""
        brand_clean = brand.replace("_", " ")
        sub_clean = subcategory.replace("_", " ") if subcategory else category.replace("_", " ")
        features_str = ", ".join(features[:4])

        system_prompt = (
            "You are an expert marketing copywriter for product advertisements. "
            "Create compelling, professional marketing content. "
            "Respond ONLY in valid JSON format."
        )

        user_prompt = f"""Create marketing content for: {brand_clean} {sub_clean}
Brand Tagline: {tagline}
Key Features: {features_str}
User Request: {query}

Respond in this exact JSON format:
{{
    "product_title": "Short compelling product title (5-8 words)",
    "product_description": "2-3 sentence product description highlighting key benefits",
    "instagram_caption": "Engaging Instagram caption with call to action (2-3 sentences)",
    "whatsapp_copy": "Short WhatsApp status copy (1-2 sentences, casual tone)",
    "hashtags": ["hashtag1", "hashtag2", "hashtag3", "hashtag4", "hashtag5"]
}}"""

        result = self.text_gen.generate_json(system_prompt, user_prompt)
        if result and isinstance(result, dict):
            return result

        # Template-based fallback
        return {
            "product_title": f"{brand_clean} {sub_clean} - {tagline}" if tagline else f"{brand_clean} {sub_clean} Collection",
            "product_description": (
                f"Discover the {brand_clean} {sub_clean} collection, "
                f"featuring {features[0].lower()} and {features[1].lower()}. "
                f"Experience excellence with {brand_clean}."
            ),
            "instagram_caption": (
                f"Elevate your style with {brand_clean} {sub_clean}! "
                f"{features[0]} | {features[1]} | {features[2]}. "
                f"Shop now and experience the difference!"
            ),
            "whatsapp_copy": (
                f"Check out {brand_clean} {sub_clean}! "
                f"{tagline if tagline else features[0]}. Get yours today!"
            ),
            "hashtags": [
                brand_clean.replace(" ", ""),
                sub_clean.replace(" ", ""),
                "ad", "creative", "shopping",
            ],
        }

    def generate_ad_content_for_language(
        self, brand: str, category: str, subcategory: str,
        features: List[str], tagline: str, query: str, language: str
    ) -> Optional[Dict[str, str]]:
        """Generate ad content directly in a specific language using Pollinations."""
        brand_clean = brand.replace("_", " ")
        sub_clean = subcategory.replace("_", " ") if subcategory else category.replace("_", " ")
        lang_name = SUPPORTED_LANGUAGES.get(language, language)

        system_prompt = (
            f"You are a marketing copywriter. Write ALL content in {lang_name} language. "
            f"Create culturally appropriate, engaging ad copy in {lang_name}. "
            "Respond ONLY in valid JSON."
        )

        user_prompt = f"""Create ad content in {lang_name} for: {brand_clean} {sub_clean}
Tagline: {tagline}
Features: {', '.join(features[:3])}
Request: {query}

Respond in JSON:
{{
    "tagline": "Catchy tagline in {lang_name}",
    "description": "Short product description in {lang_name}",
    "caption": "Social media caption in {lang_name}",
    "cta": "Call to action in {lang_name}"
}}"""

        return self.text_gen.generate_json(system_prompt, user_prompt)


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
# Pamphlet Composer (1080x1080 HD)
# ---------------------------------------------------------------------------

class PamphletComposer:
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

    BANNER_THEMES = [
        ((140, 30, 140), (180, 50, 180)),
        ((30, 100, 180), (50, 130, 210)),
        ((180, 60, 30), (210, 90, 50)),
        ((30, 140, 100), (50, 180, 130)),
        ((160, 120, 30), (200, 160, 50)),
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

    def _text_size(self, draw, text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    @staticmethod
    def _hex_to_rgb(h: str):
        h = h.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _round_corners(self, img, radius):
        mask = Image.new("L", img.size, 0)
        d = ImageDraw.Draw(mask)
        d.rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius, fill=255)
        result = img.convert("RGBA")
        result.putalpha(mask)
        return result

    def _analyze_image(self, img):
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
        text_side = "right" if left_complexity > right_complexity * 1.15 else "left"
        flat_idx = int(np.argmin(complexity))
        free_row, free_col = flat_idx // 3, flat_idx % 3
        return {
            "text_side": text_side,
            "free_row": free_row, "free_col": free_col,
            "avg_brightness": brightness.mean(),
            "is_dark": brightness.mean() < 120,
            "complexity_grid": complexity,
        }

    def _get_layout_config(self, content, analysis):
        brand_hash = sum(ord(c) for c in content.brand_name)
        text_side = analysis["text_side"]
        feat_styles = ["bullets", "pills", "lines"]
        feat_style = feat_styles[brand_hash % len(feat_styles)]
        thumb_styles = ["tilted_cascade", "horizontal_strip", "stacked"]
        thumb_style = thumb_styles[(brand_hash // 3) % len(thumb_styles)]
        banner_theme = self.BANNER_THEMES[brand_hash % len(self.BANNER_THEMES)]
        cta_colors = [
            (240, 190, 50), (220, 60, 60), (50, 180, 120),
            (60, 140, 220), (220, 120, 50),
        ]
        cta_color = cta_colors[(brand_hash // 5) % len(cta_colors)]
        brand_sizes = [52, 58, 48, 54]
        brand_size = brand_sizes[brand_hash % len(brand_sizes)]
        return {
            "text_side": text_side,
            "feat_style": feat_style,
            "thumb_style": thumb_style,
            "thumb_side": "left" if text_side == "right" else "right",
            "banner_theme": banner_theme,
            "cta_color": cta_color,
            "brand_size": brand_size,
            "free_row": analysis["free_row"],
            "free_col": analysis["free_col"],
        }

    def compose(self, content: PamphletContent) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent_rgb = self._hex_to_rgb(content.accent_color)

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

        if content.product_image:
            analysis = self._analyze_image(content.product_image)
        else:
            analysis = {
                "text_side": "left", "free_row": 0, "free_col": 2,
                "avg_brightness": 60, "is_dark": True,
                "complexity_grid": np.zeros((3, 3)),
            }

        cfg = self._get_layout_config(content, analysis)
        text_side = cfg["text_side"]
        is_right = text_side == "right"
        M = self.MARGIN

        # Gradient overlays
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for row in range(int(H * 0.35)):
            t = row / (H * 0.35)
            alpha = int(185 * (1 - t) ** 1.3)
            od.line([(0, row), (W, row)], fill=(0, 0, 0, alpha))

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

        # Logo placement
        if content.logo_image:
            logo = content.logo_image.copy().convert("RGBA")
            lw, lh = logo.size
            logo_h = 90
            logo_scale = logo_h / lh
            logo_w = min(int(lw * logo_scale), 180)
            logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
            grid = analysis.get("complexity_grid", np.zeros((3, 3)))
            best_score = float("inf")
            best_r, best_c = 1, (2 if not is_right else 0)
            for r in range(3):
                for c in range(3):
                    if is_right and c >= 1:
                        continue
                    if not is_right and c <= 1:
                        continue
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
            back_pad = 14
            back_w, back_h = logo_w + 2 * back_pad, logo_h + 2 * back_pad
            backing = Image.new("RGBA", (back_w, back_h), (0, 0, 0, 0))
            backing_d = ImageDraw.Draw(backing)
            backing_d.rounded_rectangle([0, 0, back_w - 1, back_h - 1], 16, fill=(255, 255, 255, 45))
            glow = backing.copy().filter(ImageFilter.GaussianBlur(8))
            bx, by = max(0, logo_x - back_pad), max(0, logo_y - back_pad)
            if bx + glow.width <= W and by + glow.height <= H:
                rg = canvas.crop((bx, by, bx + glow.width, by + glow.height)).convert("RGBA")
                canvas.paste(Image.alpha_composite(rg, glow), (bx, by))
            if bx + backing.width <= W and by + backing.height <= H:
                rg = canvas.crop((bx, by, bx + backing.width, by + backing.height)).convert("RGBA")
                canvas.paste(Image.alpha_composite(rg, backing), (bx, by))
            canvas.paste(logo, (logo_x, logo_y), logo)
            draw = ImageDraw.Draw(canvas)

        # Brand name
        brand_font = self._font("C:/Windows/Fonts/segoeuib.ttf", cfg["brand_size"])
        brand_display = content.brand_name.replace("_", " ").upper()
        bw_text, bh_text = self._text_size(draw, brand_display, brand_font)
        bx = (W - M - bw_text) if is_right else M
        by = 30
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

        # Features
        features = content.features[:6]
        feat_end_y = tag_y + 20
        if features:
            feat_end_y = self._draw_features(
                canvas, draw, features, tag_y + 20, M, W, is_right,
                accent_rgb, cfg["feat_style"]
            )
            draw = ImageDraw.Draw(canvas)

        # CTA Button
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
        txt_lum = cc[0] * 0.299 + cc[1] * 0.587 + cc[2] * 0.114
        txt_col = (20, 20, 20, 255) if txt_lum > 140 else (255, 255, 255, 255)
        btn_d.text((cta_px, cta_py - 2), cta_text, fill=txt_col, font=cta_font)
        cta_x = (W - M - cta_w) if is_right else M
        canvas.paste(btn, (cta_x, cta_y), btn)
        draw = ImageDraw.Draw(canvas)

        # Info banner
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

        # Thumbnails
        if content.thumbnail_images:
            thumbs = content.thumbnail_images[:3]
            self._draw_thumbnails(canvas, thumbs, W, H, M,
                                  cfg["thumb_side"], cfg["thumb_style"])
            draw = ImageDraw.Draw(canvas)

        return canvas.convert("RGB")

    def _draw_features(self, canvas, draw, features, y, margin, W, is_right, accent, style):
        if style == "pills":
            return self._draw_features_pills(canvas, draw, features, y, margin, W, is_right, accent)
        elif style == "lines":
            return self._draw_features_lines(canvas, draw, features, y, margin, W, is_right, accent)
        else:
            return self._draw_features_bullets(canvas, draw, features, y, margin, W, is_right, accent)

    def _draw_features_bullets(self, canvas, draw, features, y, margin, W, is_right, accent):
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
        font = self._font("C:/Windows/Fonts/segoeui.ttf", 16)
        cols = 2
        gap = 10
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
            pd.ellipse([10, pill_h // 2 - 3, 16, pill_h // 2 + 3], fill=(*accent, 220))
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
        font = self._font("C:/Windows/Fonts/segoeui.ttf", 20)
        for feat in features:
            text = f"\u2014  {feat}"
            tw, th = self._text_size(draw, text, font)
            x = (W - margin - tw) if is_right else margin
            draw.text((x + 1, y + 1), text, fill=(0, 0, 0, 120), font=font)
            draw.text((x, y), text, fill=(255, 255, 255, 230), font=font)
            draw.line([(x, y + th + 3), (x + tw, y + th + 3)], fill=(*accent, 60), width=1)
            y += th + 14
        return y + 6

    def _draw_thumbnails(self, canvas, thumbs, W, H, M, side, style):
        if style == "tilted_cascade":
            self._draw_thumbs_tilted(canvas, thumbs, W, H, M, side)
        elif style == "horizontal_strip":
            self._draw_thumbs_strip(canvas, thumbs, W, H, M, side)
        else:
            self._draw_thumbs_stacked(canvas, thumbs, W, H, M, side)

    def _draw_thumbs_tilted(self, canvas, thumbs, W, H, M, side):
        thumb_size = 145
        angles = [-7, 4, -4]
        base_x = (W - 50 - int(thumb_size * 1.8)) if side == "right" else 30
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
            shadow = Image.new("RGBA", (framed.width + 8, framed.height + 8), (0, 0, 0, 0))
            sd = ImageDraw.Draw(shadow)
            sd.rounded_rectangle([4, 4, shadow.width - 1, shadow.height - 1], 12, fill=(0, 0, 0, 50))
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
        thumb_w, thumb_h = 130, 90
        gap = 10
        total = len(thumbs) * thumb_w + (len(thumbs) - 1) * gap
        sx = (W - M - total) if side == "right" else M
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
        thumb_size = 130
        base_x = (W - M - thumb_size - 20) if side == "right" else (M + 10)
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
            ty = int(base_y + i * (tnh * 0.35))
            pw = min(framed.width, W - tx)
            ph = min(framed.height, H - ty)
            if pw > 0 and ph > 0 and tx >= 0 and ty >= 0:
                cr = framed.crop((0, 0, pw, ph))
                rg = canvas.crop((tx, ty, tx + pw, ty + ph)).convert("RGBA")
                canvas.paste(Image.alpha_composite(rg, cr), (tx, ty))


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
        print("  No API keys required!")
        print("=" * 60)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {self.device}")

        self._load_models()

        self.color_extractor = ColorExtractor()
        self.pamphlet_composer = PamphletComposer()
        self.image_gen = PollinationsImageGenerator()
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
                 product_id: str = None) -> GenerationResult:
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
            brand = brand_match.matched_brand or (retrieved_ads[0].brand if retrieved_ads else "Product")
            category = brand_match.category or (retrieved_ads[0].category if retrieved_ads else "product")
            subcategory = brand_match.subcategory or (retrieved_ads[0].subcategory if retrieved_ads else "")

            accent = ColorExtractor.get_accent_color(colors)
            secondary = colors[1] if len(colors) > 1 else accent
            bg_tint = ColorExtractor.lighten_color(accent, factor=0.92)

            # CLIP-based feature extraction
            features = self.clip_content_extractor.extract_features(ad_image_paths, n=6)
            diffusion_prompt = self.clip_content_extractor.generate_diffusion_prompt(
                ad_image_paths, brand, category, subcategory
            )

            # Tagline
            tagline = BRAND_TAGLINES.get(brand, f"Experience {brand.replace('_', ' ')}")

            # If user uploaded an image, analyze it too
            if uploaded_image:
                img_analysis = self.clip_content_extractor.analyze_uploaded_image(uploaded_image)
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

            # Logo
            logo_image = None
            try:
                logo_image = WebLogoFetcher.fetch(brand)
            except Exception:
                pass

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

            content = PamphletContent(
                brand_name=brand, tagline=tagline, features=features[:6],
                accent_color=accent, secondary_color=secondary,
                background_tint=bg_tint, logo_image=logo_image,
                thumbnail_images=thumbnails, thumbnail_paths=thumb_paths,
                diffusion_prompt=diffusion_prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                category=category, subcategory=subcategory,
            )

            result.content = {
                "brand": brand, "tagline": tagline, "features": features,
                "accent_color": accent, "diffusion_prompt": diffusion_prompt,
            }
            result.stage_timings["content_generation"] = time.time() - t0

            # Stage 4: Image generation
            t0 = time.time()
            if uploaded_image:
                product_img = self.enhancer.auto_enhance(uploaded_image)
            else:
                product_img = self.image_gen.generate_hd(diffusion_prompt)

            if product_img:
                content.product_image = product_img
                prod_path = str(OUTPUT_DIR / f"product_{timestamp}.png")
                product_img.save(prod_path, quality=95)
                result.product_image_path = prod_path
                result.image_generator_used = "uploaded" if uploaded_image else "pollinations_flux"
            else:
                result.image_generator_used = "none"
                result.errors.append("Image generation failed")

            result.stage_timings["image_generation"] = time.time() - t0

            # Stage 5: Pamphlet composition
            t0 = time.time()
            pamphlet = self.pamphlet_composer.compose(content)
            pamphlet_path = str(OUTPUT_DIR / f"pamphlet_{timestamp}.png")
            pamphlet.save(pamphlet_path, quality=95)
            result.pamphlet_path = pamphlet_path
            result.stage_timings["composition"] = time.time() - t0

            # Stage 6: Multi-language generation
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
                    # Try generating directly in the target language
                    lang_content = self.content_gen.generate_ad_content_for_language(
                        brand, category, subcategory, features, tagline, query, lang
                    )
                    if lang_content:
                        result.translations[lang] = lang_content
                        result.languages_generated.append(lang)
                        continue
                except Exception:
                    pass

                # Fallback: translate English content
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
