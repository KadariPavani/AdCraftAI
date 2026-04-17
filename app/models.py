# Shared data classes, constants, and path configuration for AdCraft AI.

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
FAISS_DIR = EMBEDDINGS_DIR / "faiss_indexes"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_DIR = Path(os.getenv("MADVERSE_DB_DIR", str(BASE_DIR / "products_db"))).resolve()

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, pixelated, distorted, deformed, ugly, oversaturated, "
    "extra fingers, extra limbs, disfigured face, bad anatomy, bad proportions, "
    "poorly drawn hands, poorly drawn face, mutation, amateur, clip art, cartoon, "
    "frame, border, collage, split image, multiple views, "
    "misspelled text, garbled letters, illegible text, wrong spelling, typos, spelling errors, "
    "CTA buttons, 'buy now' button, 'shop now' button, action buttons, call-to-action buttons, "
    "solid poster background, flat background color, plain colored background, poster-style layout"
)

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
    product_name: str = ""
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
    dataset_paths: Optional[Dict[str, str]] = None
