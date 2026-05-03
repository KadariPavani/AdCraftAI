# MAdVerse Pipeline orchestrator — all ad text is AI-generated via Pollinations.

import json
import os
import pickle
import re
import time
import traceback
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from transformers import CLIPModel, CLIPProcessor

from app.models import (
    BASE_DIR, EMBEDDINGS_DIR, FAISS_DIR, OUTPUT_DIR, UPLOAD_DIR,
    SUPPORTED_LANGUAGES, DEFAULT_NEGATIVE_PROMPT,
    RetrievedAd, BrandMatch, AdContent, GenerationResult,
)
from app.brands import BrandMatcher
from app.colors import ColorExtractor
from app.logo import ProLogoFetcher
from app.clip_extract import CLIPContentExtractor
from app.image_gen import ImageGenerator, LocalAdImageGenerator
from app.content_gen import GeminiTextGenerator, GroqTextGenerator, AnthropicTextGenerator, PollinationsTextGenerator, ContentGenerator, Translator, ImageEnhancer
from app.designer import ProAdDesigner
from app.database import Database
from app.dataset_enhancer import DatasetEnhancer

_dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=_dotenv_path, override=True)


class AdCraftPipeline:
    _instance = None
    MIN_BRAND_MATCH_CONFIDENCE = 0.80

    def __init__(self):
        print("\n" + "=" * 70)
        print("  MAdVerse AI - Loading Pipeline")
        print("=" * 70)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [INIT] Device: {self.device}")
        print(f"  [INIT] PyTorch version: {torch.__version__}")
        print(f"  [INIT] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  [INIT] GPU: {torch.cuda.get_device_name(0)}")

        self._load_models()

        # API tokens for image generation fallback chain
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.together_key = os.getenv("TOGETHER_API_KEY", "")
        self.xai_api_key = os.getenv("XAI_API_KEY", "")
        print(f"  [INIT] HF_TOKEN: {'set (' + self.hf_token[:8] + '...)' if self.hf_token else 'NOT SET (HuggingFace models will be skipped)'}")
        print(f"  [INIT] TOGETHER_API_KEY: {'set' if self.together_key else 'NOT SET'}")
        print(f"  [INIT] XAI_API_KEY: {'set' if self.xai_api_key else 'NOT SET'}")

        self.image_gen = ImageGenerator(
            hf_token=self.hf_token or None,
            together_key=self.together_key or None,
            xai_api_key=self.xai_api_key or None,
        )
        print(f"  [INIT] ImageGenerator initialized:")
        print(f"         Primary  : HuggingFace FLUX family {'(token available)' if self.hf_token else '(SKIPPED - no token)'}")
        print(f"                    - text-to-image: FLUX.1-schnell")
        print(f"                    - image-to-image: FLUX.1 Kontext/Redux + FLUX.2 dev/pro/max")
        print(f"         Secondary: xAI Grok image API {'(key available)' if self.xai_api_key else '(SKIPPED - no key)'}")
        print(f"         Tertiary : Pollinations image API (no key)")
        print(f"         Fallback : Gradient (local PIL, always works)")

        self.color_extractor = ColorExtractor()
        print(f"  [INIT] ColorExtractor initialized (KMeans clustering)")
        self.ad_designer = ProAdDesigner()
        print(f"  [INIT] ProAdDesigner initialized (6 theme templates)")
        self.pamphlet_composer = self.ad_designer  # backward compat
        self.local_image_gen = LocalAdImageGenerator()
        print(f"  [INIT] LocalAdImageGenerator initialized (8 category themes)")
        self.content_gen = ContentGenerator(self.clip_content_extractor)
        print(f"  [INIT] ContentGenerator initialized (Gemini -> Groq -> Anthropic -> Pollinations)")
        self.gemini_text_gen = GeminiTextGenerator()
        self.groq_text_gen = GroqTextGenerator()
        self.anthropic_text_gen = AnthropicTextGenerator()
        self.pollinations_text_gen = PollinationsTextGenerator()
        # Use best available text_gen for pipeline (category inference, image prompt)
        for name, gen in [("Gemini", self.gemini_text_gen), ("Groq", self.groq_text_gen),
                          ("Anthropic", self.anthropic_text_gen)]:
            if gen.available:
                self.text_gen = gen
                self._text_gen_name = name
                break
        else:
            self.text_gen = self.pollinations_text_gen
            self._text_gen_name = "Pollinations"
        print(f"  [INIT] Pipeline text_gen: {self._text_gen_name}")
        print(f"  [INIT] PollinationsTextGenerator initialized (URL: {PollinationsTextGenerator.URL})")
        self.translator = Translator()
        print(f"  [INIT] Translator initialized (GoogleTranslator - free, no key)")
        self.enhancer = ImageEnhancer()
        print(f"  [INIT] ImageEnhancer initialized (PIL-based)")
        self.db = Database()
        print(f"  [INIT] Database initialized (SQLite)")
        self.dataset_enhancer = DatasetEnhancer(
            faiss_index=self.index,
            id_to_metadata=self.id_to_metadata,
            brand_matcher=self.brand_matcher,
            clip_model=self.clip_model,
            clip_processor=self.clip_processor,
            device=self.device,
        )
        print(f"  [INIT] DatasetEnhancer initialized (continuous learning)")

        print("\n" + "-" * 70)
        print("  Pipeline ready! All components loaded successfully.")
        print("=" * 70)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_models(self):
        clip_name = "openai/clip-vit-base-patch32"
        print(f"\n  [MODEL] Loading CLIP model: {clip_name}")
        print(f"  [MODEL] Source: HuggingFace Transformers (local cache or download)")
        try:
            self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
            print(f"  [MODEL] CLIP loaded from: remote/cache")
        except OSError:
            self.clip_model = CLIPModel.from_pretrained(clip_name, local_files_only=True).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_name, local_files_only=True)
            print(f"  [MODEL] CLIP loaded from: local files only (offline mode)")
        self.clip_model.eval()
        print(f"  [MODEL] CLIP model loaded on {self.device} | Embedding dim: 512 | Eval mode: ON")

        print(f"\n  [FAISS] Loading FAISS index...")
        index_path = FAISS_DIR / "madverse_index.faiss"
        metadata_path = FAISS_DIR / "id_to_metadata.pkl"
        print(f"  [FAISS] Index path: {index_path}")
        print(f"  [FAISS] Metadata path: {metadata_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run BuildFAISS.py first.")
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self.id_to_metadata = pickle.load(f)
        print(f"  [FAISS] Index loaded: {self.index.ntotal:,} vectors | Dimension: {self.index.d}")
        print(f"  [FAISS] Metadata entries: {len(self.id_to_metadata):,}")

        print(f"\n  [BRAND] Building brand index from metadata...")
        self.brand_matcher = BrandMatcher(self.id_to_metadata)
        print(f"  [BRAND] Brand index built: {len(self.brand_matcher.brand_names)} brands indexed")
        print(f"  [BRAND] Brands: {', '.join(self.brand_matcher.brand_names[:10])}{'...' if len(self.brand_matcher.brand_names) > 10 else ''}")

        self.clip_content_extractor = CLIPContentExtractor(
            self.clip_model, self.clip_processor, self.device
        )
        print(f"  [CLIP] CLIPContentExtractor initialized (feature/style/mood/subject pools)")

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

    @staticmethod
    def _is_valid_brand_candidate(value: Optional[str]) -> bool:
        if not value or not isinstance(value, str):
            return False
        clean = value.strip()
        if not clean:
            return False
        blocked = {
            "unknown", "n/a", "none", "null", "product",
            "it", "this", "that", "these", "those", "my", "your", "our", "their", "his", "her"
        }
        if clean.lower() in blocked:
            return False
        if len(clean) <= 2 and clean.lower() not in {"hp", "lg"}:
            return False
        return True

    def _infer_brand_fallback(self, query: str, product_metadata: Optional[dict] = None) -> str:
        """Infer a stable fallback brand when dataset matching fails.

        Priority:
        1. Explicit brand in product metadata
        2. Product name in metadata (used as brand fallback)
        3. Possessive/capitalized token in query
        4. First meaningful 1-2 words from query
        """
        if isinstance(product_metadata, dict):
            for key in ("brand", "brand_name"):
                candidate = product_metadata.get(key)
                if self._is_valid_brand_candidate(candidate):
                    return str(candidate).strip()

            for key in ("product_name", "name", "title", "product_type"):
                candidate = product_metadata.get(key)
                if self._is_valid_brand_candidate(candidate):
                    return str(candidate).strip()

        text = (query or "").strip()
        if not text:
            return ""

        possessive_match = re.search(r"\b([A-Za-z][A-Za-z0-9&'\-]{1,30})['’]s\b", text)
        if possessive_match:
            candidate = possessive_match.group(1).strip()
            if self._is_valid_brand_candidate(candidate):
                return candidate

        skip_words = {
            "create", "make", "generate", "design", "build", "show", "need", "want",
            "please", "new", "best", "premium", "ad", "advertisement", "pamphlet",
            "poster", "banner", "campaign", "for", "with", "and", "the", "a", "an",
            "it", "this", "that", "these", "those", "style", "engagement", "instagram",
            "target", "platforms", "goal", "real", "time",
        }

        # Prefer explicit branded casing tokens (e.g., SheCodes, iPhone, eBay).
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9&'\-]{2,30}\b", text):
            if re.search(r"[a-z][A-Z]", token) and token.lower() not in skip_words:
                if self._is_valid_brand_candidate(token):
                    return token

        for token in re.findall(r"\b[A-Z][A-Za-z0-9&'\-]{1,30}\b", text):
            if token.lower() not in skip_words and self._is_valid_brand_candidate(token):
                return token

        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9&'\-]*", text)
        filtered = [w for w in words if w.lower() not in skip_words]
        if filtered:
            return " ".join(filtered[:2]).strip()

        return ""

    @staticmethod
    def _normalize_display_text(text: str, max_len: int = 120) -> str:
        value = re.sub(r"\s+", " ", (text or "").strip())
        value = re.sub(r"[^\w\s&'.,:/+\-()]", "", value)
        words = value.split()
        if len(words) >= 8 and len(words) % 2 == 0:
            half = len(words) // 2
            if [w.lower() for w in words[:half]] == [w.lower() for w in words[half:]]:
                words = words[:half]
        value = " ".join(words)
        if len(value) > max_len:
            value = value[:max_len].rsplit(" ", 1)[0].strip()
        return value

    @staticmethod
    def _hex_to_rgb_safe(value: Optional[str], fallback: tuple = (212, 175, 55)) -> tuple:
        if not value or not isinstance(value, str):
            return fallback
        raw = value.strip().lstrip("#")
        if len(raw) != 6:
            return fallback
        try:
            return tuple(int(raw[i:i + 2], 16) for i in (0, 2, 4))
        except Exception:
            return fallback

    @staticmethod
    def _safe_upper(text: Optional[str], fallback: str) -> str:
        value = (text or "").strip()
        return value.upper() if value else fallback

    @staticmethod
    def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int) -> List[str]:
        words = [w for w in (text or "").split() if w]
        if not words:
            return []
        lines: List[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
                if len(lines) >= max_lines - 1:
                    break
        if len(lines) < max_lines:
            lines.append(current)
        return lines[:max_lines]

    def _infer_category_from_query(self, query: str) -> tuple:
        """Infer category/subcategory from free-form query text."""
        try:
            inferred = self.text_gen.generate(
                system_prompt=(
                    "Given a product query, respond with ONLY the product category and subcategory "
                    "separated by a pipe character. Use short lowercase labels. "
                    "Examples: stationery|erasers, footwear|running_shoes, food|snacks, "
                    "electronics|smartphones, clothing|t_shirts, personal_care|shampoo, eyewear|spectacles. "
                    "Output ONLY the category|subcategory, nothing else."
                ),
                user_prompt=query,
            )
            if inferred and "|" in inferred:
                parts = inferred.strip().split("|", 1)
                category = parts[0].strip().replace(" ", "_")
                subcategory = parts[1].strip().replace(" ", "_")
                if category:
                    return category, subcategory
        except Exception as e:
            print(f"  [CONTENT] AI category inference failed ({e})")
        return "product", ""

    @staticmethod
    def _extract_product_alpha_mask(product_img: Image.Image) -> Optional[Image.Image]:
        """Extract alpha for products on light/plain backgrounds."""
        rgb = product_img.convert("RGB")
        w, h = rgb.size
        scale = min(1.0, 420.0 / max(w, h))
        sw, sh = max(96, int(w * scale)), max(96, int(h * scale))
        small = rgb.resize((sw, sh), Image.LANCZOS)
        arr = np.array(small).astype(np.float32)

        maxc = np.max(arr, axis=2)
        minc = np.min(arr, axis=2)
        sat = (maxc - minc) / np.maximum(maxc, 1.0)
        lum = np.mean(arr, axis=2)

        border = np.concatenate([
            arr[0, :, :],
            arr[sh - 1, :, :],
            arr[:, 0, :],
            arr[:, sw - 1, :],
        ], axis=0)
        bg = np.median(border, axis=0)
        bg_lum = float(np.median(np.mean(border, axis=1)))
        bg_dist = np.linalg.norm(arr - bg[None, None, :], axis=2)

        attempts = [
            (44, 0.18, 18),
            (58, 0.22, 22),
            (72, 0.27, 26),
            (88, 0.32, 30),
            (104, 0.36, 34),
        ]

        best_alpha = None
        best_score = None

        for dist_th, sat_th, lum_dev_th in attempts:
            candidate = (
                ((bg_dist < dist_th) & (sat < sat_th)) |
                ((lum > bg_lum + 8) & (sat < sat_th + 0.04)) |
                ((np.abs(lum - bg_lum) < lum_dev_th) & (sat < sat_th))
            ).astype(np.uint8)

            visited = np.zeros((sh, sw), dtype=np.uint8)
            q = deque()
            for x in range(sw):
                if candidate[0, x]:
                    q.append((0, x))
                if candidate[sh - 1, x]:
                    q.append((sh - 1, x))
            for y in range(sh):
                if candidate[y, 0]:
                    q.append((y, 0))
                if candidate[y, sw - 1]:
                    q.append((y, sw - 1))

            while q:
                y, x = q.popleft()
                if y < 0 or y >= sh or x < 0 or x >= sw:
                    continue
                if visited[y, x] or not candidate[y, x]:
                    continue
                visited[y, x] = 1
                q.append((y - 1, x))
                q.append((y + 1, x))
                q.append((y, x - 1))
                q.append((y, x + 1))

            fg = (1 - visited).astype(np.uint8)

            # Keep largest foreground component.
            labels_seen = np.zeros_like(fg, dtype=np.uint8)
            largest = None
            largest_count = 0
            for sy in range(sh):
                for sx in range(sw):
                    if fg[sy, sx] == 0 or labels_seen[sy, sx]:
                        continue
                    comp = []
                    qq = deque([(sy, sx)])
                    labels_seen[sy, sx] = 1
                    while qq:
                        cy, cx = qq.popleft()
                        comp.append((cy, cx))
                        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                            if 0 <= ny < sh and 0 <= nx < sw and fg[ny, nx] and not labels_seen[ny, nx]:
                                labels_seen[ny, nx] = 1
                                qq.append((ny, nx))
                    if len(comp) > largest_count:
                        largest_count = len(comp)
                        largest = comp

            if not largest:
                continue

            fg[:, :] = 0
            for y, x in largest:
                fg[y, x] = 1

            fg_ratio = float(np.count_nonzero(fg)) / float(fg.size)
            if fg_ratio < 0.03 or fg_ratio > 0.88:
                continue

            alpha = Image.fromarray((fg * 255).astype(np.uint8), mode="L").resize((w, h), Image.BILINEAR)
            alpha = alpha.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
            alpha = alpha.filter(ImageFilter.GaussianBlur(1.2))
            alpha_arr = np.array(alpha)
            alpha_arr = np.where(alpha_arr > 22, alpha_arr, 0).astype(np.uint8)
            alpha_img = Image.fromarray(alpha_arr, mode="L")

            bbox = alpha_img.getbbox()
            if not bbox:
                continue
            bx0, by0, bx1, by1 = bbox
            bbox_ratio = ((bx1 - bx0) * (by1 - by0)) / float(w * h)
            edge = np.concatenate([
                alpha_arr[:2, :].ravel(),
                alpha_arr[-2:, :].ravel(),
                alpha_arr[:, :2].ravel(),
                alpha_arr[:, -2:].ravel(),
            ])
            edge_leak = float(np.mean(edge))
            score = abs(fg_ratio - 0.22) + bbox_ratio * 0.18 + edge_leak / 255.0
            if best_score is None or score < best_score:
                best_score = score
                best_alpha = alpha_img

        return best_alpha

    @staticmethod
    def _suppress_white_halo(product_rgba: Image.Image) -> Image.Image:
        """Reduce white fringe around transparent product edges."""
        arr = np.array(product_rgba.convert("RGBA")).astype(np.uint8)
        rgb = arr[..., :3].astype(np.float32)
        alpha = arr[..., 3].astype(np.float32)
        if np.max(alpha) <= 0:
            return product_rgba

        maxc = np.max(rgb, axis=2)
        minc = np.min(rgb, axis=2)
        sat = (maxc - minc) / np.maximum(maxc, 1.0)

        edge_zone = (alpha > 0) & (alpha < 210)
        near_white = (maxc > 200) & (sat < 0.18)
        fringe = edge_zone & near_white

        alpha[fringe] = alpha[fringe] * 0.45
        arr[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGBA")

    @staticmethod
    def _suppress_white_patches(product_rgba: Image.Image) -> Image.Image:
        """Reduce large white matte/paper artifacts that look like pasted boxes."""
        arr = np.array(product_rgba.convert("RGBA")).astype(np.uint8)
        rgb = arr[..., :3].astype(np.float32)
        alpha = arr[..., 3].astype(np.float32)
        if np.max(alpha) <= 0:
            return product_rgba

        maxc = np.max(rgb, axis=2)
        minc = np.min(rgb, axis=2)
        sat = (maxc - minc) / np.maximum(maxc, 1.0)
        lum = np.mean(rgb, axis=2)

        white_matte = (alpha > 150) & (lum > 214) & (sat < 0.10)
        alpha[white_matte] = alpha[white_matte] * 0.28
        arr[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGBA")

    def _prepare_uploaded_product_hero(self, product_img: Image.Image, mask_source: Optional[Image.Image] = None) -> Image.Image:
        """Prepare a transparent hero cutout for the pamphlet renderer."""
        product_rgb = product_img.convert("RGB")
        mask_img = mask_source.convert("RGB") if mask_source is not None else product_rgb
        alpha_mask = self._extract_product_alpha_mask(mask_img)
        product_rgba = product_rgb.convert("RGBA")
        if alpha_mask is not None:
            product_rgba.putalpha(alpha_mask)
            product_rgba = self._suppress_white_halo(product_rgba)
            product_rgba = self._suppress_white_patches(product_rgba)
        else:
            h0, w0 = np.array(mask_img).shape[:2]
            yy, xx = np.mgrid[0:h0, 0:w0]
            border = np.concatenate([
                np.array(mask_img)[0, :, :],
                np.array(mask_img)[h0 - 1, :, :],
                np.array(mask_img)[:, 0, :],
                np.array(mask_img)[:, w0 - 1, :],
            ], axis=0).astype(np.float32)
            bg = np.median(border, axis=0)
            bg_dist = np.linalg.norm(np.array(mask_img).astype(np.float32) - bg[None, None, :], axis=2)
            lum = np.mean(np.array(mask_img).astype(np.float32), axis=2)
            sat = (np.max(np.array(mask_img), axis=2) - np.min(np.array(mask_img), axis=2)) / np.maximum(np.max(np.array(mask_img), axis=2), 1.0)
            candidate = ((bg_dist > 38) & (lum < 245)) | (sat > 0.15)
            alpha = (candidate.astype(np.uint8) * 255)
            alpha = Image.fromarray(alpha, mode="L")
            alpha = alpha.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
            alpha = alpha.filter(ImageFilter.GaussianBlur(1.2))
            product_rgba.putalpha(alpha)
            product_rgba = self._suppress_white_patches(product_rgba)
        bbox = product_rgba.getbbox()
        if bbox:
            product_rgba = product_rgba.crop(bbox)
        return product_rgba

    def _background_has_extra_object_noise(self, bg_img: Image.Image) -> bool:
        """Detect object-like clutter in central/lower area where uploaded flow expects clean background."""
        arr = np.array(bg_img.convert("RGB").resize((540, 540), Image.LANCZOS)).astype(np.float32)
        gray = arr.mean(axis=2)

        # Gradient backgrounds are smooth; object intrusions increase local edges/texture.
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        edge = dx[:, :-1] + dy[:-1, :]

        # Region where AI often hallucinates extra props (basket/table) in uploaded flow.
        roi = edge[230:520, 130:410]
        color_roi = arr[230:520, 130:410, :]

        edge_density = float((roi > 26.0).mean())
        color_std = float(color_roi.std())

        return edge_density > 0.16 and color_std > 34.0

    @staticmethod
    def _derive_backdrop_palette(source_img: Optional[Image.Image], fallback_a: tuple, fallback_b: tuple) -> Tuple[tuple, tuple]:
        """Derive smooth backdrop tones from the uploaded/enhanced image border colors."""
        if source_img is None:
            return fallback_a, fallback_b
        try:
            arr = np.array(source_img.convert("RGB").resize((540, 540), Image.LANCZOS)).astype(np.float32)
            border = np.concatenate([
                arr[:24, :, :].reshape(-1, 3),
                arr[-24:, :, :].reshape(-1, 3),
                arr[:, :24, :].reshape(-1, 3),
                arr[:, -24:, :].reshape(-1, 3),
            ], axis=0)
            mid = np.percentile(border, 50, axis=0)
            light = np.clip(mid * 0.92 + 22.0, 0, 255)
            dark = np.clip(mid * 0.62 + 8.0, 0, 255)
            return tuple(light.astype(np.uint8)), tuple(dark.astype(np.uint8))
        except Exception:
            return fallback_a, fallback_b

    def _compose_styled_real_product_ad(
        self,
        content: AdContent,
        product_img: Image.Image,
        palette_source: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, str]:
        """Complete full-frame cinematic pamphlet using the uploaded product image as-is."""
        w, h = 1080, 1080
        accent = self._hex_to_rgb_safe(content.accent_color, (183, 123, 45))
        bg_method = "uploaded_full_frame_cinematic"
        print(f"  [DESIGN] Uploaded-image background source: {bg_method}")

        # Full-frame cover crop from uploaded image (no object cutout/removal look).
        source = product_img.convert("RGB")
        sw, sh = source.size
        scale = max(w / max(1, sw), h / max(1, sh))
        rw, rh = max(1, int(sw * scale)), max(1, int(sh * scale))
        source = source.resize((rw, rh), Image.LANCZOS)
        left = max(0, (rw - w) // 2)
        top = max(0, (rh - h) // 2)
        source = source.crop((left, top, left + w, top + h))
        canvas = source.convert("RGBA")

        # Cinematic readability overlays.
        light = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ldraw = ImageDraw.Draw(light)
        ldraw.ellipse([int(w * 0.18), int(h * 0.16), int(w * 0.82), int(h * 0.70)], fill=(255, 245, 220, 72))
        light = light.filter(ImageFilter.GaussianBlur(40))
        canvas = Image.alpha_composite(canvas, light)

        shade = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shade)
        for y in range(int(h * 0.34)):
            a = int(124 * (1 - (y / max(1, int(h * 0.34)))))
            sdraw.line([(0, y), (w, y)], fill=(12, 10, 8, a))
        for y in range(int(h * 0.68), h):
            t = (y - int(h * 0.68)) / max(1, h - int(h * 0.68))
            a = int(138 * t)
            sdraw.line([(0, y), (w, y)], fill=(14, 10, 8, a))
        canvas = Image.alpha_composite(canvas, shade)

        # Typography (pamphlet hierarchy)
        sample_text = " ".join([content.headline or "", content.tagline or "", content.cta_text or ""])
        self.ad_designer._script = self.ad_designer._detect_script(sample_text)
        brand_font = self.ad_designer._font("elegant", 66)
        headline_font = self.ad_designer._font("accent", 80)
        sub_font = self.ad_designer._font("modern", 42)
        detail_font = self.ad_designer._font("body", 34)
        cta_font = self.ad_designer._font("headline", 34)

        draw = ImageDraw.Draw(canvas)
        brand_text = self._safe_upper(self._normalize_display_text(content.brand_name, 28), "EXQUISITE")
        b_w = draw.textbbox((0, 0), brand_text, font=brand_font)[2]
        bx = (w - b_w) // 2
        by = 78
        draw.text((bx, by + 2), brand_text, font=brand_font, fill=(18, 10, 6, 170))
        draw.text((bx, by), brand_text, font=brand_font, fill=(252, 238, 196, 246))

        title = self._normalize_display_text((content.headline or "").strip(), 64) or "Premium Artisan Craft"
        title_lines = self._wrap_lines(draw, title, headline_font, int(w * 0.88), max_lines=2)
        ty = 162
        for line in title_lines:
            tw = draw.textbbox((0, 0), line, font=headline_font)[2]
            tx = (w - tw) // 2
            draw.text((tx, ty + 2), line, font=headline_font, fill=(16, 10, 6, 170))
            draw.text((tx, ty), line, font=headline_font, fill=(255, 244, 212, 248))
            ty += 84

        subtitle = self._normalize_display_text((content.tagline or "").strip(), 54) or "Handcrafted Mastery"
        subtitle_lines = self._wrap_lines(draw, subtitle, sub_font, int(w * 0.84), max_lines=1)
        if subtitle_lines:
            sw = draw.textbbox((0, 0), subtitle_lines[0], font=sub_font)[2]
            sx = (w - sw) // 2
            sy = ty + 6
            draw.text((sx, sy + 2), subtitle_lines[0], font=sub_font, fill=(18, 10, 6, 160))
            draw.text((sx, sy), subtitle_lines[0], font=sub_font, fill=(248, 230, 186, 238))

        cleaned_features = [
            self._normalize_display_text(f, 36) for f in (content.features or []) if isinstance(f, str) and f.strip()
        ]
        footer = " • ".join([f for f in cleaned_features[:2] if f]).strip()
        if not footer:
            footer = "Premium Build • Artisan Quality"
        footer_lines = self._wrap_lines(draw, footer, detail_font, int(w * 0.84), max_lines=2)
        fy = 866
        for line in footer_lines:
            fw = draw.textbbox((0, 0), line, font=detail_font)[2]
            draw.text(((w - fw) // 2, fy), line, font=detail_font, fill=(250, 236, 192, 244))
            fy += 40

        cta_text = self._normalize_display_text((content.cta_text or "SHOP NOW"), 18).upper()
        cta_w = max(250, min(420, draw.textbbox((0, 0), cta_text, font=cta_font)[2] + 80))
        cta_h = 68
        cta_x = (w - cta_w) // 2
        cta_y = h - 112
        draw.rounded_rectangle([cta_x, cta_y, cta_x + cta_w, cta_y + cta_h], radius=32,
                               fill=(62, 30, 8, 236), outline=(240, 209, 130, 205), width=2)
        tw = draw.textbbox((0, 0), cta_text, font=cta_font)[2]
        th = draw.textbbox((0, 0), cta_text, font=cta_font)[3]
        draw.text((cta_x + (cta_w - tw) // 2, cta_y + (cta_h - th) // 2 - 4), cta_text,
                  font=cta_font, fill=(255, 239, 194, 252))

        # Subtle rectangular edge falloff (no circular frame artifact).
        edge = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        edraw = ImageDraw.Draw(edge)
        for i in range(0, 46):
            a = int(1.8 * (46 - i))
            edraw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=(22, 16, 12, a), width=1)
        edge = edge.filter(ImageFilter.GaussianBlur(2))
        canvas = Image.alpha_composite(canvas, edge)

        result = canvas.convert("RGB")
        result = ImageEnhance.Contrast(result).enhance(1.08)
        result = ImageEnhance.Color(result).enhance(1.07)
        result = ImageEnhance.Sharpness(result).enhance(1.12)
        return result, bg_method

    # ------------------------------------------------------------------
    # Full Generation Pipeline
    # ------------------------------------------------------------------
    def generate(self, query: str, languages: List[str] = None,
                 uploaded_image: Image.Image = None,
                 product_id: str = None,
                 brand_override: str = None,
                 product_metadata: dict = None,
                 image_model: str = "",
                 style_instruction: str = "") -> GenerationResult:
        if languages is None:
            languages = ["en"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = GenerationResult(
            query=query,
            timestamp=datetime.now().isoformat(),
            product_id=product_id,
        )
        total_start = time.time()

        print("\n" + "=" * 70)
        print("  GENERATION PIPELINE STARTED")
        print("=" * 70)
        print(f"  [INPUT] Query: \"{query}\"")
        print(f"  [INPUT] Languages: {languages}")
        print(f"  [INPUT] Brand override: {brand_override or 'None (auto-detect)'}")
        print(f"  [INPUT] Uploaded image: {'YES' if uploaded_image else 'NO'}")
        print(f"  [INPUT] Requested image model: {image_model or 'default'}")
        print(f"  [INPUT] Style instruction: {'YES' if style_instruction else 'NO'}")
        print(f"  [INPUT] Product ID: {product_id or 'None'}")
        print(f"  [INPUT] Timestamp: {timestamp}")

        try:
            # ── Stage 0: Brand matching ──────────────────────────────
            print("\n" + "-" * 70)
            print("  STAGE 0: Brand Matching")
            print("-" * 70)
            t0 = time.time()
            brand_match = self.match_brand(query)
            effective_brand_match = brand_match
            if brand_match.matched_brand and brand_match.confidence < self.MIN_BRAND_MATCH_CONFIDENCE:
                print(
                    f"  [BRAND] Low-confidence match ignored "
                    f"({brand_match.matched_brand}, {brand_match.confidence:.2%} < {self.MIN_BRAND_MATCH_CONFIDENCE:.0%})"
                )
                effective_brand_match = BrandMatch()
            result.stage_timings["brand_matching"] = time.time() - t0
            if effective_brand_match.matched_brand:
                result.brand_match = {
                    "matched_brand": effective_brand_match.matched_brand,
                    "confidence": effective_brand_match.confidence,
                    "category": effective_brand_match.category,
                    "subcategory": effective_brand_match.subcategory,
                    "image_count": effective_brand_match.image_count,
                }
                print(f"  [BRAND] Matched: {effective_brand_match.matched_brand}")
                print(f"  [BRAND] Confidence: {effective_brand_match.confidence:.2%}")
                print(f"  [BRAND] Query token: \"{effective_brand_match.query_token}\"")
                print(f"  [BRAND] Category: {effective_brand_match.category} / {effective_brand_match.subcategory}")
                print(f"  [BRAND] Dataset images: {effective_brand_match.image_count}")
            else:
                print(f"  [BRAND] No brand match found in query")
            print(f"  [BRAND] Time: {result.stage_timings['brand_matching']:.3f}s")

            # ── Stage 1: FAISS Retrieval ─────────────────────────────
            print("\n" + "-" * 70)
            print("  STAGE 1: FAISS Retrieval")
            print("-" * 70)
            t0 = time.time()
            print(f"  [FAISS] Model: openai/clip-vit-base-patch32 (text encoder)")
            print(f"  [FAISS] Encoding query text to 512-dim vector...")
            retrieved_ads = self.retrieve(query, effective_brand_match, k=5)
            result.stage_timings["retrieval"] = time.time() - t0
            result.retrieved_ads = [
                {"rank": ad.rank, "similarity": ad.similarity,
                 "brand": ad.brand, "category": ad.category,
                 "subcategory": ad.subcategory, "image_path": ad.image_path}
                for ad in retrieved_ads
            ]
            print(f"  [FAISS] Retrieved {len(retrieved_ads)} ads (k=5)")
            if effective_brand_match.matched_brand:
                print(f"  [FAISS] Search mode: Brand-filtered (only {effective_brand_match.matched_brand} vectors)")
            else:
                print(f"  [FAISS] Search mode: Global (all {self.index.ntotal:,} vectors)")
            for ad in retrieved_ads:
                print(f"    #{ad.rank} | Sim: {ad.similarity:.4f} | Brand: {ad.brand} | {ad.category}/{ad.subcategory}")
            print(f"  [FAISS] Time: {result.stage_timings['retrieval']:.3f}s")

            if not retrieved_ads:
                print(f"  [FAISS] ERROR: No ads retrieved from dataset!")
                result.errors.append("No ads retrieved from dataset")
                return result

            # ── Stage 2: Color extraction ────────────────────────────
            print("\n" + "-" * 70)
            print("  STAGE 2: Color Extraction")
            print("-" * 70)
            t0 = time.time()
            ad_image_paths = [ad.image_path for ad in retrieved_ads if Path(ad.image_path).exists()]
            print(f"  [COLOR] Analyzing {len(ad_image_paths)} retrieved ad images")
            print(f"  [COLOR] Algorithm: KMeans clustering (k=5, n_init=10)")
            colors = self.color_extractor.extract_from_images(ad_image_paths)
            result.stage_timings["color_extraction"] = time.time() - t0
            result.extracted_colors = colors
            print(f"  [COLOR] Extracted {len(colors)} dominant colors: {colors}")
            accent = ColorExtractor.get_accent_color(colors)
            secondary = colors[1] if len(colors) > 1 else accent
            bg_tint = ColorExtractor.lighten_color(accent, factor=0.92)
            print(f"  [COLOR] Accent: {accent} | Secondary: {secondary} | BG tint: {bg_tint}")
            print(f"  [COLOR] Time: {result.stage_timings['color_extraction']:.3f}s")

            # ── Stage 3: Content generation ──────────────────────────
            print("\n" + "-" * 70)
            print("  STAGE 3: Content Generation (AI-powered)")
            print("-" * 70)
            t0 = time.time()
            # Use user-specified brand if provided, otherwise auto-detect
            brand_source = "dataset_match"
            if brand_override:
                brand = BrandMatcher.normalize_brand(brand_override)
                brand_source = "user_override"
                print(f"  [CONTENT] Using brand override (normalized): {brand}")
            elif effective_brand_match.matched_brand:
                brand = BrandMatcher.normalize_brand(effective_brand_match.matched_brand)
                print(f"  [CONTENT] Auto-detected brand (normalized): {brand}")
            else:
                inferred = self._infer_brand_fallback(query, product_metadata)
                if self._is_valid_brand_candidate(inferred):
                    brand = BrandMatcher.normalize_brand(inferred)
                    brand_source = "query_or_product_fallback"
                    print(f"  [CONTENT] No dataset match; fallback brand inferred: {brand}")
                elif retrieved_ads:
                    brand = BrandMatcher.normalize_brand(retrieved_ads[0].brand)
                    brand_source = "retrieval_fallback"
                    print(f"  [CONTENT] No inferred brand; fallback to top retrieved brand: {brand}")
                else:
                    brand = "Product"
                    brand_source = "generic_fallback"
                    print(f"  [CONTENT] No inferred brand; using generic fallback: {brand}")

            if effective_brand_match.category:
                category = effective_brand_match.category
                subcategory = effective_brand_match.subcategory or ""
                print(f"  [CONTENT] Category from brand match: {category} / {subcategory}")
            else:
                category = "product"
                subcategory = ""

            if uploaded_image:
                # For user-uploaded products, trust query/product context over dataset brand category.
                ai_category, ai_subcategory = self._infer_category_from_query(query)
                print(f"  [CONTENT] Uploaded image flow - AI category: {ai_category} / {ai_subcategory}")
                if ai_category and ai_category != "product":
                    category, subcategory = ai_category, ai_subcategory
            elif not effective_brand_match.category:
                print(f"  [CONTENT] Brand not in dataset - inferring category from query via AI...")
                category, subcategory = self._infer_category_from_query(query)
                print(f"  [CONTENT] AI-inferred category: {category} / {subcategory}")
            print(f"  [CONTENT] Final category: {category} / Subcategory: {subcategory}")

            # Always expose the final brand used by generation so clients can display
            # consistent brand info even when no dataset match exists.
            if effective_brand_match.matched_brand and not brand_override:
                confidence = effective_brand_match.confidence
                image_count = effective_brand_match.image_count
            elif brand_override:
                confidence = 1.0
                image_count = len(self.brand_matcher.get_brand_indices(brand))
            else:
                confidence = 0.0
                image_count = 0

            result.brand_match = {
                "matched_brand": brand,
                "confidence": confidence,
                "category": category,
                "subcategory": subcategory,
                "image_count": image_count,
                "source": brand_source,
            }

            # Build a comprehensive ad image prompt using LLM
            print(f"\n  [AI-PROMPT] Building ad image prompt via LLM...")
            print(f"  [AI-PROMPT] Goal: Generate accurate product image with ad layout")
            brand_clean = brand.replace("_", " ")

            # Extract specific product name from metadata or query
            import re as _re
            product_name = ""
            if product_metadata:
                product_name = product_metadata.get("product_name", "") or product_metadata.get("product_type", "")
            if not product_name:
                # Fall back to extracting from query
                product_name = self.content_gen._extract_product_from_query(query, brand_clean)
            if not product_name:
                product_name = subcategory.replace("_", " ") if subcategory else category.replace("_", " ")
            print(f"  [AI-PROMPT] Product name for image: \"{product_name}\"")

            # Build product detail string for image prompt
            product_detail_parts = [product_name]
            if product_metadata:
                for field in ["material", "color", "key_features", "occasion"]:
                    val = product_metadata.get(field, "")
                    if val:
                        product_detail_parts.append(f"{field}: {val}")
            product_detail_str = ", ".join(product_detail_parts)

            # Get scene description from metadata if available
            scene_desc = ""
            if product_metadata:
                scene_desc = product_metadata.get("scene_description", "")
            # Also check if rich_prompt has scene info
            rich_prompt = ""
            if product_metadata:
                rich_prompt = product_metadata.get("rich_prompt", "")

            uploaded_flow = uploaded_image is not None
            # First generate the ad copy so we can include it in the image prompt
            print(f"\n  [TEXT-GEN-PRE] Pre-generating ad copy for image prompt inclusion...")
            pre_text_content = self.content_gen.generate_product_content(
                brand, category, subcategory, [], "", query,
                product_metadata=product_metadata,
            )
            pre_headline = pre_text_content.get("headline", "") or brand_clean
            pre_tagline = pre_text_content.get("tagline", "")
            pre_cta = pre_text_content.get("cta_text", "") or "SHOP NOW"
            pre_features = pre_text_content.get("features", [])[:3]
            print(f"  [TEXT-GEN-PRE] Pre-headline: \"{pre_headline}\"")
            print(f"  [TEXT-GEN-PRE] Pre-CTA: \"{pre_cta}\"")

            # Extract price from query or metadata
            price_text = ""
            if product_metadata and product_metadata.get("price"):
                price_text = str(product_metadata["price"])
            if not price_text:
                price_match = _re.search(r'(?:rs\.?|inr|usd|\$|₹|price[:\s]+)\s*[\d,]+(?:\.\d{2})?', query, _re.I)
                price_text = price_match.group().strip() if price_match else ""

            model_guidance = ""
            normalized_model = (image_model or "").strip().lower().replace("_", "-")
            if normalized_model in {"flux1-kontext-dev", "flux.1-kontext-dev"}:
                model_guidance = (
                    "Use FLUX.1 Kontext style: preserve exact product identity and structure while "
                    "performing high-quality image-to-image enhancement with realistic lighting."
                )
            elif normalized_model in {"flux1-redux-dev", "flux.1-redux-dev"}:
                model_guidance = (
                    "Use FLUX.1 Redux style: conservative refinement, cleaner details, reduced artifacts, "
                    "minimal composition drift."
                )
            elif normalized_model in {"flux2-dev", "flux.2-dev", "flux2-pro", "flux.2-pro", "flux2-max", "flux.2-max"}:
                model_guidance = (
                    "Use FLUX.2 quality style: premium realism, strong detail fidelity, "
                    "high commercial ad polish while preserving product authenticity."
                )
            if style_instruction:
                model_guidance = f"{model_guidance} {style_instruction}".strip()

            if uploaded_flow:
                prompt_goal = (
                    "Generate an image-editing prompt for uploaded product enhancement. "
                    "The output image must contain only the real product with realistic scene lighting. "
                    "Keep the entire product visible with original aspect ratio, no stretching, no warping, no cropping. "
                    "Product should be slightly smaller in frame (about 55-65% of frame height) with clean negative space for content layout. "
                    "Use cinematic composition flow with layered depth and premium ad mood. "
                    "No text, no letters, no logos, no watermarks, no slogans, no CTA buttons, no duplicate products."
                )
            else:
                prompt_goal = (
                    "Generate an image prompt that shows the EXACT product accurately with premium ad layout, "
                    "natural backgrounds (no solid poster colors), and text spelled EXACTLY as provided above."
                )

            try:
                ai_prompt = self.text_gen.generate(
                    system_prompt=(
                        "You are an expert at writing image generation prompts for product advertisement images.\n\n"
                        "CRITICAL RULES:\n"
                        "1. The PRODUCT must be the EXACT product described — be very specific about what the "
                        "product looks like. For example, 'gold bangles' should show BANGLES (circular wrist "
                        "jewelry), NOT earrings or necklaces. Describe the exact product shape, style, and appearance.\n"
                        "2. Show the product ACCURATELY with balanced composition and no distortion.\n"
                        + (
                            "3. Do NOT generate text overlays. NO letters, NO words, NO logos, NO watermarks, NO CTA buttons.\n"
                            if uploaded_flow else
                            "3. Include text overlay elements: headline text (with CORRECT spelling), brand name (accurate spelling), "
                            "price if provided. NO CTA buttons, NO 'Shop Now' buttons, NO 'Buy Now' buttons, NO action buttons.\n"
                        )
                        +
                        "4. Use a cinematic premium magazine-style ad layout with sophisticated typography and elegant composition.\n"
                        "5. Describe warm studio lighting (key light + soft rim light), subtle depth, and premium cinematic mood appropriate for the product category.\n"
                        "6. CRITICAL: Ensure all text in the image has ACCURATE SPELLING with NO TYPOS. "
                        "7. Use natural photographic backgrounds or elegant gradients — NO solid poster-style backgrounds.\n"
                        "8. Keep the visual style realistic and premium. Avoid cartoonish, fantasy, odd, neon, gimmicky, or cluttered themes.\n"
                        "9. Keep clean negative space and high readability. Do not overcrowd the layout.\n\n"
                        "Output ONLY the image generation prompt (4-6 sentences), nothing else."
                    ),
                    user_prompt=(
                        f"Brand: {brand_clean}\n"
                        f"Product: {product_detail_str}\n"
                        f"User request: {query}\n"
                        + (f"Scene: {scene_desc}\n" if scene_desc else "")
                        + (
                            f"\nIMPORTANT: This is uploaded-image enhancement mode. Keep ONE product only. "
                            f"Keep full product visible with original aspect ratio; no stretch, no squeeze, no crop. "
                            f"Keep product slightly smaller in frame (~55-65% frame height), centered with breathing room around it. "
                            f"Leave clean negative space so ad content can be placed cleanly. "
                            f"Use cinematic flow: warm key light, soft rim light, subtle depth, elegant premium mood. "
                            f"No text overlays or graphic typography in the generated image.\n"
                            if uploaded_flow else
                            (
                                f"\nText overlay elements:\n"
                                f"- Headline: \"{pre_headline}\" (spell EXACTLY as written)\n"
                                f"- Brand: \"{brand_clean}\" (spell EXACTLY as written)\n"
                                + (f"- Tagline: \"{pre_tagline}\" (spell EXACTLY as written)\n" if pre_tagline else "")
                                + (f"- Price: \"{price_text}\"\n" if price_text else "")
                                + f"\nIMPORTANT: Do NOT include any CTA buttons, 'Shop Now', 'Buy Now', or action buttons in the image.\n"
                            )
                        )
                        + f"IMPORTANT: Keep style cinematic, realistic, premium, and minimal. Avoid odd/cartoon/fantasy/neon themes.\n"
                        + (f"IMPORTANT: {model_guidance}\n" if model_guidance else "")
                        + prompt_goal
                    ),
                )
                if ai_prompt and len(ai_prompt) > 50:
                    if uploaded_flow:
                        diffusion_prompt = (
                            f"{ai_prompt.strip()}, "
                            f"ultra-realistic single-product photography, studio-grade detail, "
                            f"cinematic clean composition, natural lighting, premium color grading, "
                            f"full product visible, preserve original product aspect ratio, no stretching or geometric distortion, "
                            f"product scaled slightly smaller in frame (around 55-65% frame height), leave balanced negative space for ad content, "
                            f"no text, no letters, no logos, no watermark, "
                            f"single product only, no duplicate product instances, "
                            f"8k commercial product photography"
                        )
                    else:
                        diffusion_prompt = (
                            f"{ai_prompt.strip()}, "
                            f"ultra-professional cinematic magazine advertisement, premium product photography, "
                            f"accurate product depiction, sophisticated typography with correct spelling, "
                            f"warm studio key light and soft rim light, elegant minimal composition, "
                            f"natural backgrounds, no odd themes, no cartoon look, no neon gimmicks, "
                            f"no CTA buttons, no solid poster backgrounds, "
                            f"8k, highest quality, luxury commercial photography"
                        )
                    if model_guidance:
                        diffusion_prompt = f"{diffusion_prompt}, {model_guidance}"
                    print(f"  [AI-PROMPT] SUCCESS - Ad prompt generated ({len(diffusion_prompt)} chars)")
                    print(f"  [AI-PROMPT] Prompt: {diffusion_prompt[:200]}...")
                else:
                    raise ValueError(f"AI response too short ({len(ai_prompt) if ai_prompt else 0} chars)")
            except Exception as e:
                print(f"  [AI-PROMPT] LLM prompt failed: {e}, using structured fallback")
                # Structured fallback with specific product description
                features_text = ", ".join(pre_features) if pre_features else ""
                if uploaded_flow:
                    diffusion_prompt = (
                        f"Ultra-realistic commercial product photography of {product_detail_str}. "
                        + (f"Scene: {scene_desc}. " if scene_desc else
                           f"Elegant warm studio key light with soft rim light and clean depth. ")
                        + (f"Key features visible: {features_text}. " if features_text else "")
                        + f"Single product only, no duplicate objects. "
                        f"Full product visible with original aspect ratio, no stretching, no warping, no crop. "
                        f"Product slightly smaller in frame (about 55-65% frame height) with clean negative space for content placement. "
                        f"Cinematic composition flow with layered depth, subtle vignette, premium commercial mood. "
                        f"No text, no letters, no logos, no watermark, no typography overlays. "
                        f"Natural premium background, sharp details, accurate product geometry, 8k quality."
                    )
                else:
                    diffusion_prompt = (
                        f"Ultra-professional cinematic magazine advertisement for {brand_clean}. "
                        f"A stunning photograph of {product_detail_str} shown prominently in the center. "
                        + (f"Scene: {scene_desc}. " if scene_desc else
                           f"Elegant warm studio key light with soft rim light, sophisticated realistic backdrop. ")
                        + f"Elegant headline text \"{pre_headline}\" with accurate spelling at the top. "
                        f"Brand name \"{brand_clean}\" displayed clearly with correct spelling. "
                        + (f"Tagline \"{pre_tagline}\" spelled correctly. " if pre_tagline else "")
                        + (f"Price \"{price_text}\" shown elegantly. " if price_text else "")
                        + (f"Key features: {features_text}. " if features_text else "")
                        + f"NO CTA buttons, NO 'Shop Now' or 'Buy Now' buttons, NO action buttons. "
                        f"NO odd, cartoon, fantasy, neon, or gimmicky visual styles. "
                        f"Premium magazine-style design, sophisticated modern typography with accurate spelling, "
                        f"cinematic realistic mood, natural backgrounds without solid poster colors, precise product depiction, "
                        f"8k quality, luxury commercial photography layout."
                    )
                if model_guidance:
                    diffusion_prompt = f"{diffusion_prompt} {model_guidance}"

            # Reuse the pre-generated text content (already generated above for the image prompt)
            print(f"\n  [TEXT-GEN] Using pre-generated ad copy (already created for image prompt)...")
            text_content = pre_text_content

            # Extract AI-generated fields
            headline = text_content.get("headline", "") or brand.replace("_", " ")
            tagline = text_content.get("tagline", "")
            features = text_content.get("features", [])
            cta_text = text_content.get("cta_text", "")

            result.product_title = text_content.get("product_title", "")
            result.product_description = text_content.get("product_description", "")
            result.instagram_caption = text_content.get("instagram_caption", "")
            result.whatsapp_copy = text_content.get("whatsapp_copy", "")
            result.hashtags = text_content.get("hashtags", [])

            print(f"  [TEXT-GEN] Generated content:")
            print(f"    Headline: \"{headline}\"")
            print(f"    Tagline: \"{tagline}\"")
            print(f"    Features: {features}")
            print(f"    CTA: \"{cta_text}\"")
            print(f"    Product title: \"{result.product_title}\"")
            print(f"    Hashtags: {result.hashtags}")

            logo_image = None
            can_use_logo = (
                len((brand or "").strip()) >= 3 and
                brand_source in {"dataset_match", "user_override"}
            )
            if can_use_logo:
                print(f"\n  [LOGO] Fetching logo for brand: {brand}")
                try:
                    logo_image = ProLogoFetcher.fetch(brand)
                    if logo_image:
                        print(f"  [LOGO] Logo fetched: {logo_image.size} | Mode: {logo_image.mode}")
                    else:
                        print(f"  [LOGO] No logo found")
                except Exception as e:
                    print(f"  [LOGO] Logo fetch failed: {e}")
            else:
                print(f"\n  [LOGO] Skipping logo fetch for inferred/short brand: {brand} (source={brand_source})")

            # Get brand domain for URL display
            brand_domain = ""
            try:
                domain = ProLogoFetcher._get_domain(brand)
                brand_domain = f"www.{domain}"
                print(f"  [LOGO] Brand domain: {brand_domain}")
            except Exception:
                print(f"  [LOGO] Could not determine brand domain")

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
            print(f"  [CONTENT] Loaded {len(thumbnails)} thumbnail images")

            content = AdContent(
                brand_name=brand, product_name=product_name, headline=headline, tagline=tagline,
                features=features[:6],
                accent_color=accent, secondary_color=secondary,
                background_tint=bg_tint, logo_image=logo_image,
                thumbnail_images=thumbnails, thumbnail_paths=thumb_paths,
                diffusion_prompt=diffusion_prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                category=category, subcategory=subcategory,
                brand_domain=brand_domain,
                cta_text=cta_text,
            )

            result.content = {
                "brand": brand, "tagline": tagline, "features": features,
                "accent_color": accent, "diffusion_prompt": diffusion_prompt,
            }
            result.stage_timings["content_generation"] = time.time() - t0
            print(f"  [CONTENT] Time: {result.stage_timings['content_generation']:.3f}s")

            # ── Stage 4: Image Generation ────────────────────────────
            print("\n" + "-" * 70)
            print("  STAGE 4: Image Generation")
            print("-" * 70)
            t0 = time.time()
            uploaded_palette_source = None
            if uploaded_image:
                print(f"  [IMAGE] Using uploaded image (model-based img2img enhancement)")
                print(f"  [IMAGE] Original size: {uploaded_image.size}")
                print(f"  [IMAGE] Note: Stage 6 builds cinematic hero-product pamphlet from this image.")
                edit_prompt = (
                    f"{diffusion_prompt}. Preserve the exact product shape, logo, materials, and proportions. "
                    f"Keep the complete product visible with original aspect ratio; no stretch, squeeze, or crop. "
                    f"Scale product slightly smaller in frame (about 55-65% frame height) and maintain clean negative space for content. "
                    f"Follow cinematic composition flow with realistic layered depth and premium lighting. "
                    f"Output must be a real advertisement-ready product visual with photorealistic quality."
                )
                print(f"  [IMAGE] Running image-to-image enhancement for uploaded image...")
                edited_img, method = self.image_gen.edit(
                    image=uploaded_image,
                    prompt=edit_prompt,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    width=1080,
                    height=1080,
                    model_preference=image_model or None,
                )
                if edited_img is not None:
                    uploaded_palette_source = edited_img
                    product_img = edited_img
                    result.image_generator_used = method
                    print(f"  [IMAGE] Uploaded image enhanced via model pipeline: {method}")
                else:
                    print(f"  [IMAGE] Model-based img2img unavailable; trying alternate model styling pass")
                    alt_prompt = (
                        f"{edit_prompt} Create cinematic product-ad background only. "
                        f"Do not add extra products, logos, labels, or text."
                    )
                    alt_img, alt_method = self.image_gen.generate(
                        prompt=alt_prompt,
                        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                        width=1080,
                        height=1080,
                        model_preference="flux1-schnell",
                    )

                    enhanced_img = self.enhancer.auto_enhance(uploaded_image)
                    if alt_img is not None and alt_method != "gradient_fallback":
                        uploaded_palette_source = alt_img
                        product_img = enhanced_img
                        result.image_generator_used = f"uploaded_local_enhance+{alt_method}_style_fallback"
                        print(f"  [IMAGE] Alternate model styling succeeded: {alt_method}")
                    else:
                        print(f"  [IMAGE] Alternate model styling unavailable; falling back to local enhancement")
                        uploaded_palette_source = enhanced_img
                        product_img = enhanced_img
                        result.image_generator_used = "uploaded_local_enhance"
                print(f"  [IMAGE] Enhanced size: {product_img.size}")
            else:
                print(f"  [IMAGE] Starting COMPLETE AD image generation (all text/buttons/CTAs in image):")
                print(f"  [IMAGE] Target size: 1080x1080 (final ad size)")
                print(f"  [IMAGE] Prompt length: {len(diffusion_prompt)} chars")
                product_img, method = self.image_gen.generate(
                    prompt=diffusion_prompt,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    width=1080,
                    height=1080,
                    model_preference=image_model or None,
                )

                # Use brand colors for gradient fallback
                if method == "gradient_fallback" and colors:
                    print(f"  [IMAGE] Applying brand colors to gradient: {colors[:2]}")
                    product_img = self.image_gen._make_gradient(1080, 1080, colors)

                result.image_generator_used = method
                print(f"  [IMAGE] RESULT: Generated via -> {method}")
                print(f"  [IMAGE] Output size: {product_img.size}")

            if product_img:
                content.product_image = product_img
                prod_path = str(OUTPUT_DIR / f"product_{timestamp}.png")
                product_img.save(prod_path, quality=95)
                result.product_image_path = prod_path
                print(f"  [IMAGE] Saved to: {prod_path}")

            result.stage_timings["image_generation"] = time.time() - t0
            print(f"  [IMAGE] Time: {result.stage_timings['image_generation']:.3f}s")

            # ── Stage 5: Translations ────────────────────────────────
            print("\n" + "-" * 70)
            print("  STAGE 5: Multi-Language Translations")
            print("-" * 70)
            t0 = time.time()
            print(f"  [TRANS] Requested languages: {languages}")
            print(f"  [TRANS] Translation engine: GoogleTranslator (deep_translator, free)")
            result.languages_generated = ["en"]
            result.translations["en"] = {
                "product_title": result.product_title,
                "product_description": result.product_description,
                "instagram_caption": result.instagram_caption,
                "whatsapp_copy": result.whatsapp_copy,
                "tagline": tagline,
            }
            print(f"  [TRANS] English (en): base content set")

            for lang in languages:
                if lang == "en":
                    continue
                lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
                print(f"\n  [TRANS] Translating to {lang_name} ({lang})...")
                try:
                    lang_content = self.content_gen.generate_ad_content_for_language(
                        brand, category, subcategory, features, tagline, query, lang
                    )
                    if lang_content:
                        result.translations[lang] = lang_content
                        result.languages_generated.append(lang)
                        print(f"  [TRANS] {lang}: AI-generated content (direct)")
                        continue
                except Exception as e:
                    print(f"  [TRANS] {lang}: AI generation failed ({e}), falling back to translation")

                try:
                    print(f"  [TRANS] {lang}: Using GoogleTranslator fallback...")
                    translated = {
                        "product_title": self.translator.translate(result.product_title, lang),
                        "product_description": self.translator.translate(result.product_description, lang),
                        "instagram_caption": self.translator.translate(result.instagram_caption, lang),
                        "whatsapp_copy": self.translator.translate(result.whatsapp_copy, lang),
                        "tagline": self.translator.translate(tagline, lang),
                    }
                    result.translations[lang] = translated
                    result.languages_generated.append(lang)
                    print(f"  [TRANS] {lang}: SUCCESS (5 fields translated)")
                    print(f"    Title: \"{translated['product_title'][:60]}...\"")
                except Exception as e:
                    print(f"  [TRANS] {lang}: FAILED - {str(e)}")
                    result.errors.append(f"Translation to {lang} failed: {str(e)}")

            result.stage_timings["translations"] = time.time() - t0
            print(f"\n  [TRANS] Languages generated: {result.languages_generated}")
            print(f"  [TRANS] Time: {result.stage_timings['translations']:.3f}s")

            # ── Stage 6: Ad Composition ──────────────────────────────
            print("\n" + "-" * 70)
            print("  STAGE 6: Ad Composition")
            print("-" * 70)
            t0 = time.time()
            if product_img:
                if uploaded_image:
                    print("  [DESIGN] Uploaded product image detected - using color-matched cinematic pamphlet")
                    try:
                        content.theme_name = "realistic_pamphlet"
                        pamphlet, bg_method = self._compose_styled_real_product_ad(
                            content,
                            product_img,
                            palette_source=uploaded_palette_source
                        )
                        current_method = result.image_generator_used or "uploaded"
                        result.image_generator_used = f"{current_method}+{bg_method}+designer"
                        print(f"  [DESIGN] Color-matched uploaded pamphlet composition complete ({bg_method})")
                    except Exception as e:
                        print(f"  [DESIGN] Color-matched composition failed ({e}) - using designer fallback")
                        content.product_image = product_img
                        pamphlet = self.ad_designer.compose(content).convert("RGB")
                        result.image_generator_used = f"{result.image_generator_used}+designer"
                else:
                    if result.image_generator_used == "gradient_fallback":
                        # HF image generation was unavailable; build a proper prompt-based pamphlet
                        # instead of returning a plain gradient.
                        print("  [DESIGN] Gradient fallback detected - composing rich typography pamphlet from prompt")
                        content.product_image = None
                        content.theme_name = "premium_dark"
                        pamphlet = self.ad_designer.compose(content).convert("RGB")
                        result.image_generator_used = "gradient_fallback+designer"
                    else:
                        # For text-only generation, the AI model already returns a complete ad.
                        print("  [DESIGN] Using AI-generated image directly as final ad")
                        pamphlet = product_img.convert("RGB")

                if pamphlet.size != (1080, 1080):
                    print(f"  [DESIGN] Resizing from {pamphlet.size} to 1080x1080")
                    pamphlet = pamphlet.resize((1080, 1080), Image.LANCZOS)

                pamphlet_path = str(OUTPUT_DIR / f"pamphlet_{timestamp}.png")
                pamphlet.save(pamphlet_path, quality=95)
                result.pamphlet_path = pamphlet_path
                print(f"  [DESIGN] Pamphlet saved: {pamphlet_path}")
                print(f"  [DESIGN] Pamphlet size: {pamphlet.size}")
            else:
                print("  [DESIGN] WARNING: No product image available!")
                result.errors.append("No image generated for ad")

            result.stage_timings["composition"] = time.time() - t0
            print(f"  [DESIGN] Time: {result.stage_timings['composition']:.3f}s")

            # Save to database
            if product_id:
                print(f"\n  [DB] Saving generated content to database for product: {product_id}")
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
                print(f"  [DB] Content saved successfully")

            result.stage_timings["total"] = time.time() - total_start

            # Save result JSON
            json_path = OUTPUT_DIR / f"result_{timestamp}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False, default=str)
            print(f"\n  [OUTPUT] Result JSON saved: {json_path}")

            # ── Stage 7: Dataset Enhancement ─────────────────────────
            print("\n" + "-" * 70)
            print("  STAGE 7: Dataset Enhancement (Continuous Learning)")
            print("-" * 70)
            try:
                if result.pamphlet_path:
                    t0 = time.time()
                    print(f"  [DATASET] Enhancing dataset with generated pamphlet...")
                    print(f"  [DATASET] Source: {result.pamphlet_path}")
                    print(f"  [DATASET] Brand: {brand} | Category: {category}/{subcategory}")
                    ds_paths = self.dataset_enhancer.enhance(
                        pamphlet_path=result.pamphlet_path,
                        brand=brand,
                        category=category,
                        subcategory=subcategory,
                        query=query,
                        timestamp=timestamp,
                    )
                    if ds_paths:
                        result.dataset_paths = ds_paths
                        print(f"  [DATASET] Enhancement successful!")
                        print(f"  [DATASET] New index size: {self.index.ntotal:,} vectors")
                    else:
                        print(f"  [DATASET] Enhancement returned None (disabled or failed)")
                    result.stage_timings["dataset_enhancement"] = time.time() - t0
                    print(f"  [DATASET] Time: {result.stage_timings['dataset_enhancement']:.3f}s")
                else:
                    print(f"  [DATASET] Skipped - no pamphlet path")
            except Exception as de_err:
                print(f"  [DATASET] Skipped (non-blocking error): {de_err}")

        except Exception as e:
            result.errors.append(f"Pipeline error: {str(e)}")
            print(f"\n  [ERROR] Pipeline failed: {str(e)}")
            traceback.print_exc()

        # ── Final Summary ────────────────────────────────────────
        total_time = time.time() - total_start
        print("\n" + "=" * 70)
        print("  GENERATION PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  [SUMMARY] Query: \"{query}\"")
        print(f"  [SUMMARY] Brand: {result.brand_match.get('matched_brand', 'Unknown') if result.brand_match else 'None'}")
        print(f"  [SUMMARY] Image model used: {result.image_generator_used or 'None'}")
        print(f"  [SUMMARY] Text model: openai (via Pollinations)")
        print(f"  [SUMMARY] Translation engine: GoogleTranslator")
        print(f"  [SUMMARY] Languages: {result.languages_generated}")
        print(f"  [SUMMARY] Errors: {len(result.errors)}")
        if result.errors:
            for err in result.errors:
                print(f"    - {err}")
        print(f"\n  [TIMINGS]")
        for stage, t in result.stage_timings.items():
            bar = "#" * int(min(t / max(total_time, 0.001) * 30, 30))
            print(f"    {stage:<22} {t:>7.3f}s  {bar}")
        print(f"    {'TOTAL':<22} {total_time:>7.3f}s")
        print("=" * 70 + "\n")

        return result

    # ------------------------------------------------------------------
    # Individual feature methods
    # ------------------------------------------------------------------
    def enhance_image(self, image: Image.Image) -> Image.Image:
        print(f"\n  [ENHANCE] Auto-enhancing image: {image.size} | Mode: {image.mode}")
        print(f"  [ENHANCE] Engine: PIL ImageEnhance (brightness, contrast, sharpness, color)")
        enhanced = self.enhancer.auto_enhance(image)
        print(f"  [ENHANCE] Enhancement complete")
        return enhanced

    def generate_description(self, query: str, image: Image.Image = None) -> dict:
        print(f"\n  [DESCRIBE] Generating description for: \"{query}\"")
        brand_match = self.match_brand(query)
        brand = brand_match.matched_brand or "Product"
        category = brand_match.category or "general"
        subcategory = brand_match.subcategory or ""
        print(f"  [DESCRIBE] Brand: {brand} | Category: {category}/{subcategory}")
        print(f"  [DESCRIBE] Model: openai (via Pollinations AI)")

        result = self.content_gen.generate_product_content(
            brand, category, subcategory, [], "", query
        )
        print(f"  [DESCRIBE] Generated {len(result)} fields")
        return result

    def generate_captions(self, query: str, languages: List[str] = None) -> dict:
        if languages is None:
            languages = ["en"]

        print(f"\n  [CAPTIONS] Generating captions for: \"{query}\"")
        print(f"  [CAPTIONS] Languages: {languages}")

        desc = self.generate_description(query)
        result = {"en": desc}

        for lang in languages:
            if lang == "en":
                continue
            lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
            print(f"  [CAPTIONS] Translating to {lang_name} ({lang})...")
            try:
                translated = self.translator.translate_content(desc, lang)
                result[lang] = translated
                print(f"  [CAPTIONS] {lang}: SUCCESS")
            except Exception as e:
                print(f"  [CAPTIONS] {lang}: FAILED ({e}), using English fallback")
                result[lang] = desc

        print(f"  [CAPTIONS] Generated for {len(result)} languages: {list(result.keys())}")
        return result
