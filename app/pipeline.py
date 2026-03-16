# MAdVerse Pipeline orchestrator — all ad text is AI-generated via Pollinations.

import json
import os
import pickle
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
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
from app.content_gen import PollinationsTextGenerator, ContentGenerator, Translator, ImageEnhancer
from app.designer import ProAdDesigner
from app.database import Database
from app.dataset_enhancer import DatasetEnhancer

load_dotenv()


class AdCraftPipeline:
    _instance = None

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
        print(f"  [INIT] HF_TOKEN: {'set (' + self.hf_token[:8] + '...)' if self.hf_token else 'NOT SET (HuggingFace models will be skipped)'}")
        print(f"  [INIT] TOGETHER_API_KEY: {'set' if self.together_key else 'NOT SET'}")

        self.image_gen = ImageGenerator(
            hf_token=self.hf_token or None,
            together_key=self.together_key or None,
        )
        print(f"  [INIT] ImageGenerator initialized:")
        print(f"         Primary : HuggingFace FLUX.1-schnell {'(token available)' if self.hf_token else '(SKIPPED - no token)'}")
        print(f"         Fallback: Gradient (local PIL, always works)")

        self.color_extractor = ColorExtractor()
        print(f"  [INIT] ColorExtractor initialized (KMeans clustering)")
        self.ad_designer = ProAdDesigner()
        print(f"  [INIT] ProAdDesigner initialized (6 theme templates)")
        self.pamphlet_composer = self.ad_designer  # backward compat
        self.local_image_gen = LocalAdImageGenerator()
        print(f"  [INIT] LocalAdImageGenerator initialized (8 category themes)")
        self.content_gen = ContentGenerator(self.clip_content_extractor)
        print(f"  [INIT] ContentGenerator initialized (Pollinations AI text)")
        self.text_gen = PollinationsTextGenerator()
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

        print("\n" + "=" * 70)
        print("  GENERATION PIPELINE STARTED")
        print("=" * 70)
        print(f"  [INPUT] Query: \"{query}\"")
        print(f"  [INPUT] Languages: {languages}")
        print(f"  [INPUT] Brand override: {brand_override or 'None (auto-detect)'}")
        print(f"  [INPUT] Uploaded image: {'YES' if uploaded_image else 'NO'}")
        print(f"  [INPUT] Product ID: {product_id or 'None'}")
        print(f"  [INPUT] Timestamp: {timestamp}")

        try:
            # ── Stage 0: Brand matching ──────────────────────────────
            print("\n" + "-" * 70)
            print("  STAGE 0: Brand Matching")
            print("-" * 70)
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
                print(f"  [BRAND] Matched: {brand_match.matched_brand}")
                print(f"  [BRAND] Confidence: {brand_match.confidence:.2%}")
                print(f"  [BRAND] Query token: \"{brand_match.query_token}\"")
                print(f"  [BRAND] Category: {brand_match.category} / {brand_match.subcategory}")
                print(f"  [BRAND] Dataset images: {brand_match.image_count}")
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
            retrieved_ads = self.retrieve(query, brand_match, k=5)
            result.stage_timings["retrieval"] = time.time() - t0
            result.retrieved_ads = [
                {"rank": ad.rank, "similarity": ad.similarity,
                 "brand": ad.brand, "category": ad.category,
                 "subcategory": ad.subcategory, "image_path": ad.image_path}
                for ad in retrieved_ads
            ]
            print(f"  [FAISS] Retrieved {len(retrieved_ads)} ads (k=5)")
            if brand_match.matched_brand:
                print(f"  [FAISS] Search mode: Brand-filtered (only {brand_match.matched_brand} vectors)")
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
            if brand_override:
                brand = BrandMatcher.normalize_brand(brand_override)
                print(f"  [CONTENT] Using brand override (normalized): {brand}")
            else:
                raw_brand = brand_match.matched_brand or (retrieved_ads[0].brand if retrieved_ads else "Product")
                brand = BrandMatcher.normalize_brand(raw_brand)
                print(f"  [CONTENT] Auto-detected brand (normalized): {brand}")

            if brand_match.category:
                # Brand was found in the dataset — use its known category
                category = brand_match.category
                subcategory = brand_match.subcategory or ""
                print(f"  [CONTENT] Category from brand match: {category} / {subcategory}")
            else:
                # Brand NOT in dataset — ask AI to infer category from the query
                # Do NOT use retrieved ad's category (random FAISS neighbor is unreliable)
                print(f"  [CONTENT] Brand not in dataset — inferring category from query via AI...")
                try:
                    inferred = self.text_gen.generate(
                        system_prompt=(
                            "Given a product query, respond with ONLY the product category and subcategory "
                            "separated by a pipe character. Use short lowercase labels. "
                            "Examples: stationery|erasers, footwear|running_shoes, food|snacks, "
                            "electronics|smartphones, clothing|t_shirts, personal_care|shampoo. "
                            "Output ONLY the category|subcategory, nothing else."
                        ),
                        user_prompt=query,
                    )
                    if inferred and "|" in inferred:
                        parts = inferred.strip().split("|", 1)
                        category = parts[0].strip().replace(" ", "_")
                        subcategory = parts[1].strip().replace(" ", "_")
                        print(f"  [CONTENT] AI-inferred category: {category} / {subcategory}")
                    else:
                        category = "product"
                        subcategory = ""
                        print(f"  [CONTENT] AI inference unclear, using generic: {category}")
                except Exception as e:
                    category = "product"
                    subcategory = ""
                    print(f"  [CONTENT] AI category inference failed ({e}), using generic")
            print(f"  [CONTENT] Final category: {category} / Subcategory: {subcategory}")

            # CLIP-based diffusion prompt (for image generation only)
            print(f"\n  [CLIP] Generating diffusion prompt via CLIP content extraction...")
            print(f"  [CLIP] Model: openai/clip-vit-base-patch32 (vision+text)")
            print(f"  [CLIP] Ranking styles/moods/subjects against {len(ad_image_paths)} reference images")
            diffusion_prompt = self.clip_content_extractor.generate_diffusion_prompt(
                ad_image_paths, brand, category, subcategory, user_query=query
            )
            print(f"  [CLIP] Base prompt: {diffusion_prompt[:100]}...")

            # Try AI-enhanced diffusion prompt
            print(f"\n  [AI-PROMPT] Enhancing image prompt via Pollinations AI...")
            print(f"  [AI-PROMPT] Model: openai (via Pollinations text API)")
            print(f"  [AI-PROMPT] URL: {self.text_gen.URL}")
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
                    print(f"  [AI-PROMPT] SUCCESS - AI-enhanced prompt generated ({len(diffusion_prompt)} chars)")
                    print(f"  [AI-PROMPT] Prompt: {diffusion_prompt[:120]}...")
                else:
                    print(f"  [AI-PROMPT] FALLBACK - AI response too short ({len(ai_prompt) if ai_prompt else 0} chars), using CLIP prompt")
            except Exception as e:
                print(f"  [AI-PROMPT] FALLBACK - AI prompt enhancement failed: {e}")
                print(f"  [AI-PROMPT] Using CLIP-generated prompt instead")

            # Generate ALL ad text via AI (headline, tagline, features, CTA, etc.)
            print(f"\n  [TEXT-GEN] Generating ALL ad copy via Pollinations AI...")
            print(f"  [TEXT-GEN] Model: openai (via Pollinations)")
            print(f"  [TEXT-GEN] Mode: JSON (structured output)")
            print(f"  [TEXT-GEN] Input: brand={brand}, category={category}, query=\"{query}\"")
            text_content = self.content_gen.generate_product_content(
                brand, category, subcategory, [], "", query
            )

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

            # Logo (with transparent background)
            print(f"\n  [LOGO] Fetching logo for brand: {brand}")
            logo_image = None
            try:
                logo_image = ProLogoFetcher.fetch(brand)
                if logo_image:
                    print(f"  [LOGO] Logo fetched: {logo_image.size} | Mode: {logo_image.mode}")
                else:
                    print(f"  [LOGO] No logo found")
            except Exception as e:
                print(f"  [LOGO] Logo fetch failed: {e}")

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
                brand_name=brand, headline=headline, tagline=tagline,
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
            if uploaded_image:
                print(f"  [IMAGE] Using uploaded image (skipping generation)")
                print(f"  [IMAGE] Original size: {uploaded_image.size}")
                print(f"  [IMAGE] Auto-enhancing uploaded image...")
                product_img = self.enhancer.auto_enhance(uploaded_image)
                result.image_generator_used = "uploaded"
                print(f"  [IMAGE] Enhanced size: {product_img.size}")
            else:
                print(f"  [IMAGE] Starting image generation (Primary: HF FLUX.1-schnell | Fallback: gradient):")
                print(f"  [IMAGE] Target size: 1024x768")
                print(f"  [IMAGE] Prompt length: {len(diffusion_prompt)} chars")
                product_img, method = self.image_gen.generate(
                    prompt=diffusion_prompt,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    width=1024,
                    height=768,
                )

                # Use brand colors for gradient fallback
                if method == "gradient_fallback" and colors:
                    print(f"  [IMAGE] Applying brand colors to gradient: {colors[:2]}")
                    product_img = self.image_gen._make_gradient(1024, 768, colors)

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
            print("  STAGE 6: Ad Composition (Pamphlet Design)")
            print("-" * 70)
            t0 = time.time()
            # Determine the primary display language (first non-en if available, else en)
            primary_lang = "en"
            for lang in languages:
                if lang != "en" and lang in result.translations:
                    primary_lang = lang
                    break
            print(f"  [DESIGN] Primary display language: {primary_lang} ({SUPPORTED_LANGUAGES.get(primary_lang, primary_lang)})")
            print(f"  [DESIGN] Bilingual mode: {'YES (English secondary)' if primary_lang != 'en' and 'en' in languages else 'NO (single language)'}")
            print(f"  [DESIGN] Designer: ProAdDesigner (6 theme templates)")
            print(f"  [DESIGN] Canvas size: 1080x1080")

            # Save English text as secondary BEFORE translating
            english_headline = content.headline
            english_features = list(content.features)
            english_cta = self.ad_designer._get_cta_text(content)

            # Update ad content text to use the primary language
            if primary_lang != "en" and primary_lang in result.translations:
                print(f"  [DESIGN] Applying {primary_lang} translations to ad text...")
                t = result.translations[primary_lang]
                translated_headline = (
                    t.get("product_title", "")
                    or t.get("tagline", "")
                    or t.get("description", "")
                )
                if translated_headline:
                    content.headline = translated_headline
                    print(f"  [DESIGN] Translated headline: \"{translated_headline[:60]}\"")
                content.tagline = t.get("tagline", "") or content.tagline

                translated_cta = t.get("cta", "")
                if translated_cta:
                    content.cta_text = translated_cta
                    print(f"  [DESIGN] Translated CTA: \"{translated_cta}\"")
                else:
                    try:
                        content.cta_text = self.translator.translate(english_cta, primary_lang) or english_cta
                        print(f"  [DESIGN] CTA translated via GoogleTranslator: \"{content.cta_text}\"")
                    except Exception:
                        print(f"  [DESIGN] CTA translation failed, using English: \"{english_cta}\"")

                translated_features = []
                for feat in content.features[:6]:
                    try:
                        tf = self.translator.translate(feat, primary_lang)
                        translated_features.append(tf if tf else feat)
                    except Exception:
                        translated_features.append(feat)
                if translated_features:
                    content.features = translated_features
                    print(f"  [DESIGN] Translated {len(translated_features)} features")

                if "en" in languages:
                    content.headline_secondary = english_headline
                    content.features_secondary = english_features[:4]
                    content.cta_secondary = english_cta
                    print(f"  [DESIGN] English secondary text set for bilingual ad")

            print(f"  [DESIGN] Composing pamphlet...")
            pamphlet = self.ad_designer.compose(content)
            pamphlet_path = str(OUTPUT_DIR / f"pamphlet_{timestamp}.png")
            pamphlet.save(pamphlet_path, quality=95)
            result.pamphlet_path = pamphlet_path
            result.stage_timings["composition"] = time.time() - t0
            print(f"  [DESIGN] Pamphlet saved: {pamphlet_path}")
            print(f"  [DESIGN] Pamphlet size: {pamphlet.size}")
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
