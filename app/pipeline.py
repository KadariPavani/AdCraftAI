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

load_dotenv()


class AdCraftPipeline:
    _instance = None

    def __init__(self):
        print("\n" + "=" * 60)
        print("  AdCraft AI - Loading Pipeline")
        print("=" * 60)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {self.device}")

        self._load_models()

        # API tokens for image generation fallback chain
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.together_key = os.getenv("TOGETHER_API_KEY", "")
        self.image_gen = ImageGenerator(
            hf_token=self.hf_token or None,
            together_key=self.together_key or None,
        )

        self.color_extractor = ColorExtractor()
        self.ad_designer = ProAdDesigner()
        self.pamphlet_composer = self.ad_designer  # backward compat
        self.local_image_gen = LocalAdImageGenerator()
        self.content_gen = ContentGenerator(self.clip_content_extractor)
        self.text_gen = PollinationsTextGenerator()
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

            # Stage 3: Content generation — ALL text via AI (Pollinations)
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

            # CLIP-based diffusion prompt (for image generation only)
            diffusion_prompt = self.clip_content_extractor.generate_diffusion_prompt(
                ad_image_paths, brand, category, subcategory, user_query=query
            )

            # Try AI-enhanced diffusion prompt
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
                pass

            # Generate ALL ad text via AI (headline, tagline, features, CTA, etc.)
            print("  Generating ad copy via AI...")
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

            # headline, tagline, features, cta_text are already set from AI generation above

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
                cta_text=cta_text,
            )

            result.content = {
                "brand": brand, "tagline": tagline, "features": features,
                "accent_color": accent, "diffusion_prompt": diffusion_prompt,
            }
            result.stage_timings["content_generation"] = time.time() - t0

            # Stage 4: Image Generation — exact same as RAGPipeline.py
            # Pollinations FLUX -> HuggingFace Inference FLUX.1-schnell -> gradient fallback
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
                translated_headline = (
                    t.get("product_title", "")
                    or t.get("tagline", "")
                    or t.get("description", "")
                )
                if translated_headline:
                    content.headline = translated_headline
                content.tagline = t.get("tagline", "") or content.tagline

                translated_cta = t.get("cta", "")
                if translated_cta:
                    content.cta_text = translated_cta
                else:
                    try:
                        content.cta_text = self.translator.translate(english_cta, primary_lang) or english_cta
                    except Exception:
                        pass

                translated_features = []
                for feat in content.features[:6]:
                    try:
                        tf = self.translator.translate(feat, primary_lang)
                        translated_features.append(tf if tf else feat)
                    except Exception:
                        translated_features.append(feat)
                if translated_features:
                    content.features = translated_features

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

        # All content is AI-generated — no hardcoded taglines or feature pools
        return self.content_gen.generate_product_content(
            brand, category, subcategory, [], "", query
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
