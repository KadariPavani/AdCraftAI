# Text generation, translation, and image enhancement.
# Clean chain: Groq text generation + Translator + PIL enhancement.

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from PIL import Image, ImageEnhance


class GroqTextGenerator:
    """Groq API text generator."""

    URL = "https://api.groq.com/openai/v1/chat/completions"
    MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    TIMEOUT = 30

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.available = bool(self.api_key)
        if self.available:
            print(f"      [GROQ-TEXT] Initialized with API key ({self.api_key[:8]}...)")
        else:
            print("      [GROQ-TEXT] No GROQ_API_KEY - Groq text gen disabled")

    def generate(self, system_prompt: str, user_prompt: str, model: str = None) -> Optional[str]:
        if not self.available:
            return None
        models_to_try = list(self.MODELS)
        if model and model in models_to_try:
            models_to_try.remove(model)
            models_to_try.insert(0, model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for m in models_to_try:
            payload = {
                "model": m,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.9,
                "max_tokens": 1024,
            }
            try:
                print(f"      [GROQ-TEXT] Trying {m}...")
                t0 = time.time()
                resp = requests.post(self.URL, json=payload, headers=headers, timeout=self.TIMEOUT)
                elapsed = time.time() - t0
                print(f"      [GROQ-TEXT] Response: {m} status={resp.status_code} | time={elapsed:.2f}s")
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if text and len(text) > 5:
                        print(f"      [GROQ-TEXT] SUCCESS: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
                        return text.strip()
                elif resp.status_code == 429:
                    print(f"      [GROQ-TEXT] Rate limited on {m}, trying next...")
                    continue
                else:
                    print(f"      [GROQ-TEXT] Failed on {m}: {resp.text[:200]}")
            except requests.Timeout:
                print(f"      [GROQ-TEXT] TIMEOUT on {m}")
            except Exception as e:
                print(f"      [GROQ-TEXT] ERROR on {m}: {e}")
        return None

    def generate_json(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        if not self.available:
            return None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        json_system = system_prompt + "\n\nIMPORTANT: Output ONLY valid JSON, no markdown fences, no extra text."
        for m in self.MODELS:
            payload = {
                "model": m,
                "messages": [
                    {"role": "system", "content": json_system},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"},
            }
            try:
                print(f"      [GROQ-JSON] Trying {m} with JSON mode...")
                t0 = time.time()
                resp = requests.post(self.URL, json=payload, headers=headers, timeout=self.TIMEOUT)
                elapsed = time.time() - t0
                print(f"      [GROQ-JSON] Response: {m} status={resp.status_code} | time={elapsed:.2f}s")
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if text:
                        parsed = json.loads(text)
                        print(f"      [GROQ-JSON] SUCCESS: Parsed JSON with {len(parsed)} keys: {list(parsed.keys())}")
                        return parsed
                elif resp.status_code == 429:
                    print(f"      [GROQ-JSON] Rate limited on {m}, trying next...")
                    continue
                else:
                    print(f"      [GROQ-JSON] Failed on {m}: {resp.text[:200]}")
            except json.JSONDecodeError as e:
                print(f"      [GROQ-JSON] JSON parse error on {m}: {e}")
            except requests.Timeout:
                print(f"      [GROQ-JSON] TIMEOUT on {m}")
            except Exception as e:
                print(f"      [GROQ-JSON] ERROR on {m}: {e}")
        return None


class ContentGenerator:
    """Generate ad copy using Groq only."""

    def __init__(self, clip_extractor):
        self.clip = clip_extractor
        self.groq_gen = GroqTextGenerator()
        print("    [CONTENT-GEN] ContentGenerator initialized - Engine: Groq")

    def generate_product_content(
        self,
        brand: str,
        category: str,
        subcategory: str,
        features: List[str],
        tagline: str,
        query: str,
        product_metadata: dict = None,
    ) -> Dict[str, Any]:
        brand_clean = brand.replace("_", " ")
        sub_clean = subcategory.replace("_", " ") if subcategory else category.replace("_", " ")
        product_desc = self._extract_product_from_query(query, brand_clean)
        display_product = product_desc if product_desc else sub_clean
        print(f"    [CONTENT-GEN] Extracted product: \"{display_product}\" from query: \"{query[:100]}\"")

        product_details = self._build_product_details(product_metadata) if product_metadata else ""
        ai_content = self._generate_all_via_ai(brand_clean, category, display_product, query, product_details)
        if ai_content:
            print(f"    [CONTENT-GEN] SUCCESS: Groq-generated content with {len(ai_content)} fields")
            return ai_content

        print("    [CONTENT-GEN] FALLBACK: Groq unavailable/failed, using minimal brand-only content")
        return {
            "product_title": f"{brand_clean} {display_product.title()}",
            "product_description": f"{brand_clean} {display_product}",
            "instagram_caption": f"{brand_clean} {display_product}",
            "whatsapp_copy": f"{brand_clean} {display_product}",
            "hashtags": [brand_clean.replace(" ", ""), display_product.replace(" ", "")],
            "headline": brand_clean,
            "tagline": "",
            "features": [],
            "cta_text": "",
        }

    @staticmethod
    def _build_product_details(metadata: dict) -> str:
        lines = []
        field_map = {
            "product_name": "Product Name",
            "product_type": "Product Type",
            "material": "Material",
            "color": "Color",
            "key_features": "Key Features",
            "target_audience": "Target Audience",
            "occasion": "Occasion",
            "pack_size": "Pack Size",
            "size_range": "Size Range",
            "flavor": "Flavor",
            "price": "Price",
            "scene_description": "Scene",
        }
        for key, label in field_map.items():
            val = metadata.get(key, "")
            if val and str(val).strip():
                lines.append(f"{label}: {val}")
        return "\n".join(lines)

    @staticmethod
    def _extract_product_from_query(query: str, brand_clean: str) -> str:
        if not query:
            return ""
        desc = query.lower()
        for word in brand_clean.lower().split():
            desc = desc.replace(word, "")
        filler = {
            "ad", "ads", "advertisement", "pamphlet", "poster", "banner",
            "create", "make", "generate", "please", "want", "need", "i",
            "a", "an", "the", "for", "with", "and", "of", "in", "on",
            "me", "my", "give", "show", "by", "it", "that", "this",
            "wearing", "worn", "using", "holding", "showing", "displaying",
            "person", "people", "man", "woman", "women", "men", "model",
            "models", "girl", "boy", "lady", "ladies",
        }
        words = [w for w in desc.split() if w not in filler]
        return " ".join(words).strip()

    def _generate_all_via_ai(
        self,
        brand: str,
        category: str,
        product: str,
        query: str,
        product_details: str = "",
    ) -> Optional[Dict[str, Any]]:
        print("    [AI-CONTENT] Attempting full JSON content generation via Groq...")
        system_prompt = (
            "You are an expert advertising copywriter. Given a brand and product, "
            "generate compelling, creative ad content. Output ONLY valid JSON with these exact keys:\n"
            "{\n"
            '  "headline": "Short catchy headline (3-6 words, no brand name)",\n'
            '  "tagline": "Brand tagline or slogan (3-8 words)",\n'
            '  "features": ["feature 1 (3-5 words)", "feature 2", "feature 3", "feature 4"],\n'
            '  "cta_text": "Call to action (2-3 words, ALL CAPS)",\n'
            '  "product_title": "Full product title with brand",\n'
            '  "product_description": "2-3 sentence product description",\n'
            '  "instagram_caption": "Engaging Instagram caption with emojis",\n'
            '  "whatsapp_copy": "Short WhatsApp promotional message",\n'
            '  "hashtags": ["tag1", "tag2", "tag3", "tag4", "tag5"]\n'
            "}\n"
        )

        user_parts = [f"Brand: {brand}", f"Category: {category}", f"Product: {product}"]
        if product_details:
            user_parts.append(f"\nProduct Details:\n{product_details}")
        else:
            user_parts.append(f"User request: {query}")
        user_parts.append("\nGenerate creative, compelling ad content for this product.")
        user_prompt = "\n".join(user_parts)

        try:
            result = self.groq_gen.generate_json(system_prompt, user_prompt)
            if result and isinstance(result, dict):
                headline = result.get("headline", "")
                generated_features = result.get("features", [])
                cta = result.get("cta_text", "")
                if headline and len(headline) > 2 and generated_features and len(generated_features) >= 2:
                    print("    [AI-CONTENT] Groq JSON SUCCESS!")
                    return {
                        "product_title": result.get("product_title", f"{brand} {product}"),
                        "product_description": result.get("product_description", ""),
                        "instagram_caption": result.get("instagram_caption", ""),
                        "whatsapp_copy": result.get("whatsapp_copy", ""),
                        "hashtags": result.get("hashtags", []),
                        "headline": headline,
                        "tagline": result.get("tagline", ""),
                        "features": generated_features[:6],
                        "cta_text": cta.upper() if cta else "",
                    }
        except Exception as e:
            print(f"    [AI-CONTENT] Groq JSON FAILED: {e}")

        try:
            headline = self.groq_gen.generate(
                "Output only a short catchy headline, no quotes, no explanation.",
                (
                    "You are an ad copywriter. Write a short catchy headline (3-6 words) "
                    "for this product ad. Output ONLY the headline, nothing else.\n\n"
                    f"Brand: {brand}, Product: {product}, Category: {category}"
                ),
            )
            if headline and 3 < len(headline) < 60:
                headline = headline.strip('"').strip("'").strip()
                return {
                    "product_title": f"{brand} {product.title()}",
                    "product_description": f"{brand} {product}",
                    "instagram_caption": f"{brand} {product}",
                    "whatsapp_copy": f"{brand} {product}",
                    "hashtags": [brand.replace(" ", "")],
                    "headline": headline,
                    "tagline": "",
                    "features": [],
                    "cta_text": "",
                }
        except Exception as e:
            print(f"    [AI-CONTENT] Groq headline fallback FAILED: {e}")

        return None

    def generate_ad_content_for_language(
        self, brand: str, category: str, subcategory: str,
        features: List[str], tagline: str, query: str, language: str
    ) -> Optional[Dict[str, str]]:
        print(f"    [CONTENT-GEN] generate_ad_content_for_language({language}) -> None (handled by Translator)")
        return None


class Translator:
    def __init__(self):
        self._translator = None
        print("    [TRANSLATOR] Translator initialized (engine: GoogleTranslator via deep_translator)")

    @staticmethod
    def _get_translator(source: str, target: str):
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source=source, target=target)
        except ImportError:
            print("    [TRANSLATOR] ERROR: deep_translator not installed")
            return None

    def translate(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        if target_lang == source_lang or target_lang == "en":
            return text
        try:
            translator = self._get_translator(source_lang, target_lang)
            if translator:
                t0 = time.time()
                result = translator.translate(text)
                elapsed = time.time() - t0
                print(f"      [TRANSLATE] {source_lang}->{target_lang} | {len(text)} chars -> {len(result)} chars | {elapsed:.2f}s")
                return result
        except Exception as e:
            print(f"      [TRANSLATE] FAILED {source_lang}->{target_lang}: {e}")
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


class ImageEnhancer:
    @staticmethod
    def enhance(
        image: Image.Image,
        brightness: float = 1.15,
        contrast: float = 1.2,
        sharpness: float = 1.3,
        color: float = 1.1,
    ) -> Image.Image:
        img = image.copy()
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        img = ImageEnhance.Color(img).enhance(color)
        return img

    @staticmethod
    def auto_enhance(image: Image.Image) -> Image.Image:
        img = image.copy().convert("RGB")
        arr = np.array(img).astype(float)
        mean_brightness = arr.mean() / 255.0
        std_val = arr.std() / 255.0

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
