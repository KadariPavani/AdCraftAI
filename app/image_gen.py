# Image generation: HuggingFace FLUX.1-schnell (primary), Pollinations, xAI Grok, then gradient fallback.
# Multi-provider chain for robust prompt-based ad image generation.

import io
import math
import os
import random
import time
import base64
from typing import Dict, Optional, Tuple
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
try:
    from huggingface_hub import InferenceClient
except Exception:  # optional runtime dependency
    InferenceClient = None


# ---------------------------------------------------------------------------
# Remote Image Generator — HF FLUX + Pollinations + xAI Grok + gradient fallback
# ---------------------------------------------------------------------------

class ImageGenerator:
    """Image generation with provider chain + one local fallback:
    1. HuggingFace FLUX.1-schnell (primary — fast, high quality, needs HF_TOKEN)
    2. xAI Grok image API (secondary, needs XAI_API_KEY)
    3. Pollinations image API (tertiary, no key required)
    4. Gradient fallback (local PIL — always works, no API needed)
    """

    HF_MODEL_NAME = "FLUX.1-schnell"
    HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    HF_TIMEOUT = 120
    POLLINATIONS_URL = "https://image.pollinations.ai/prompt/"
    POLLINATIONS_TIMEOUT = 90
    XAI_IMAGE_URL = "https://api.x.ai/v1/images/generations"
    XAI_TIMEOUT = 90
    HF_MODEL_ENDPOINT = "https://router.huggingface.co/hf-inference/models/{model_id}"
    HF_DEFAULT_TEXT_MODEL = "black-forest-labs/FLUX.1-schnell"
    HF_DEFAULT_IMG2IMG_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
    HF_MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
        "flux1-schnell": {
            "hf_model_id": "black-forest-labs/FLUX.1-schnell",
            "task": "text-to-image",
            "label": "FLUX.1-schnell",
        },
        "flux1-kontext-dev": {
            "hf_model_id": "black-forest-labs/FLUX.1-Kontext-dev",
            "task": "image-to-image",
            "label": "FLUX.1 Kontext [dev]",
        },
        "flux1-redux-dev": {
            "hf_model_id": "black-forest-labs/FLUX.1-Redux-dev",
            "task": "image-to-image",
            "label": "FLUX.1 Redux [dev]",
        },
        "flux2-dev": {
            "hf_model_id": "black-forest-labs/FLUX.2-dev",
            "task": "image-to-image",
            "label": "FLUX.2 [dev]",
        },
        "flux2-pro": {
            "hf_model_id": "black-forest-labs/FLUX.2-pro",
            "task": "image-to-image",
            "label": "FLUX.2 [pro]",
        },
        "flux2-max": {
            "hf_model_id": "black-forest-labs/FLUX.2-max",
            "task": "image-to-image",
            "label": "FLUX.2 [max]",
        },
    }

    def __init__(
        self,
        hf_token: Optional[str] = None,
        together_key: Optional[str] = None,
        xai_api_key: Optional[str] = None,
        xai_image_model: str = "grok-2-image-1212",
        **kwargs
    ):
        self.hf_token = hf_token
        self.together_key = together_key
        self.xai_api_key = xai_api_key
        self.xai_image_model = xai_image_model
        raw_providers = os.getenv(
            "HF_IMG2IMG_PROVIDERS",
            "fal-ai,blackforestlabs,replicate,together,auto"
        )
        self.hf_img2img_providers = [p.strip() for p in raw_providers.split(",") if p.strip()]

    @staticmethod
    def _decode_image_any(payload) -> Optional[Image.Image]:
        if payload is None:
            return None
        if isinstance(payload, Image.Image):
            return payload
        if isinstance(payload, (bytes, bytearray)):
            try:
                return Image.open(io.BytesIO(payload))
            except Exception:
                return None
        if isinstance(payload, dict):
            for key in ("image", "b64_json"):
                raw = payload.get(key)
                if isinstance(raw, str):
                    encoded = raw.split(",", 1)[1] if raw.startswith("data:image") and "," in raw else raw
                    try:
                        return Image.open(io.BytesIO(base64.b64decode(encoded)))
                    except Exception:
                        continue
        if isinstance(payload, list):
            for item in payload:
                img = ImageGenerator._decode_image_any(item)
                if img is not None:
                    return img
        return None

    @classmethod
    def _normalize_model_name(cls, model_name: Optional[str]) -> str:
        if not model_name:
            return "flux1-schnell"
        normalized = model_name.strip().lower().replace("_", "-")
        aliases = {
            "flux.1-schnell": "flux1-schnell",
            "flux.1-kontext-dev": "flux1-kontext-dev",
            "flux.1-redux-dev": "flux1-redux-dev",
            "flux.2-dev": "flux2-dev",
            "flux.2-pro": "flux2-pro",
            "flux.2-max": "flux2-max",
        }
        return aliases.get(normalized, normalized)

    @classmethod
    def _resolve_model_config(cls, model_name: Optional[str]) -> Dict[str, str]:
        key = cls._normalize_model_name(model_name)
        return cls.HF_MODEL_CONFIGS.get(key, cls.HF_MODEL_CONFIGS["flux1-schnell"])

    def generate(
        self, prompt: str, negative_prompt: str = "",
        width: int = 1024, height: int = 768,
        model_preference: Optional[str] = None,
    ) -> Tuple[Optional[Image.Image], str]:
        """Generate image via provider chain; fallback to gradient if all fail."""
        print(f"\n    [IMG-GEN] Starting image generation")
        print(f"    [IMG-GEN] Target: {width}x{height}")
        print(f"    [IMG-GEN] Prompt ({len(prompt)} chars): \"{prompt[:100]}...\"")
        selected = self._resolve_model_config(model_preference)
        selected_model = selected["hf_model_id"]
        selected_label = selected["label"]
        if selected.get("task") == "image-to-image":
            print(f"    [IMG-GEN] Requested model is image-to-image only: {selected_label}")
            print(f"    [IMG-GEN] Falling back to text-capable model: {self.HF_MODEL_NAME}")
            selected_model = self.HF_DEFAULT_TEXT_MODEL
            selected_label = self.HF_MODEL_NAME

        # Primary: HuggingFace FLUX.1-schnell
        print(f"\n    [IMG-GEN] === PRIMARY: HuggingFace {selected_label} ===")
        img = self._generate_hf_text_to_image(prompt, selected_model, selected_label)
        if img:
            print(f"    [IMG-GEN] SUCCESS! Image generated via {selected_label}")
            return img, f"hf_{self._normalize_model_name(model_preference)}"

        # Secondary: xAI Grok image API
        print(f"\n    [IMG-GEN] === SECONDARY: xAI Grok Image ===")
        img = self._generate_xai_grok_image(prompt, width, height)
        if img:
            print(f"    [IMG-GEN] SUCCESS! Image generated via xAI Grok")
            return img, "xai_grok_image"

        # Tertiary: Pollinations image API
        print(f"\n    [IMG-GEN] === TERTIARY: Pollinations Image ===")
        img = self._generate_pollinations(prompt, negative_prompt, width, height)
        if img:
            print(f"    [IMG-GEN] SUCCESS! Image generated via Pollinations")
            return img, "pollinations_image"

        # Fallback: Local gradient
        print(f"\n    [IMG-GEN] === FALLBACK: Gradient (Local PIL) ===")
        print(f"    [IMG-GEN] All remote providers failed, creating local gradient...")
        img = self._make_gradient(width, height)
        print(f"    [IMG-GEN] Gradient generated: {img.size}")
        return img, "gradient_fallback"

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1080,
        height: int = 1080,
        model_preference: Optional[str] = None,
    ) -> Tuple[Optional[Image.Image], str]:
        """Edit/enhance an existing image using FLUX image-to-image models."""
        selected_key = self._normalize_model_name(model_preference)
        selected = self._resolve_model_config(model_preference)
        selected_model = selected["hf_model_id"]
        selected_label = selected["label"]
        if selected.get("task") != "image-to-image":
            selected_key = "flux1-kontext-dev"
            selected_model = self.HF_DEFAULT_IMG2IMG_MODEL
            selected_label = self.HF_MODEL_CONFIGS["flux1-kontext-dev"]["label"]

        print(f"\n    [IMG-EDIT] Starting image-to-image enhancement")
        print(f"    [IMG-EDIT] Model: {selected_label}")
        print(f"    [IMG-EDIT] Input size: {image.size} | Target: {width}x{height}")
        edited = self._generate_hf_image_to_image(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=selected_model,
            model_label=selected_label,
            width=width,
            height=height,
        )
        if edited:
            return edited, f"hf_{selected_key}_img2img"
        return None, "img2img_failed"

    def _generate_hf_text_to_image(self, prompt: str, model_id: str, model_label: str) -> Optional[Image.Image]:
        """Generate image using HuggingFace FLUX inference API."""
        if not self.hf_token:
            print(f"      [HF-FLUX] SKIPPED: No HF_TOKEN set")
            print(f"      [HF-FLUX] To enable: set HF_TOKEN in .env file")
            return None

        endpoint = self.HF_MODEL_ENDPOINT.format(model_id=model_id)
        print(f"      [HF-FLUX] Model: {model_label}")
        print(f"      [HF-FLUX] URL: {endpoint}")
        print(f"      [HF-FLUX] Token: {self.hf_token[:8]}...{self.hf_token[-4:]}")
        print(f"      [HF-FLUX] Timeout: {self.HF_TIMEOUT}s")

        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {"inputs": prompt}

        try:
            print(f"      [HF-FLUX] Sending inference request...")
            t0 = time.time()
            resp = requests.post(
                endpoint, headers=headers,
                json=payload, timeout=self.HF_TIMEOUT,
            )
            elapsed = time.time() - t0
            print(f"      [HF-FLUX] Response: status={resp.status_code} | size={len(resp.content):,} bytes | time={elapsed:.1f}s")

            decoded = self._decode_hf_image_response(resp)
            if resp.status_code == 200 and decoded is not None:
                img = decoded.convert("RGB")
                print(f"      [HF-FLUX] SUCCESS! Image decoded: {img.size} | Mode: {img.mode}")
                return img

            elif resp.status_code == 503:
                # Model cold start — wait and retry once
                print(f"      [HF-FLUX] Model is loading (503)... waiting 20s for cold start")
                time.sleep(20)
                print(f"      [HF-FLUX] Retrying after cold start...")
                t0 = time.time()
                resp = requests.post(
                    endpoint, headers=headers,
                    json=payload, timeout=self.HF_TIMEOUT,
                )
                elapsed = time.time() - t0
                print(f"      [HF-FLUX] Retry response: status={resp.status_code} | size={len(resp.content):,} bytes | time={elapsed:.1f}s")
                decoded = self._decode_hf_image_response(resp)
                if resp.status_code == 200 and decoded is not None:
                    img = decoded.convert("RGB")
                    print(f"      [HF-FLUX] SUCCESS after retry! Image: {img.size}")
                    return img
                print(f"      [HF-FLUX] Still not ready after retry: status={resp.status_code}")

            elif resp.status_code == 401:
                print(f"      [HF-FLUX] UNAUTHORIZED (401): Token invalid or expired")
            elif resp.status_code == 429:
                print(f"      [HF-FLUX] RATE LIMITED (429): Too many requests")
            elif resp.status_code == 402:
                print(f"      [HF-FLUX] PAYMENT REQUIRED (402): Insufficient credits")
            else:
                print(f"      [HF-FLUX] FAILED: status={resp.status_code}")
                try:
                    print(f"      [HF-FLUX] Error body: {resp.text[:200]}")
                except Exception:
                    pass

        except requests.Timeout:
            print(f"      [HF-FLUX] TIMEOUT after {self.HF_TIMEOUT}s")
        except Exception as e:
            print(f"      [HF-FLUX] ERROR: {type(e).__name__}: {e}")

        return None

    def _generate_hf_image_to_image(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        model_id: str,
        model_label: str,
        width: int,
        height: int,
    ) -> Optional[Image.Image]:
        if not self.hf_token:
            print("      [HF-IMG2IMG] SKIPPED: No HF_TOKEN set")
            return None

        client_img = self._generate_hf_image_to_image_via_client(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=model_id,
            width=width,
            height=height,
        )
        if client_img is not None:
            return client_img

        endpoint = self.HF_MODEL_ENDPOINT.format(model_id=model_id)
        print(f"      [HF-IMG2IMG] Model: {model_label}")
        print(f"      [HF-IMG2IMG] URL: {endpoint}")
        print(f"      [HF-IMG2IMG] Timeout: {self.HF_TIMEOUT}s")

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        payload = {
            "inputs": image_b64,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt[:900] if negative_prompt else "",
                "guidance_scale": 2.5,
                "num_inference_steps": 30,
                "target_size": {"width": width, "height": height},
            },
        }
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

        try:
            t0 = time.time()
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.HF_TIMEOUT)
            elapsed = time.time() - t0
            print(f"      [HF-IMG2IMG] Response: status={resp.status_code} | size={len(resp.content):,} bytes | time={elapsed:.1f}s")
            if resp.status_code != 200:
                try:
                    print(f"      [HF-IMG2IMG] Error body: {resp.text[:280]}")
                except Exception:
                    pass
                return None

            decoded = self._decode_hf_image_response(resp)
            if decoded is None:
                print("      [HF-IMG2IMG] Failed to decode image payload")
                return None
            print(f"      [HF-IMG2IMG] SUCCESS! Image decoded: {decoded.size}")
            return decoded.convert("RGB")
        except requests.Timeout:
            print(f"      [HF-IMG2IMG] TIMEOUT after {self.HF_TIMEOUT}s")
        except Exception as e:
            print(f"      [HF-IMG2IMG] ERROR: {type(e).__name__}: {e}")
        return None

    def _generate_hf_image_to_image_via_client(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        model_id: str,
        width: int,
        height: int,
    ) -> Optional[Image.Image]:
        if InferenceClient is None:
            print("      [HF-IMG2IMG-CLIENT] huggingface_hub not available, skipping provider client")
            return None

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        providers = self.hf_img2img_providers or ["auto"]
        print(f"      [HF-IMG2IMG-CLIENT] Providers: {providers}")
        for provider in providers:
            client_kwargs = {"api_key": self.hf_token}
            if provider != "auto":
                client_kwargs["provider"] = provider
            try:
                print(f"      [HF-IMG2IMG-CLIENT] Trying provider: {provider}")
                client = InferenceClient(**client_kwargs)
                t0 = time.time()
                response = client.image_to_image(
                    image_bytes,
                    prompt=prompt,
                    model=model_id,
                    negative_prompt=negative_prompt[:900] if negative_prompt else None,
                    guidance_scale=2.5,
                    num_inference_steps=30,
                )
                elapsed = time.time() - t0
                decoded = self._decode_image_any(response)
                if decoded is None:
                    print(f"      [HF-IMG2IMG-CLIENT] Provider {provider} returned undecodable payload")
                    continue
                decoded = decoded.convert("RGB")
                if decoded.size != (width, height):
                    decoded = decoded.resize((width, height), Image.LANCZOS)
                print(f"      [HF-IMG2IMG-CLIENT] SUCCESS via {provider}: {decoded.size} | time={elapsed:.1f}s")
                return decoded
            except TypeError:
                # Backward/alternate method signature fallback
                try:
                    print(f"      [HF-IMG2IMG-CLIENT] Retrying provider {provider} with alternate signature")
                    client = InferenceClient(**client_kwargs)
                    t0 = time.time()
                    response = client.image_to_image(
                        inputs=image_bytes,
                        model=model_id,
                        parameters={
                            "prompt": prompt,
                            "negative_prompt": negative_prompt[:900] if negative_prompt else "",
                            "guidance_scale": 2.5,
                            "num_inference_steps": 30,
                            "target_size": {"width": width, "height": height},
                        },
                    )
                    elapsed = time.time() - t0
                    decoded = self._decode_image_any(response)
                    if decoded is None:
                        print(f"      [HF-IMG2IMG-CLIENT] Alternate signature undecodable for {provider}")
                        continue
                    decoded = decoded.convert("RGB")
                    if decoded.size != (width, height):
                        decoded = decoded.resize((width, height), Image.LANCZOS)
                    print(f"      [HF-IMG2IMG-CLIENT] SUCCESS via {provider} (alt): {decoded.size} | time={elapsed:.1f}s")
                    return decoded
                except Exception as e:
                    print(f"      [HF-IMG2IMG-CLIENT] Provider {provider} failed (alt): {type(e).__name__}: {e}")
            except Exception as e:
                print(f"      [HF-IMG2IMG-CLIENT] Provider {provider} failed: {type(e).__name__}: {e}")
        print("      [HF-IMG2IMG-CLIENT] All providers failed, falling back to legacy endpoint call")
        return None

    @staticmethod
    def _decode_hf_image_response(resp: requests.Response) -> Optional[Image.Image]:
        content_type = (resp.headers.get("content-type") or "").lower()
        try:
            if content_type.startswith("image/"):
                return Image.open(io.BytesIO(resp.content))

            data = resp.json()
            candidates = []
            if isinstance(data, dict):
                if isinstance(data.get("image"), str):
                    candidates.append(data["image"])
                images = data.get("images")
                if isinstance(images, list):
                    for img in images:
                        if isinstance(img, str):
                            candidates.append(img)
                        elif isinstance(img, dict):
                            maybe = img.get("b64_json") or img.get("image")
                            if isinstance(maybe, str):
                                candidates.append(maybe)
                if isinstance(data.get("data"), list):
                    for item in data["data"]:
                        if isinstance(item, dict):
                            maybe = item.get("b64_json") or item.get("image")
                            if isinstance(maybe, str):
                                candidates.append(maybe)

            for raw in candidates:
                payload = raw.split(",", 1)[1] if raw.startswith("data:image") and "," in raw else raw
                try:
                    decoded = base64.b64decode(payload)
                    if len(decoded) > 1000:
                        return Image.open(io.BytesIO(decoded))
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def _generate_pollinations(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 768,
    ) -> Optional[Image.Image]:
        """Generate image using Pollinations image API (no API key required)."""
        try:
            seed = int(time.time()) % 100000
            safe_prompt = quote(prompt, safe="")
            url = f"{self.POLLINATIONS_URL}{safe_prompt}"
            params = {
                "width": width,
                "height": height,
                "seed": seed,
                "model": "flux",
                "nologo": "true",
            }
            if negative_prompt:
                params["negative_prompt"] = negative_prompt[:900]

            print(f"      [POLL-IMG] URL: {self.POLLINATIONS_URL}<encoded-prompt>")
            print(f"      [POLL-IMG] Params: model=flux, size={width}x{height}, seed={seed}")
            t0 = time.time()
            resp = requests.get(url, params=params, timeout=self.POLLINATIONS_TIMEOUT)
            elapsed = time.time() - t0
            print(f"      [POLL-IMG] Response: status={resp.status_code} | size={len(resp.content):,} bytes | time={elapsed:.1f}s")

            if resp.status_code == 200 and len(resp.content) > 1000:
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                print(f"      [POLL-IMG] SUCCESS! Image decoded: {img.size} | Mode: {img.mode}")
                return img

            print(f"      [POLL-IMG] FAILED: status={resp.status_code}")
            return None
        except requests.Timeout:
            print(f"      [POLL-IMG] TIMEOUT after {self.POLLINATIONS_TIMEOUT}s")
        except Exception as e:
            print(f"      [POLL-IMG] ERROR: {type(e).__name__}: {e}")
        return None

    def _generate_xai_grok_image(self, prompt: str, width: int = 1024, height: int = 768) -> Optional[Image.Image]:
        """Generate image using xAI Grok image API if XAI_API_KEY is configured."""
        if not self.xai_api_key:
            print("      [XAI-IMG] SKIPPED: No XAI_API_KEY set")
            return None

        try:
            headers = {
                "Authorization": f"Bearer {self.xai_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.xai_image_model,
                "prompt": prompt,
                "size": f"{width}x{height}",
            }

            print(f"      [XAI-IMG] URL: {self.XAI_IMAGE_URL}")
            print(f"      [XAI-IMG] Model: {self.xai_image_model} | Size: {width}x{height}")
            t0 = time.time()
            resp = requests.post(self.XAI_IMAGE_URL, json=payload, headers=headers, timeout=self.XAI_TIMEOUT)
            elapsed = time.time() - t0
            print(f"      [XAI-IMG] Response: status={resp.status_code} | time={elapsed:.1f}s")
            if resp.status_code != 200:
                print(f"      [XAI-IMG] FAILED: {resp.text[:200]}")
                return None

            data = resp.json()
            candidates = data.get("data", [])
            if not candidates:
                print("      [XAI-IMG] FAILED: No image data returned")
                return None

            first = candidates[0]
            if first.get("url"):
                img_resp = requests.get(first["url"], timeout=self.XAI_TIMEOUT)
                if img_resp.status_code == 200 and len(img_resp.content) > 1000:
                    img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                    print(f"      [XAI-IMG] SUCCESS via URL: {img.size}")
                    return img
                print(f"      [XAI-IMG] URL download failed: status={img_resp.status_code}")
                return None

            if first.get("b64_json"):
                raw = base64.b64decode(first["b64_json"])
                if len(raw) > 1000:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    print(f"      [XAI-IMG] SUCCESS via b64_json: {img.size}")
                    return img

            print("      [XAI-IMG] FAILED: Unsupported response payload")
        except requests.Timeout:
            print(f"      [XAI-IMG] TIMEOUT after {self.XAI_TIMEOUT}s")
        except Exception as e:
            print(f"      [XAI-IMG] ERROR: {type(e).__name__}: {e}")
        return None

    def _make_gradient(self, width=1024, height=768, colors=None) -> Image.Image:
        """Create a gradient background using brand colors (local, no API)."""
        print(f"      [GRADIENT] Creating gradient background: {width}x{height}")
        print(f"      [GRADIENT] Input colors: {colors}")
        if not colors or len(colors) < 2:
            print(f"      [GRADIENT] Using default dark gradient colors")
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
        print(f"      [GRADIENT] Generated: {colors[0]} -> {colors[1]}")
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
        print(f"    [LOCAL-IMG] Generating local ad background")
        print(f"    [LOCAL-IMG] Category: {category} | Subcategory: {subcategory}")
        print(f"    [LOCAL-IMG] Brand: {brand_name or 'N/A'}")
        print(f"    [LOCAL-IMG] Accent: {accent_hex} | Secondary: {secondary_hex}")
        accent = self._hex_to_rgb(accent_hex) if isinstance(accent_hex, str) else accent_hex
        secondary = self._hex_to_rgb(secondary_hex) if isinstance(secondary_hex, str) else secondary_hex
        theme = self.CATEGORY_THEMES.get(
            subcategory, self.CATEGORY_THEMES.get(category, "modern"))
        print(f"    [LOCAL-IMG] Selected theme: {theme}")
        print(f"    [LOCAL-IMG] Canvas size: {self.WIDTH}x{self.HEIGHT}")

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
        print(f"    [LOCAL-IMG] Rendering theme: {theme}...")
        img = renderers.get(theme, self._render_modern)(accent, secondary)
        img = ImageEnhance.Sharpness(img).enhance(1.15)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        print(f"    [LOCAL-IMG] Local image generated: {img.size} | Mode: {img.mode}")
        return img

    # -- Shared drawing utilities --

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

    # -- AQUA: Water / Beverages --
    def _render_aqua(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        canvas = self._linear_gradient(W, H, (0, 50, 120), (0, 120, 200)).convert("RGBA")

        glow = self._radial_gradient(W, H, W // 2, H // 3, int(W * 0.6),
                                     (100, 200, 255), 50)
        canvas = Image.alpha_composite(canvas, glow)
        glow2 = self._radial_gradient(W, H, W // 3, int(H * 0.7),
                                      int(W * 0.4), accent, 30)
        canvas = Image.alpha_composite(canvas, glow2)

        canvas = self._draw_wave_lines(canvas, H // 4, 30, (150, 220, 255),
                                       count=4, thickness=2)
        canvas = self._draw_wave_lines(canvas, int(H * 0.65), 20,
                                       (100, 180, 240), count=3, thickness=1)

        canvas = self._draw_light_rays(canvas, (W // 2, -50),
                                       (180, 220, 255), count=10, max_alpha=25)

        canvas = self._draw_droplets(canvas, 60, (180, 230, 255), size_range=(2, 15))
        canvas = self._draw_droplets(canvas, 15, (200, 240, 255), size_range=(15, 35))

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

    # -- DYNAMIC: Sports / Fashion / Footwear --
    def _render_dynamic(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        dark = tuple(max(0, c - 80) for c in accent)
        canvas = self._linear_gradient(W, H, (20, 20, 30), dark).convert("RGBA")

        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.polygon([(0, int(H * 0.3)), (W, 0), (W, int(H * 0.15)),
                       (0, int(H * 0.45))], fill=(*accent, 40))
        draw.polygon([(0, int(H * 0.55)), (W, int(H * 0.25)),
                       (W, int(H * 0.35)), (0, int(H * 0.65))],
                     fill=(*secondary, 25))
        canvas = Image.alpha_composite(canvas, layer)

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

        glow = self._radial_gradient(W, H, int(W * 0.6), int(H * 0.4),
                                     int(W * 0.5), accent, 40)
        canvas = Image.alpha_composite(canvas, glow)

        layer3 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d3 = ImageDraw.Draw(layer3)
        for _ in range(8):
            cx, cy = rng.randint(0, W), rng.randint(0, H)
            size = rng.randint(50, 200)
            d3.polygon([(cx, cy - size), (cx - size, cy + size // 2),
                        (cx + size, cy + size // 2)],
                       outline=(*accent, rng.randint(10, 30)))
        canvas = Image.alpha_composite(canvas, layer3)

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

    # -- TECH: Mobile / Computers --
    def _render_tech(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        canvas = self._linear_gradient(W, H, (8, 8, 20), (15, 15, 35)).convert("RGBA")

        canvas = self._draw_hexagon_grid(canvas, accent, spacing=70, alpha=18)

        glow = self._radial_gradient(W, H, W // 2, H // 2, int(W * 0.45),
                                     accent, 35)
        canvas = Image.alpha_composite(canvas, glow)
        glow2 = self._radial_gradient(W, H, int(W * 0.2), int(H * 0.7),
                                      int(W * 0.3), secondary, 20)
        canvas = Image.alpha_composite(canvas, glow2)

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

        layer2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(layer2)
        for _ in range(50):
            px, py = rng.randint(0, W), rng.randint(0, H)
            pr = rng.randint(1, 3)
            d2.ellipse([px - pr, py - pr, px + pr, py + pr],
                       fill=(*accent, rng.randint(30, 100)))
        canvas = Image.alpha_composite(canvas, layer2)

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

        layer3 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d3 = ImageDraw.Draw(layer3)
        for y in range(0, H, 3):
            d3.line([(0, y), (W, y)], fill=(0, 0, 0, 8))
        canvas = Image.alpha_composite(canvas, layer3)

        return canvas.convert("RGB")

    # -- LUXURY: Jewelry / Watches --
    def _render_luxury(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        canvas = self._linear_gradient(W, H, (10, 5, 15), (25, 15, 35)).convert("RGBA")

        gold = accent if sum(accent) > 200 else (212, 175, 55)

        glow = self._radial_gradient(W, H, W // 2, H // 2, int(W * 0.5),
                                     gold, 25)
        canvas = Image.alpha_composite(canvas, glow)

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

        canvas = self._draw_sparkles(canvas, 45, gold)

        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt(((X - W / 2) / (W * 0.6)) ** 2
                       + ((Y - H / 2) / (H * 0.6)) ** 2)
        alpha_vig = np.clip((dist - 0.5) * 200, 0, 150).astype(np.uint8)
        vig = np.zeros((H, W, 4), dtype=np.uint8)
        vig[:, :, 3] = alpha_vig
        canvas = Image.alpha_composite(canvas,
                                       Image.fromarray(vig, "RGBA"))

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

    # -- POWER: Automotive --
    def _render_power(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        canvas = self._linear_gradient(W, H, (20, 20, 25), (40, 40, 50)).convert("RGBA")

        sheen = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sheen)
        for y in range(int(H * 0.3), int(H * 0.5)):
            t = (y - int(H * 0.3)) / (H * 0.2)
            b = int(40 * math.sin(t * math.pi))
            sd.line([(0, y), (W, y)], fill=(b, b, b + 5, 25))
        canvas = Image.alpha_composite(canvas, sheen)

        glow = self._radial_gradient(W, H, int(W * 0.6), int(H * 0.3),
                                     int(W * 0.4), accent, 35)
        canvas = Image.alpha_composite(canvas, glow)

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

        layer2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(layer2)
        d2.polygon([(int(W * 0.6), 0), (W, 0), (W, int(H * 0.4)),
                    (int(W * 0.7), int(H * 0.3))], fill=(*accent, 18))
        d2.polygon([(0, int(H * 0.7)), (int(W * 0.3), H), (0, H)],
                   fill=(*secondary, 14))
        canvas = Image.alpha_composite(canvas, layer2)

        grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(grid)
        for x in range(int(W * 0.7), W, 20):
            for y in range(int(H * 0.65), H, 20):
                gd.rectangle([x, y, x + 8, y + 3], fill=(*accent, 12))
        canvas = Image.alpha_composite(canvas, grid)

        flare = self._radial_gradient(W, H, int(W * 0.7), int(H * 0.25),
                                      120, (255, 255, 240), 50)
        canvas = Image.alpha_composite(canvas, flare)

        canvas = Image.alpha_composite(canvas, self._noise_layer(W, H, 5))
        return canvas.convert("RGB")

    # -- ELEGANT: Beauty / Skincare --
    def _render_elegant(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        light = tuple(min(255, c + 120) for c in accent)
        canvas = self._linear_gradient(W, H, (255, 240, 245), light).convert("RGBA")

        glow = self._radial_gradient(W, H, W // 2, int(H * 0.4),
                                     int(W * 0.6), (255, 200, 220), 40)
        canvas = Image.alpha_composite(canvas, glow)

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

        petals = Image.new("RGBA", (W, H), (0, 0, 0, 0))
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

    # -- CORPORATE: Banks / Insurance --
    def _render_corporate(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        canvas = self._linear_gradient(W, H, (240, 245, 255),
                                       (200, 215, 240)).convert("RGBA")

        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.ellipse([int(W * 0.55), int(-H * 0.2),
                      int(W * 1.2), int(H * 0.5)],
                     fill=(*accent, 12), outline=(*accent, 20))
        draw.ellipse([int(-W * 0.1), int(H * 0.6),
                      int(W * 0.3), int(H * 1.1)],
                     fill=(*secondary, 10))
        canvas = Image.alpha_composite(canvas, layer)

        layer2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(layer2)
        for x in range(0, W, 80):
            d2.line([(x, 0), (x, H)], fill=(*accent, 8), width=1)
        for y in range(0, H, 80):
            d2.line([(0, y), (W, y)], fill=(*accent, 8), width=1)
        canvas = Image.alpha_composite(canvas, layer2)

        bar = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(bar).rectangle([0, 0, int(W * 0.03), H],
                                       fill=(*accent, 60))
        canvas = Image.alpha_composite(canvas, bar)

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

    # -- MODERN: Default --
    def _render_modern(self, accent, secondary):
        W, H = self.WIDTH, self.HEIGHT
        dark_accent = tuple(max(0, c - 60) for c in accent)
        canvas = self._linear_gradient(W, H, dark_accent, (20, 20, 30)).convert("RGBA")

        for cx, cy, rf, c, a in [
            (0.2, 0.2, 0.5, accent, 40),
            (0.8, 0.3, 0.4, secondary, 30),
            (0.5, 0.8, 0.45, tuple(max(0, int(v * 0.7)) for v in accent), 25),
        ]:
            g = self._radial_gradient(W, H, int(W * cx), int(H * cy),
                                      int(max(W, H) * rf), c, a)
            canvas = Image.alpha_composite(canvas, g)

        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        m, s = 30, 80
        c = (*accent, 60)
        for cx, cy, dx, dy in [(m, m, 1, 1), (W - m, m, -1, 1),
                                (m, H - m, 1, -1), (W - m, H - m, -1, -1)]:
            draw.line([(cx, cy), (cx + s * dx, cy)], fill=c, width=2)
            draw.line([(cx, cy), (cx, cy + s * dy)], fill=c, width=2)
        canvas = Image.alpha_composite(canvas, layer)

        layer2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(layer2)
        for offset in range(-H, W + H, 45):
            d2.line([(offset, 0), (offset + H, H)],
                    fill=(*accent, 12), width=1)
        canvas = Image.alpha_composite(canvas, layer2)

        dots = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dd = ImageDraw.Draw(dots)
        for x in range(50, int(W * 0.3), 28):
            for y in range(50, int(H * 0.25), 28):
                dd.ellipse([x - 1, y - 1, x + 1, y + 1],
                           fill=(*accent, 25))
        canvas = Image.alpha_composite(canvas, dots)

        canvas = Image.alpha_composite(canvas, self._noise_layer(W, H, 5))
        return canvas.convert("RGB")
