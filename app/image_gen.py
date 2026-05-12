# Image generation: FLUX only (text-to-image + image-to-image).

import base64
import io
import os
import time
from typing import Dict, Optional, Tuple

import requests
from PIL import Image, ImageFilter

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None


class ImageGenerator:
    HF_TIMEOUT = 120
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

    def __init__(self, hf_token: Optional[str] = None, **kwargs):
        self.hf_token = hf_token
        raw_providers = os.getenv("HF_IMG2IMG_PROVIDERS", "fal-ai,black-forest-labs,replicate,auto")
        provider_aliases = {
            "blackforestlabs": "black-forest-labs",
            "black_forest_labs": "black-forest-labs",
            "black-forest-labs": "black-forest-labs",
        }
        providers: list[str] = []
        for p in raw_providers.split(","):
            normalized = p.strip().lower()
            if normalized:
                providers.append(provider_aliases.get(normalized, normalized))
        self.hf_img2img_providers = providers

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

    @staticmethod
    def _resize_to_target_preserving_subject(image: Image.Image, width: int, height: int) -> Image.Image:
        src = image.convert("RGB")
        if src.size == (width, height):
            return src
        scale = min(width / src.width, height / src.height)
        fit_w = max(1, int(round(src.width * scale)))
        fit_h = max(1, int(round(src.height * scale)))
        foreground = src.resize((fit_w, fit_h), Image.LANCZOS)
        if (fit_w, fit_h) == (width, height):
            return foreground
        backdrop = src.resize((width, height), Image.LANCZOS).filter(ImageFilter.GaussianBlur(20))
        x = (width - fit_w) // 2
        y = (height - fit_h) // 2
        backdrop.paste(foreground, (x, y))
        return backdrop

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
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 768,
        model_preference: Optional[str] = None,
    ) -> Tuple[Optional[Image.Image], str]:
        selected = self._resolve_model_config(model_preference)
        selected_model = selected["hf_model_id"]
        selected_label = selected["label"]
        if selected.get("task") == "image-to-image":
            selected_model = self.HF_DEFAULT_TEXT_MODEL
            selected_label = self.HF_MODEL_CONFIGS["flux1-schnell"]["label"]

        img = self._generate_hf_text_to_image(prompt, selected_model, selected_label)
        if img is None:
            return None, "hf_text2img_failed"
        return self._resize_to_target_preserving_subject(img, width, height), f"hf_{self._normalize_model_name(model_preference)}"

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1080,
        height: int = 1080,
        model_preference: Optional[str] = None,
    ) -> Tuple[Optional[Image.Image], str]:
        selected_key = self._normalize_model_name(model_preference)
        selected = self._resolve_model_config(model_preference)
        selected_model = selected["hf_model_id"]
        selected_label = selected["label"]
        if selected.get("task") != "image-to-image":
            selected_key = "flux1-kontext-dev"
            selected_model = self.HF_DEFAULT_IMG2IMG_MODEL
            selected_label = self.HF_MODEL_CONFIGS["flux1-kontext-dev"]["label"]

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

    @staticmethod
    def _decode_hf_image_response(resp: requests.Response) -> Optional[Image.Image]:
        content_type = (resp.headers.get("content-type", "") or "").lower()
        body = resp.content or b""
        if resp.status_code != 200 or not body:
            return None
        if content_type.startswith("image/"):
            try:
                return Image.open(io.BytesIO(body))
            except Exception:
                return None
        try:
            data = resp.json()
        except Exception:
            return None
        return ImageGenerator._decode_image_any(data)

    def _generate_hf_text_to_image(self, prompt: str, model_id: str, model_label: str) -> Optional[Image.Image]:
        if not self.hf_token:
            return None
        endpoint = self.HF_MODEL_ENDPOINT.format(model_id=model_id)
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {"inputs": prompt}

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.HF_TIMEOUT)
            decoded = self._decode_hf_image_response(resp)
            if decoded is not None:
                return decoded.convert("RGB")
            if resp.status_code == 503:
                time.sleep(20)
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.HF_TIMEOUT)
                decoded = self._decode_hf_image_response(resp)
                if decoded is not None:
                    return decoded.convert("RGB")
        except Exception:
            return None
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
        if not self.hf_token or InferenceClient is None:
            return None
        try:
            client = InferenceClient(token=self.hf_token)
            resized = image.convert("RGB").resize((width, height), Image.LANCZOS)
            for provider in self.hf_img2img_providers:
                kwargs = {
                    "prompt": prompt,
                    "image": resized,
                    "model": model_id,
                    "negative_prompt": negative_prompt,
                    "provider": provider,
                }
                generated = client.image_to_image(**kwargs)
                decoded = self._decode_image_any(generated)
                if decoded is not None:
                    return self._resize_to_target_preserving_subject(decoded.convert("RGB"), width, height)
        except Exception:
            return None
        return None
