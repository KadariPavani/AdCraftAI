# Image generation: Multiple free API fallbacks + gradient fallback + local PIL backgrounds.
# Fallback chain: Pollinations (3 models) → HF free models (SDXL, SD2.1) → gradient

import io
import math
import random
import time
import urllib.parse
from typing import Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# Remote Image Generator (7-tier fallback chain, all FREE, no paid keys)
# ---------------------------------------------------------------------------

class ImageGenerator:
    """Tries multiple free image generation APIs in order:
    1. Pollinations FLUX (free, no key)
    2. Pollinations Turbo (free, no key, different backend)
    3. Pollinations flux-realism (free, no key, different backend)
    4. HuggingFace FLUX.1-schnell (needs HF_TOKEN + credits)
    5. HuggingFace SDXL (FREE model, needs HF_TOKEN, NO credits needed)
    6. HuggingFace SD 2.1 (FREE model, needs HF_TOKEN, NO credits needed)
    7. Gradient fallback (local PIL)
    """

    POLLINATIONS_TIMEOUT = 120
    POLLINATIONS_RETRIES = 2

    # HuggingFace model URLs — FLUX needs credits, SDXL and SD2.1 are FREE
    HF_MODELS = [
        ("FLUX.1-schnell", "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"),
        ("Stable Diffusion XL", "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"),
        ("Stable Diffusion 2.1", "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"),
    ]

    # Pollinations models to try — different backends, different availability
    POLLINATIONS_MODELS = ["flux", "turbo", "flux-realism"]

    def __init__(self, hf_token: Optional[str] = None, **kwargs):
        self.hf_token = hf_token

    def generate(
        self, prompt: str, negative_prompt: str = "",
        width: int = 1024, height: int = 768,
    ) -> Tuple[Optional[Image.Image], str]:
        """Try all sources in order."""

        # Tier 1-3: Try all Pollinations models
        for model in self.POLLINATIONS_MODELS:
            img = self._try_pollinations(prompt, width, height, model=model)
            if img:
                return img, f"pollinations_{model}"

        # Tier 4-6: Try all HuggingFace models (FLUX → SDXL → SD2.1)
        for model_name, model_url in self.HF_MODELS:
            img = self._try_hf_inference(prompt, model_name, model_url)
            if img:
                safe_name = model_name.lower().replace(" ", "_").replace(".", "")
                return img, f"hf_{safe_name}"

        # Tier 7: Local gradient fallback
        img = self._make_gradient(width, height)
        return img, "gradient_fallback"

    def _try_pollinations(self, prompt, width, height,
                          model="flux") -> Optional[Image.Image]:
        print(f"\n    Trying Pollinations.ai ({model})...")
        encoded = urllib.parse.quote(prompt, safe="")
        seed = random.randint(1, 999999)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={width}&height={height}&model={model}&nologo=true&seed={seed}"
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

    def _try_hf_inference(self, prompt, model_name="FLUX.1-schnell",
                          model_url=None) -> Optional[Image.Image]:
        if not self.hf_token:
            print(f"\n    Skipping HF {model_name} (no token)")
            return None
        if model_url is None:
            model_url = self.HF_MODELS[0][1]
        print(f"\n    Trying HuggingFace ({model_name})...")
        try:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {"inputs": prompt}
            resp = requests.post(
                model_url, headers=headers,
                json=payload, timeout=120,
            )
            if resp.status_code == 200 and len(resp.content) > 1000:
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                print(f"      Success! Image size: {img.size}")
                return img
            elif resp.status_code == 503:
                # Model is loading, wait and retry once
                print(f"      Model loading, waiting 20s...")
                time.sleep(20)
                resp = requests.post(
                    model_url, headers=headers,
                    json=payload, timeout=120,
                )
                if resp.status_code == 200 and len(resp.content) > 1000:
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    print(f"      Success! Image size: {img.size}")
                    return img
                print(f"      Still not ready: status={resp.status_code}")
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
