# Multi-source logo fetcher (free APIs, no keys required).

import io
import re
import urllib.parse
from collections import Counter
from typing import List, Optional

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont


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
        print(f"    [LOGO] Fetching logo for: {brand_name}")
        domain = cls._get_domain(brand_name)
        print(f"    [LOGO] Resolved domain: {domain}")
        print(f"    [LOGO] 5-source fallback chain starting...")

        # Source 1: Scrape website for apple-touch-icon (best: up to 512px)
        print(f"    [LOGO] Source 1: Website apple-touch-icon ({domain})")
        logo = cls._try_website_icon(domain)
        if logo and min(logo.size) >= 64:
            print(f"    [LOGO] Source 1 SUCCESS: {logo.size}")
            cleaned = cls._remove_background(logo)
            if cleaned:
                result = cls._polish_logo(cleaned)
                print(f"    [LOGO] Final logo: {result.size} | Mode: {result.mode}")
                return result
        else:
            print(f"    [LOGO] Source 1 FAILED{f' (too small: {logo.size})' if logo else ''}")

        # Source 2: Google faviconV2 (reliable, up to 256px)
        print(f"    [LOGO] Source 2: Google faviconV2")
        logo = cls._try_google_favicon(domain)
        if logo and min(logo.size) >= 48:
            print(f"    [LOGO] Source 2 SUCCESS: {logo.size}")
            cleaned = cls._remove_background(logo)
            if cleaned:
                result = cls._polish_logo(cleaned)
                print(f"    [LOGO] Final logo: {result.size} | Mode: {result.mode}")
                return result
        else:
            print(f"    [LOGO] Source 2 FAILED{f' (too small: {logo.size})' if logo else ''}")

        # Source 3: icon.horse (good backup)
        print(f"    [LOGO] Source 3: icon.horse")
        logo = cls._try_icon_horse(domain)
        if logo and min(logo.size) >= 48:
            print(f"    [LOGO] Source 3 SUCCESS: {logo.size}")
            cleaned = cls._remove_background(logo)
            if cleaned:
                result = cls._polish_logo(cleaned)
                print(f"    [LOGO] Final logo: {result.size} | Mode: {result.mode}")
                return result
        else:
            print(f"    [LOGO] Source 3 FAILED{f' (too small: {logo.size})' if logo else ''}")

        # Source 4: DuckDuckGo icon
        print(f"    [LOGO] Source 4: DuckDuckGo icon")
        logo = cls._try_duckduckgo_icon(domain)
        if logo and min(logo.size) >= 48:
            print(f"    [LOGO] Source 4 SUCCESS: {logo.size}")
            cleaned = cls._remove_background(logo)
            if cleaned:
                result = cls._polish_logo(cleaned)
                print(f"    [LOGO] Final logo: {result.size} | Mode: {result.mode}")
                return result
        else:
            print(f"    [LOGO] Source 4 FAILED{f' (too small: {logo.size})' if logo else ''}")

        # Final fallback: text-based logo
        print(f"    [LOGO] Source 5: Text-based logo generator (FALLBACK)")
        result = cls._generate_text_logo(brand_name)
        print(f"    [LOGO] Generated text logo: {result.size}")
        return result

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
