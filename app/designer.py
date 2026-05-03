# Professional ad designer with 6 theme templates and bilingual support.
# Theme selection and visual parameters are derived from analyzing retrieved
# dataset reference images — not from hardcoded category-to-theme mappings.

import math
import os
import platform
import random
import numpy as np
from typing import Dict, List
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


# ---------------------------------------------------------------------------
# Cross-platform font resolution (Windows vs Linux/Docker)
# ---------------------------------------------------------------------------
def _is_windows():
    return platform.system() == "Windows"


def _get_latin_fonts():
    if _is_windows():
        return {
            "headline": "C:/Windows/Fonts/segoeuib.ttf",
            "body": "C:/Windows/Fonts/segoeui.ttf",
            "accent": "C:/Windows/Fonts/georgiai.ttf",
            "display": "C:/Windows/Fonts/impact.ttf",
            "elegant": "C:/Windows/Fonts/times.ttf",
            "modern": "C:/Windows/Fonts/calibri.ttf",
        }
    else:
        # Linux / Docker — uses fonts installed via:
        #   apt-get install fonts-dejavu-core fonts-dejavu-extra
        _dj = "/usr/share/fonts/truetype/dejavu"
        return {
            "headline": f"{_dj}/DejaVuSans-Bold.ttf",
            "body": f"{_dj}/DejaVuSans.ttf",
            "accent": f"{_dj}/DejaVuSerif-Italic.ttf",
            "display": f"{_dj}/DejaVuSans-Bold.ttf",
            "elegant": f"{_dj}/DejaVuSerif.ttf",
            "modern": f"{_dj}/DejaVuSans.ttf",
        }


def _get_indic_fonts():
    if _is_windows():
        return {
            "headline": ("C:/Windows/Fonts/Nirmala.ttc", 1),
            "body": ("C:/Windows/Fonts/Nirmala.ttc", 0),
            "accent": ("C:/Windows/Fonts/Nirmala.ttc", 0),
            "display": ("C:/Windows/Fonts/Nirmala.ttc", 1),
            "elegant": ("C:/Windows/Fonts/Nirmala.ttc", 0),
            "modern": ("C:/Windows/Fonts/Nirmala.ttc", 0),
        }
    else:
        _noto = "/usr/share/fonts/truetype/noto"
        _bold = f"{_noto}/NotoSansDevanagari-Bold.ttf"
        _regular = f"{_noto}/NotoSansDevanagari-Regular.ttf"
        # Fallback to DejaVu if Noto not available
        if not os.path.exists(_bold):
            _bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            _regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        return {
            "headline": _bold,
            "body": _regular,
            "accent": _regular,
            "display": _bold,
            "elegant": _regular,
            "modern": _regular,
        }


def _get_cjk_fonts():
    if _is_windows():
        return {
            "headline": ("C:/Windows/Fonts/msyhbd.ttc", 0),
            "body": ("C:/Windows/Fonts/msyh.ttc", 0),
            "accent": ("C:/Windows/Fonts/msyh.ttc", 0),
            "display": ("C:/Windows/Fonts/msyhbd.ttc", 0),
            "elegant": ("C:/Windows/Fonts/msyh.ttc", 0),
            "modern": ("C:/Windows/Fonts/msyh.ttc", 0),
        }
    else:
        _noto = "/usr/share/fonts/truetype/noto"
        _bold = f"{_noto}/NotoSansCJK-Bold.ttc"
        _regular = f"{_noto}/NotoSansCJK-Regular.ttc"
        if not os.path.exists(_bold):
            _bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            _regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        return {
            "headline": (_bold, 0),
            "body": (_regular, 0),
            "accent": (_regular, 0),
            "display": (_bold, 0),
            "elegant": (_regular, 0),
            "modern": (_regular, 0),
        }


def _get_arabic_fonts():
    if _is_windows():
        return {
            "headline": "C:/Windows/Fonts/segoeuib.ttf",
            "body": "C:/Windows/Fonts/segoeui.ttf",
            "accent": "C:/Windows/Fonts/segoeui.ttf",
            "display": "C:/Windows/Fonts/segoeuib.ttf",
            "elegant": "C:/Windows/Fonts/segoeui.ttf",
            "modern": "C:/Windows/Fonts/segoeui.ttf",
        }
    else:
        _noto = "/usr/share/fonts/truetype/noto"
        _bold = f"{_noto}/NotoSansArabic-Bold.ttf"
        _regular = f"{_noto}/NotoSansArabic-Regular.ttf"
        if not os.path.exists(_bold):
            _bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            _regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        return {
            "headline": _bold,
            "body": _regular,
            "accent": _regular,
            "display": _bold,
            "elegant": _regular,
            "modern": _regular,
        }


class ProAdDesigner:
    """Professional ad designer with multiple theme templates.
    Design decisions (theme, decorative elements) are driven by image analysis
    of retrieved reference ads from the dataset."""

    WIDTH = 1080
    HEIGHT = 1080

    FONTS = _get_latin_fonts()

    # Fonts for non-Latin scripts
    INDIC_FONTS = _get_indic_fonts()
    CJK_FONTS = _get_cjk_fonts()
    ARABIC_FONTS = _get_arabic_fonts()

    THEMES = [
        "minimal_clean", "bold_hero", "premium_dark",
        "split_layout", "card_float", "gradient_mesh",
        "realistic_pamphlet",
    ]

    def __init__(self):
        self._font_cache: Dict[str, ImageFont.FreeTypeFont] = {}
        self._script = "latin"  # "latin", "indic", "cjk", "arabic"
        self._theme_renderers = {
            "minimal_clean": self._render_minimal_clean,
            "bold_hero": self._render_bold_hero,
            "premium_dark": self._render_premium_dark,
            "split_layout": self._render_split_layout,
            "card_float": self._render_card_float,
            "gradient_mesh": self._render_gradient_mesh,
            "realistic_pamphlet": self._render_realistic_pamphlet,
        }

    # --- Shared utilities ---

    @staticmethod
    def _detect_script(text: str) -> str:
        """Detect the primary script of text: 'indic', 'cjk', 'arabic', or 'latin'."""
        if not text:
            return "latin"
        indic_count = 0
        cjk_count = 0
        arabic_count = 0
        total = 0
        for ch in text:
            cp = ord(ch)
            if cp < 0x20 or ch.isspace():
                continue
            total += 1
            if 0x0900 <= cp <= 0x0D7F:
                indic_count += 1
            elif (0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF
                  or 0xAC00 <= cp <= 0xD7AF or 0x3400 <= cp <= 0x4DBF):
                cjk_count += 1
            elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
                arabic_count += 1
        if total == 0:
            return "latin"
        if indic_count / total > 0.15:
            return "indic"
        if cjk_count / total > 0.15:
            return "cjk"
        if arabic_count / total > 0.15:
            return "arabic"
        return "latin"

    def _font(self, role: str, size: int) -> ImageFont.FreeTypeFont:
        if self._script == "indic":
            font_dict = self.INDIC_FONTS
        elif self._script == "cjk":
            font_dict = self.CJK_FONTS
        elif self._script == "arabic":
            font_dict = self.ARABIC_FONTS
        else:
            font_dict = self.FONTS

        entry = font_dict.get(role, self.FONTS.get(role, role))

        if isinstance(entry, tuple):
            path, ttc_index = entry
        else:
            path, ttc_index = entry, 0

        key = f"{path}_{ttc_index}_{size}"
        if key not in self._font_cache:
            try:
                self._font_cache[key] = ImageFont.truetype(path, size, index=ttc_index)
            except Exception:
                fallbacks = []
                if self._script == "indic":
                    fallbacks = [
                        ("C:/Windows/Fonts/Nirmala.ttc", 1),
                        ("C:/Windows/Fonts/Nirmala.ttc", 0),
                    ]
                elif self._script == "cjk":
                    fallbacks = [
                        ("C:/Windows/Fonts/msyh.ttc", 0),
                        ("C:/Windows/Fonts/simsun.ttc", 0),
                    ]
                fallbacks.append(("arial.ttf", 0))
                loaded = False
                for fb_path, fb_idx in fallbacks:
                    try:
                        self._font_cache[key] = ImageFont.truetype(fb_path, size, index=fb_idx)
                        loaded = True
                        break
                    except Exception:
                        continue
                if not loaded:
                    self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def _text_size(self, draw, text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    @staticmethod
    def _hex_to_rgb(h: str):
        h = h.lstrip("#")
        if len(h) < 6:
            h = h.ljust(6, "0")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _round_corners(self, img, radius):
        mask = Image.new("L", img.size, 0)
        d = ImageDraw.Draw(mask)
        d.rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius, fill=255)
        result = img.convert("RGBA")
        result.putalpha(mask)
        return result

    def _word_wrap(self, draw, text, font, max_width):
        words = text.split()
        lines, current = [], ""
        for word in words:
            test = f"{current} {word}".strip()
            tw, _ = self._text_size(draw, test, font)
            if tw <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [text]

    def _draw_text_shadow(self, draw, xy, text, font, fill=(255, 255, 255, 255),
                          shadow_color=(0, 0, 0, 120), offset=3):
        x, y = xy
        for dx, dy, a_mult in [(offset, offset, 0.3), (offset // 2, offset // 2, 0.6)]:
            sc = (*shadow_color[:3], int(shadow_color[3] * a_mult))
            draw.text((x + dx, y + dy), text, fill=sc, font=font)
        draw.text((x, y), text, fill=fill, font=font)

    def _draw_text_outline(self, draw, xy, text, font, fill=(255, 255, 255, 255),
                           outline_color=(0, 0, 0, 200), width=2):
        x, y = xy
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                if dx * dx + dy * dy <= width * width:
                    draw.text((x + dx, y + dy), text, fill=outline_color, font=font)
        draw.text((x, y), text, fill=fill, font=font)

    def _draw_spaced_text(self, canvas, xy, text, font, fill, spacing=4,
                          center_x=None, y_pos=None):
        draw = ImageDraw.Draw(canvas)
        if center_x is not None:
            total_w = sum(self._text_size(draw, c, font)[0] + spacing for c in text) - spacing
            x = center_x - total_w // 2
            y = y_pos if y_pos is not None else (xy[1] if xy else 0)
        else:
            x, y = xy
        for char in text:
            draw.text((x, y), char, fill=fill, font=font)
            cw, _ = self._text_size(draw, char, font)
            x += cw + spacing
        return canvas

    def _draw_secondary_text(self, canvas, content, y, margin, max_width,
                                light_text=True, centered=False):
        """Draw the secondary language headline + features below the primary if bilingual."""
        if not content.headline_secondary:
            return y
        draw = ImageDraw.Draw(canvas)
        saved_script = self._script
        self._script = "latin"

        sec_font = self._font("body", 18)
        color = (200, 200, 210, 180) if light_text else (80, 80, 90, 180)
        lines = self._word_wrap(draw, content.headline_secondary, sec_font, max_width)
        for line in lines[:2]:
            if centered:
                tw, _ = self._text_size(draw, line, sec_font)
                draw.text(((canvas.width - tw) // 2, y), line, fill=color, font=sec_font)
            else:
                draw.text((margin, y), line, fill=color, font=sec_font)
            y += 24
        y += 6

        if content.features_secondary:
            sec_feat_font = self._font("body", 14)
            feat_color = (170, 170, 180, 150) if light_text else (110, 110, 120, 150)
            for feat in content.features_secondary[:3]:
                text = f"  {feat}"
                if centered:
                    tw, _ = self._text_size(draw, text, sec_feat_font)
                    draw.text(((canvas.width - tw) // 2, y), text, fill=feat_color, font=sec_feat_font)
                else:
                    draw.text((margin, y), text, fill=feat_color, font=sec_feat_font)
                y += 20
            y += 6

        self._script = saved_script
        return y

    def _place_logo(self, canvas, logo, position="top-left", max_height=70, padding=30, backing=True):
        if logo is None:
            return canvas
        logo = logo.copy().convert("RGBA")
        lw, lh = logo.size
        if lh > 0:
            scale = max_height / lh
            logo_w = min(int(lw * scale), max_height * 3)
            logo = logo.resize((logo_w, max_height), Image.LANCZOS)
        else:
            return canvas

        W, H = canvas.size
        positions = {
            "top-left": (padding, padding),
            "top-right": (W - logo.width - padding, padding),
            "top-center": ((W - logo.width) // 2, padding),
            "bottom-left": (padding, H - max_height - padding),
            "bottom-right": (W - logo.width - padding, H - max_height - padding),
        }
        x, y = positions.get(position, positions["top-left"])

        if backing:
            back_pad = 10
            bw, bh = logo.width + 2 * back_pad, max_height + 2 * back_pad
            bx, by = x - back_pad, y - back_pad
            if 0 <= bx and bx + bw <= W and 0 <= by and by + bh <= H:
                region = canvas.crop((bx, by, bx + bw, by + bh)).convert("RGBA")
                blurred = region.filter(ImageFilter.GaussianBlur(12))
                frost = Image.new("RGBA", (bw, bh), (255, 255, 255, 35))
                backing_img = Image.alpha_composite(blurred, frost)
                mask = Image.new("L", (bw, bh), 0)
                ImageDraw.Draw(mask).rounded_rectangle([0, 0, bw - 1, bh - 1], radius=10, fill=255)
                backing_img.putalpha(mask)
                canvas.paste(Image.alpha_composite(
                    canvas.crop((bx, by, bx + bw, by + bh)).convert("RGBA"),
                    backing_img), (bx, by))

        shadow_offset = 2
        if logo.width > 10 and logo.height > 10:
            shadow = Image.new("RGBA", logo.size, (0, 0, 0, 0))
            if logo.mode == "RGBA":
                alpha = logo.split()[3]
                shadow_alpha = alpha.point(lambda p: min(p, 40))
                shadow.putalpha(shadow_alpha)
                shadow = shadow.filter(ImageFilter.GaussianBlur(2))
                sx, sy = x + shadow_offset, y + shadow_offset
                if sx + shadow.width <= W and sy + shadow.height <= H:
                    rg = canvas.crop((sx, sy, sx + shadow.width, sy + shadow.height)).convert("RGBA")
                    canvas.paste(Image.alpha_composite(rg, shadow), (sx, sy))

        canvas.paste(logo, (x, y), logo)
        return canvas

    def _draw_cta_button(self, canvas, text, position, color, style="rounded", font_size=22):
        font = self._font("headline", font_size)
        draw = ImageDraw.Draw(canvas)
        tw, th = self._text_size(draw, text, font)
        pad_x, pad_y = 32, 14
        btn_w = max(tw + 2 * pad_x, 180)
        btn_h = th + 2 * pad_y

        scale = 2
        btn = Image.new("RGBA", (btn_w * scale, btn_h * scale), (0, 0, 0, 0))
        bd = ImageDraw.Draw(btn)

        if style == "outline":
            bd.rounded_rectangle([0, 0, btn_w * scale - 1, btn_h * scale - 1],
                                 radius=btn_h * scale // 2, outline=(*color, 240), width=3 * scale)
        else:
            bd.rounded_rectangle([0, 0, btn_w * scale - 1, btn_h * scale - 1],
                                 radius=btn_h * scale // 2, fill=(*color, 240))

        lum = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
        if style == "outline":
            txt_col = (*color, 255)
        else:
            txt_col = (20, 20, 20, 255) if lum > 140 else (255, 255, 255, 255)

        big_font = self._font("headline", font_size * scale)
        btw, bth = self._text_size(bd, text, big_font)
        bd.text(((btn_w * scale - btw) // 2, (btn_h * scale - bth) // 2 - 2),
                text, fill=txt_col, font=big_font)

        btn = btn.resize((btn_w, btn_h), Image.LANCZOS)

        x, y = position
        shadow = Image.new("RGBA", (btn_w + 10, btn_h + 10), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.rounded_rectangle([5, 5, btn_w + 4, btn_h + 4], radius=btn_h // 2, fill=(0, 0, 0, 35))
        shadow = shadow.filter(ImageFilter.GaussianBlur(4))
        sx, sy = max(0, x - 3), max(0, y - 3)
        sw, sh = min(shadow.width, canvas.width - sx), min(shadow.height, canvas.height - sy)
        if sw > 0 and sh > 0:
            sr = canvas.crop((sx, sy, sx + sw, sy + sh)).convert("RGBA")
            sc = shadow.crop((0, 0, sw, sh))
            canvas.paste(Image.alpha_composite(sr, sc), (sx, sy))

        if x + btn_w <= canvas.width and y + btn_h <= canvas.height:
            canvas.paste(btn, (x, y), btn)
        return canvas, btn_h

    def _draw_feature_list(self, canvas, features, start_xy, max_width,
                           style="icons", color=(255, 255, 255, 240),
                           accent=(255, 200, 50), font_size=18):
        draw = ImageDraw.Draw(canvas)
        font = self._font("body", font_size)
        x, y = start_xy
        icons = ["\u2713", "\u2605", "\u25CF", "\u25B6"]

        for i, feat in enumerate(features[:5]):
            if style == "numbered":
                num_font = self._font("headline", font_size + 4)
                num = f"{i + 1:02d}"
                draw.text((x, y - 2), num, fill=(*accent[:3], 160), font=num_font)
                draw.text((x + 38, y + 2), feat, fill=color, font=font)
                y += font_size + 18
            elif style == "dividers":
                draw.text((x, y), feat, fill=color, font=font)
                ftw, fth = self._text_size(draw, feat, font)
                y += fth + 6
                draw.line([(x, y), (x + min(ftw, int(max_width * 0.6)), y)],
                          fill=(*accent[:3], 50), width=1)
                y += 10
            else:  # icons
                icon = icons[i % len(icons)]
                icon_font = self._font("body", font_size + 2)
                draw.text((x, y), icon, fill=(*accent[:3], 200), font=icon_font)
                draw.text((x + 26, y), feat, fill=color, font=font)
                y += font_size + 14
        return y

    def _get_cta_text(self, content):
        """Return AI-generated CTA text. Falls back to brand name if AI produced nothing."""
        if content.cta_text:
            return content.cta_text
        # Minimal fallback — just the brand name, not a hardcoded marketing phrase
        brand = content.brand_name.replace("_", " ").upper()
        return brand if brand else ""

    def _cover_fill(self, img, W, H):
        """Scale image to cover entire canvas (crop overflow)."""
        img = img.copy().convert("RGB")
        pw, ph = img.size
        scale = max(W / pw, H / ph)
        img = img.resize((int(pw * scale), int(ph * scale)), Image.LANCZOS)
        lx = (img.width - W) // 2
        ly = (img.height - H) // 2
        return img.crop((lx, ly, lx + W, ly + H))

    def _generate_subtle_noise(self, W, H, intensity=8):
        noise = np.random.randint(0, intensity, (H, W), dtype=np.uint8)
        noise_rgba = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        noise_rgba.putalpha(Image.fromarray(noise, "L"))
        return noise_rgba

    # --- Decorative element methods ---

    def _draw_wave_divider(self, canvas, y_pos, color, amplitude=40, thickness=3, fill_below=None):
        """Draw a smooth curved wave separator line across the canvas."""
        W, H = canvas.size
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        points = []
        for x in range(0, W + 1, 2):
            t = x / W
            y = y_pos + int(amplitude * math.sin(t * math.pi * 2.0))
            points.append((x, y))
        if fill_below:
            fill_points = list(points) + [(W, H), (0, H)]
            draw.polygon(fill_points, fill=fill_below)
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=color, width=thickness)
        canvas = Image.alpha_composite(canvas, layer)
        return canvas

    def _draw_badge(self, canvas, position, text, bg_color, text_color=(255, 255, 255, 255),
                    size=80, style="circle"):
        """Draw a circular or ribbon badge with text."""
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        x, y = position
        if style == "circle":
            draw.ellipse([x, y, x + size, y + size], fill=bg_color,
                         outline=(255, 255, 255, 180), width=2)
            font = self._font("headline", max(10, size // 5))
            lines = text.upper().split('\n') if '\n' in text else [text.upper()]
            ty = y + size // 2 - len(lines) * (size // 5 + 2) // 2
            for line in lines[:3]:
                tw, _ = self._text_size(draw, line, font)
                draw.text(((2 * x + size - tw) // 2, ty), line, fill=text_color, font=font)
                ty += size // 5 + 4
        elif style == "ribbon":
            rw, rh = size * 2, size // 2
            pts = [(x, y), (x + rw, y), (x + rw - 10, y + rh // 2),
                   (x + rw, y + rh), (x, y + rh), (x + 10, y + rh // 2)]
            draw.polygon(pts, fill=bg_color)
            font = self._font("headline", max(10, rh // 2 - 2))
            tw, _ = self._text_size(draw, text.upper(), font)
            draw.text(((2 * x + rw - tw) // 2, y + rh // 4), text.upper(),
                      fill=text_color, font=font)
        canvas = Image.alpha_composite(canvas, layer)
        return canvas

    def _draw_glow_circle(self, canvas, center, radius, color, intensity=30):
        """Draw a soft glowing circle (radial gradient) at given position."""
        W, H = canvas.size
        cx, cy = center
        layer_arr = np.zeros((H, W, 4), dtype=np.uint8)
        Y_g, X_g = np.ogrid[:H, :W]
        dist = np.sqrt((X_g - cx) ** 2 + (Y_g - cy) ** 2)
        falloff = np.clip(1 - dist / radius, 0, 1) ** 2
        for c in range(3):
            layer_arr[:, :, c] = np.clip(color[c] * falloff, 0, 255).astype(np.uint8)
        layer_arr[:, :, 3] = np.clip(intensity * falloff, 0, 255).astype(np.uint8)
        layer = Image.fromarray(layer_arr, "RGBA")
        canvas = Image.alpha_composite(canvas, layer)
        return canvas

    def _draw_corner_accents(self, canvas, color, size=80, thickness=3):
        """Draw elegant corner bracket accents."""
        W, H = canvas.size
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        m = 25
        c = (*color[:3], min(color[3] if len(color) > 3 else 255, 120))
        for cx, cy, dx, dy in [(m, m, 1, 1), (W - m, m, -1, 1),
                                (m, H - m, 1, -1), (W - m, H - m, -1, -1)]:
            draw.line([(cx, cy), (cx + size * dx, cy)], fill=c, width=thickness)
            draw.line([(cx, cy), (cx, cy + size * dy)], fill=c, width=thickness)
        canvas = Image.alpha_composite(canvas, layer)
        return canvas

    def _apply_post_processing(self, canvas):
        rgb = canvas.convert("RGB")
        w, h = rgb.size
        up = rgb.resize((w * 2, h * 2), Image.LANCZOS)
        up = up.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=3))
        rgb = up.resize((w, h), Image.LANCZOS)
        rgb = ImageEnhance.Sharpness(rgb).enhance(1.15)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.1)
        rgb = ImageEnhance.Color(rgb).enhance(1.05)
        return rgb

    def _real_ad_background(self, W, H, ad_images, accent, style="dark"):
        """Use a REAL ad from the dataset as blurred background for authentic look."""
        canvas = Image.new("RGBA", (W, H), (30, 30, 40, 255))
        if not ad_images:
            return canvas
        bg = self._cover_fill(ad_images[0], W, H).convert("RGBA")
        if style == "dark":
            bg = bg.filter(ImageFilter.GaussianBlur(18))
            dark = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            dd = ImageDraw.Draw(dark)
            for row in range(H):
                t = row / H
                a = int(90 + 130 * (t ** 1.2))
                tint = tuple(int(accent[i] * 0.08 * t) for i in range(3))
                dd.line([(0, row), (W, row)], fill=(*tint, min(a, 225)))
            bg = Image.alpha_composite(bg, dark)
        elif style == "light":
            bg = bg.filter(ImageFilter.GaussianBlur(25))
            light = Image.new("RGBA", (W, H), (255, 255, 255, 180))
            bg = Image.alpha_composite(bg, light)
        elif style == "tinted":
            bg = bg.filter(ImageFilter.GaussianBlur(20))
            tint = Image.new("RGBA", (W, H), (*accent, 100))
            bg = Image.alpha_composite(bg, tint)
            dark = Image.new("RGBA", (W, H), (0, 0, 0, 80))
            bg = Image.alpha_composite(bg, dark)
        return bg

    def _draw_ref_thumbnails(self, canvas, thumbs, position="bottom-right",
                              thumb_size=100, style="tilted"):
        """Draw reference ad thumbnails from the dataset on the ad."""
        if not thumbs:
            return canvas
        W, H = canvas.size
        thumbs = thumbs[:3]

        if style == "tilted":
            angles = [-6, 3, -3]
            if position == "bottom-right":
                base_x = W - 40 - int(thumb_size * 1.7)
                base_y = H - 40 - thumb_size
            else:
                base_x = 25
                base_y = H - 40 - thumb_size
            for i, thumb in enumerate(thumbs):
                t = thumb.copy().convert("RGB")
                tw, th = t.size
                sc = min(thumb_size / tw, thumb_size / th)
                t = t.resize((int(tw * sc), int(th * sc)), Image.LANCZOS)
                bdr = 4
                framed = Image.new("RGBA", (t.width + 2 * bdr, t.height + 2 * bdr), (255, 255, 255, 200))
                framed.paste(t.convert("RGBA"), (bdr, bdr))
                framed = self._round_corners(framed, 8)
                shadow = Image.new("RGBA", (framed.width + 6, framed.height + 6), (0, 0, 0, 0))
                sd = ImageDraw.Draw(shadow)
                sd.rounded_rectangle([3, 3, shadow.width - 1, shadow.height - 1], 10, fill=(0, 0, 0, 40))
                shadow = shadow.filter(ImageFilter.GaussianBlur(3))
                shadow.paste(framed, (0, 0), framed)
                framed = shadow
                rotated = framed.rotate(angles[i % 3], expand=True, resample=Image.BICUBIC)
                tx = int(base_x + i * (thumb_size * 0.48))
                ty = int(base_y - i * 8)
                pw = min(rotated.width, W - tx)
                ph = min(rotated.height, H - ty)
                if pw > 0 and ph > 0 and 0 <= tx and 0 <= ty:
                    cr = rotated.crop((0, 0, pw, ph))
                    rg = canvas.crop((tx, ty, tx + pw, ty + ph)).convert("RGBA")
                    canvas.paste(Image.alpha_composite(rg, cr), (tx, ty))
        elif style == "strip":
            gap = 8
            total_w = len(thumbs) * (thumb_size + gap) - gap
            if position == "bottom-right":
                sx = W - total_w - 30
            else:
                sx = 30
            sy = H - int(thumb_size * 0.7) - 25
            for i, thumb in enumerate(thumbs):
                t = thumb.copy().convert("RGB")
                tw, th = t.size
                sc = min(thumb_size / tw, (thumb_size * 0.7) / th)
                t = t.resize((int(tw * sc), int(th * sc)), Image.LANCZOS)
                bdr = 3
                framed = Image.new("RGBA", (t.width + 2 * bdr, t.height + 2 * bdr), (255, 255, 255, 180))
                framed.paste(t.convert("RGBA"), (bdr, bdr))
                framed = self._round_corners(framed, 6)
                tx = sx + i * (thumb_size + gap)
                ty = sy
                if 0 <= tx and tx + framed.width <= W and 0 <= ty and ty + framed.height <= H:
                    rg = canvas.crop((tx, ty, tx + framed.width, ty + framed.height)).convert("RGBA")
                    canvas.paste(Image.alpha_composite(rg, framed), (tx, ty))
        return canvas

    def _draw_brand_url_bar(self, canvas, brand_name, domain, color_pair, position="bottom", side="left"):
        """Draw a professional brand info bar with gradient."""
        W, H = canvas.size
        bar_h = 55
        bar_w = int(W * 0.45)
        bar_y = H - bar_h - 10
        c1, c2 = color_pair

        bar = Image.new("RGBA", (bar_w, bar_h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bar)
        for col in range(bar_w):
            t = col / bar_w
            rgb = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
            alpha = int(210 * (1 - t * 0.3))
            bd.line([(col, 0), (col, bar_h)], fill=(*rgb, alpha))
        for col in range(bar_w - 30, bar_w):
            t = (col - (bar_w - 30)) / 30
            for row in range(bar_h):
                px = bar.getpixel((col, row))
                bar.putpixel((col, row), (px[0], px[1], px[2], max(0, int(px[3] * (1 - t)))))

        bar_x = 0 if side == "left" else (W - bar_w)
        if side == "right":
            bar = bar.transpose(Image.FLIP_LEFT_RIGHT)
        canvas.paste(bar, (bar_x, bar_y), bar)

        draw = ImageDraw.Draw(canvas)
        info_font = self._font("body", 12)
        url_font = self._font("headline", 18)
        brand_url = domain or f"www.{brand_name.lower().replace('_', '').replace(' ', '')}.com"
        margin = 20 if side == "left" else (W - bar_w + 20)
        draw.text((margin, bar_y + 6), "VISIT US", fill=(255, 255, 255, 220), font=info_font)
        draw.text((margin, bar_y + 24), brand_url, fill=(255, 255, 80, 255), font=url_font)
        return canvas

    # --- Image Analysis for dynamic theme selection ---

    def _analyze_image(self, img):
        """Divide image into 3x3 grid, compute complexity per cell.
        Returns which side is busy, where free space is, avg brightness."""
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

        if left_complexity > right_complexity * 1.15:
            text_side = "right"
        elif right_complexity > left_complexity * 1.15:
            text_side = "left"
        else:
            text_side = "left"

        flat_idx = int(np.argmin(complexity))
        free_row, free_col = flat_idx // 3, flat_idx % 3

        avg_brightness = brightness.mean()
        is_dark = avg_brightness < 120

        return {
            "text_side": text_side,
            "free_row": free_row,
            "free_col": free_col,
            "avg_brightness": avg_brightness,
            "is_dark": is_dark,
            "complexity_grid": complexity,
        }

    def _analyze_reference_ads(self, ref_images: List[Image.Image]) -> dict:
        """Analyze reference ad images from the dataset to determine design characteristics.
        This drives theme selection based on what the actual ads look like."""
        if not ref_images:
            return {"avg_brightness": 128, "is_dark": False, "avg_saturation": 0.3,
                    "avg_complexity": 30, "dominant_tone": "neutral"}

        total_brightness = 0
        total_saturation = 0
        total_complexity = 0

        for img in ref_images[:5]:
            arr = np.array(img.convert("RGB")).astype(float)
            total_brightness += arr.mean() / 255.0

            # Saturation from max-min channel range
            max_c = arr.max(axis=2).mean()
            min_c = arr.min(axis=2).mean()
            total_saturation += (max_c - min_c) / max(max_c, 1)

            # Edge complexity
            dx = np.abs(np.diff(arr, axis=1)).mean()
            dy = np.abs(np.diff(arr, axis=0)).mean()
            total_complexity += dx + dy

        n = min(len(ref_images), 5)
        avg_brightness = total_brightness / n
        avg_saturation = total_saturation / n
        avg_complexity = total_complexity / n

        # Determine dominant tone
        if avg_brightness < 0.35:
            dominant_tone = "dark"
        elif avg_brightness > 0.65:
            dominant_tone = "light"
        else:
            dominant_tone = "neutral"

        return {
            "avg_brightness": avg_brightness,
            "is_dark": avg_brightness < 0.4,
            "avg_saturation": avg_saturation,
            "avg_complexity": avg_complexity,
            "dominant_tone": dominant_tone,
        }

    # --- Theme selection driven by reference image analysis ---

    def _select_theme(self, content) -> str:
        """Select theme based on analysis of the product image and reference ads.
        No hardcoded category-to-theme mapping — the visual characteristics of the
        retrieved reference images drive the choice."""
        if content.theme_name and content.theme_name in self.THEMES:
            return content.theme_name

        # Analyze reference ad thumbnails from the dataset
        ref_analysis = self._analyze_reference_ads(content.thumbnail_images)

        # Build weighted pool based on what the reference images look like
        pool = []

        if ref_analysis["is_dark"]:
            # Dark reference ads → prefer dark/bold themes
            pool.extend(["premium_dark"] * 3)
            pool.extend(["bold_hero"] * 2)
            pool.extend(["gradient_mesh"] * 2)
        elif ref_analysis["dominant_tone"] == "light":
            # Light reference ads → prefer clean/minimal themes
            pool.extend(["minimal_clean"] * 3)
            pool.extend(["card_float"] * 2)
            pool.extend(["split_layout"] * 2)
        else:
            # Neutral → balanced mix
            pool.extend(["minimal_clean"] * 2)
            pool.extend(["bold_hero"] * 2)
            pool.extend(["split_layout"] * 2)

        if ref_analysis["avg_saturation"] > 0.4:
            # Highly saturated reference images → vibrant themes
            pool.extend(["gradient_mesh"] * 2)
            pool.extend(["bold_hero"] * 1)

        if ref_analysis["avg_complexity"] > 40:
            # Complex reference images → simpler layouts
            pool.extend(["minimal_clean"] * 2)
            pool.extend(["card_float"] * 1)
        else:
            # Simple reference images → can use richer layouts
            pool.extend(["split_layout"] * 1)
            pool.extend(["premium_dark"] * 1)

        # Ensure all themes have at least some chance
        for t in self.THEMES:
            if t not in pool:
                pool.append(t)

        theme = random.choice(pool)
        print(f"  Theme: {theme} (ref tone: {ref_analysis['dominant_tone']}, sat: {ref_analysis['avg_saturation']:.2f})")
        return theme

    def get_layout_hint(self, content) -> str:
        """Get layout hint for image generation based on theme."""
        return "center"

    # --- Main compose entry point ---

    def compose(self, content) -> Image.Image:
        print(f"    [DESIGNER] Composing ad pamphlet...")
        print(f"    [DESIGNER] Canvas: {self.WIDTH}x{self.HEIGHT}")
        print(f"    [DESIGNER] Brand: {content.brand_name}")
        print(f"    [DESIGNER] Headline: \"{content.headline[:50]}{'...' if len(content.headline) > 50 else ''}\"")
        print(f"    [DESIGNER] Tagline: \"{content.tagline}\"")
        print(f"    [DESIGNER] Features: {len(content.features)}")
        print(f"    [DESIGNER] CTA: \"{content.cta_text}\"")
        print(f"    [DESIGNER] Product image: {'YES' if content.product_image else 'NO'}")
        print(f"    [DESIGNER] Logo: {'YES' if content.logo_image else 'NO'}")
        print(f"    [DESIGNER] Thumbnails: {len(content.thumbnail_images)}")
        print(f"    [DESIGNER] Colors: accent={content.accent_color}, secondary={content.secondary_color}")
        print(f"    [DESIGNER] Bilingual: {'YES' if content.headline_secondary else 'NO'}")

        # Detect script from headline/features/CTA for correct font selection
        sample_text = " ".join([
            content.headline or "",
            content.tagline or "",
            " ".join(content.features[:3]),
            content.cta_text or "",
        ])
        self._script = self._detect_script(sample_text)
        print(f"    [DESIGNER] Detected script: {self._script}")
        font_system = {"latin": "Segoe UI / Impact / Calibri", "indic": "Nirmala UI", "cjk": "Microsoft YaHei", "arabic": "Segoe UI"}
        print(f"    [DESIGNER] Font system: {font_system.get(self._script, 'default')}")

        # If no product image, use branded typography-focused ad
        if content.product_image is None:
            print(f"    [DESIGNER] No product image -> Typography-focused ad layout")
            self._analysis = {"text_side": "left", "free_row": 0, "free_col": 2,
                              "avg_brightness": 60, "is_dark": True,
                              "complexity_grid": np.zeros((3, 3))}
            canvas = self._render_no_image_ad(content)
            print(f"    [DESIGNER] Rendered: no-image typography ad")
            return self._apply_post_processing(canvas)

        # AI analysis of product image for dynamic layout
        print(f"    [DESIGNER] Analyzing product image for dynamic layout...")
        self._analysis = self._analyze_image(content.product_image)
        print(f"    [DESIGNER] Analysis: text_side={self._analysis.get('text_side')}, is_dark={self._analysis.get('is_dark')}, avg_brightness={self._analysis.get('avg_brightness', 0):.0f}")

        forced_theme = (getattr(content, "theme_name", "") or "").strip()
        if forced_theme and forced_theme in self._theme_renderers:
            theme = forced_theme
            print(f"    [DESIGNER] Forced theme from content: {theme}")
        else:
            theme = self._select_theme(content)
        print(f"    [DESIGNER] Selected theme: {theme}")
        print(f"    [DESIGNER] Available themes: {self.THEMES}")
        renderer = self._theme_renderers[theme]
        canvas = renderer(content)
        print(f"    [DESIGNER] Rendered: {theme} theme")
        return self._apply_post_processing(canvas)

    # === NO-IMAGE AD: Professional branded typography ad ===
    def _render_no_image_ad(self, content) -> Image.Image:
        """When no product image is available, create a visually striking
        typography-focused ad that uses the ENTIRE canvas — no empty space."""
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)
        secondary = self._hex_to_rgb(content.secondary_color)
        dark_accent = tuple(max(0, c - 60) for c in accent)
        bright_accent = tuple(min(255, c + 60) for c in accent)

        canvas = Image.new("RGBA", (W, H), (18, 18, 28, 255))
        draw = ImageDraw.Draw(canvas)
        for row in range(H):
            t = row / H
            r = int(dark_accent[0] * (1 - t) + accent[0] * 0.2 * t)
            g = int(dark_accent[1] * (1 - t) + accent[1] * 0.2 * t)
            b = int(dark_accent[2] * (1 - t) + accent[2] * 0.2 * t)
            draw.line([(0, row), (W, row)], fill=(max(0, min(255, r)),
                       max(0, min(255, g)), max(0, min(255, b)), 255))

        # Decorative circle
        circle_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        cd = ImageDraw.Draw(circle_layer)
        cr = int(W * 0.42)
        cx, cy = int(W * 0.78), int(H * 0.22)
        cd.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                   outline=(*accent, 35), width=2)
        cd.ellipse([cx - cr + 30, cy - cr + 30, cx + cr - 30, cy + cr - 30],
                   fill=(*accent, 18))
        inner_r = int(cr * 0.55)
        cd.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
                   outline=(*bright_accent, 25), width=1)
        canvas = Image.alpha_composite(canvas, circle_layer)

        # Diagonal accent band
        band_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bd = ImageDraw.Draw(band_layer)
        band_pts = [(W, int(H * 0.55)), (W, int(H * 0.65)),
                    (int(W * 0.35), H), (int(W * 0.25), H)]
        bd.polygon(band_pts, fill=(*accent, 22))
        band_pts2 = [(W, int(H * 0.68)), (W, int(H * 0.72)),
                     (int(W * 0.50), H), (int(W * 0.45), H)]
        bd.polygon(band_pts2, fill=(*secondary, 18))
        canvas = Image.alpha_composite(canvas, band_layer)

        # Dot pattern
        dot_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dd = ImageDraw.Draw(dot_layer)
        dot_spacing = 30
        for dx in range(45, int(W * 0.3), dot_spacing):
            for dy in range(45, int(H * 0.25), dot_spacing):
                dd.ellipse([dx - 1, dy - 1, dx + 1, dy + 1], fill=(*accent, 30))
        canvas = Image.alpha_composite(canvas, dot_layer)

        noise = self._generate_subtle_noise(W, H, intensity=5)
        canvas = Image.alpha_composite(canvas, noise)
        draw = ImageDraw.Draw(canvas)

        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=60, padding=55, backing=False)
        draw = ImageDraw.Draw(canvas)

        margin_l = 65
        margin_r = 65
        text_w = W - margin_l - margin_r

        brand_display = content.brand_name.replace("_", " ").upper()
        brand_font = self._font("body", 14)
        self._draw_spaced_text(canvas, (margin_l, int(H * 0.20)), brand_display, brand_font,
                               fill=(*accent, 220), spacing=6)
        draw = ImageDraw.Draw(canvas)

        y = int(H * 0.24)
        draw.rounded_rectangle([margin_l, y, margin_l + 55, y + 4], radius=2,
                               fill=(*accent, 220))
        y += 28

        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("headline", 54)
        lines = self._word_wrap(draw, headline, head_font, text_w)
        for line in lines[:3]:
            self._draw_text_shadow(draw, (margin_l, y), line, head_font,
                                   fill=(255, 255, 255, 255),
                                   shadow_color=(0, 0, 0, 80), offset=2)
            y += 64
        y += 12

        if content.tagline and content.tagline != content.headline:
            tag_font = self._font("accent", 21)
            draw.text((margin_l, y), content.tagline, fill=(*bright_accent, 220), font=tag_font)
            y += 34

        feats = content.features[:5]
        if feats:
            y += 16
            feat_font = self._font("body", 18)
            for feat in feats:
                draw.ellipse([margin_l, y + 7, margin_l + 8, y + 15],
                             fill=(*accent, 220))
                draw.text((margin_l + 22, y), feat,
                          fill=(225, 225, 235, 240), font=feat_font)
                y += 30

        y += 8
        y = self._draw_secondary_text(canvas, content, y, margin_l, text_w,
                                      light_text=True, centered=False)
        draw = ImageDraw.Draw(canvas)

        cta = self._get_cta_text(content)
        cta_font = self._font("headline", 22)
        ctw, _ = self._text_size(draw, cta, cta_font)
        btn_w = max(ctw + 70, 220)
        cta_y = max(y + 20, H - 160)
        cta_x = (W - btn_w) // 2
        canvas, btn_h = self._draw_cta_button(canvas, cta,
                                              (cta_x, cta_y), accent,
                                              style="rounded", font_size=22)

        draw = ImageDraw.Draw(canvas)
        domain = content.brand_domain or f"www.{content.brand_name.lower().replace('_', '').replace(' ', '')}.com"
        url_font = self._font("body", 13)
        uw, _ = self._text_size(draw, domain, url_font)
        draw.text(((W - uw) // 2, H - 45), domain, fill=(*accent, 150), font=url_font)

        draw = ImageDraw.Draw(canvas)
        for bx in range(W):
            t = bx / W
            c = tuple(int(accent[i] + (secondary[i] - accent[i]) * t) for i in range(3))
            draw.line([(bx, H - 5), (bx, H)], fill=(*c, 220))

        return canvas

    # === THEME 1: Clean Wave ===
    def _render_minimal_clean(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)
        darker = tuple(max(0, c - 40) for c in accent)

        canvas = Image.new("RGBA", (W, H), (*darker, 255))

        prod_h = int(H * 0.65)
        if content.product_image:
            prod = self._cover_fill(content.product_image, W, prod_h)
            canvas.paste(prod.convert("RGBA"), (0, 0))

        wave_y = prod_h - 20
        canvas = self._draw_wave_divider(canvas, wave_y, (255, 255, 255, 200),
                                          amplitude=25, thickness=3,
                                          fill_below=(*darker, 252))

        draw = ImageDraw.Draw(canvas)
        margin = 60
        text_w = W - 2 * margin

        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=55, padding=30, backing=False)
        draw = ImageDraw.Draw(canvas)

        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("headline", 36)
        lines = self._word_wrap(draw, headline, head_font, text_w)
        y = wave_y + 50
        for line in lines[:2]:
            tw, _ = self._text_size(draw, line, head_font)
            draw.text(((W - tw) // 2, y), line, fill=(255, 255, 255, 255), font=head_font)
            y += 44

        y += 12
        feat_font = self._font("body", 14)
        feat_text = "  \u2022  ".join(content.features[:3])
        ftw, _ = self._text_size(draw, feat_text, feat_font)
        draw.text(((W - ftw) // 2, y), feat_text, fill=(200, 200, 210, 200), font=feat_font)
        y += 28

        cta = self._get_cta_text(content)
        cta_font = self._font("headline", 18)
        ctw, _ = self._text_size(draw, cta, cta_font)
        btn_w = max(ctw + 50, 180)
        canvas, _ = self._draw_cta_button(canvas, cta,
                                          ((W - btn_w) // 2, min(y + 8, H - 65)),
                                          accent, style="rounded", font_size=18)

        return canvas

    # === THEME 2: Bold Hero ===
    def _render_bold_hero(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)

        canvas = Image.new("RGBA", (W, H), (15, 15, 22, 255))

        prod_h = int(H * 0.65)
        if content.product_image:
            prod = self._cover_fill(content.product_image, W, prod_h)
            canvas.paste(prod.convert("RGBA"), (0, 0))

        wave_y = prod_h - 20
        canvas = self._draw_wave_divider(canvas, wave_y, (*accent, 220),
                                          amplitude=30, thickness=3,
                                          fill_below=(15, 15, 22, 252))

        draw = ImageDraw.Draw(canvas)
        margin = 60

        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=50, padding=30, backing=False)
        draw = ImageDraw.Draw(canvas)

        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("display", 40)
        lines = self._word_wrap(draw, headline.upper(), head_font, W - 2 * margin)
        y = wave_y + 48
        for line in lines[:2]:
            draw.text((margin, y), line, fill=(255, 255, 255, 255), font=head_font)
            y += 48

        y += 10
        feat_font = self._font("body", 14)
        feat_text = "  \u2022  ".join(content.features[:3])
        draw.text((margin, y), feat_text, fill=(180, 180, 190, 200), font=feat_font)
        y += 28

        canvas, _ = self._draw_cta_button(canvas, self._get_cta_text(content),
                                          (margin, min(y + 6, H - 65)), accent,
                                          style="rounded", font_size=18)

        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, H - 4, W, H], fill=(*accent, 220))

        return canvas

    # === THEME 3: Premium Dark ===
    def _render_premium_dark(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)
        if sum(accent) < 200:
            accent = (212, 175, 55)

        canvas = Image.new("RGBA", (W, H), (12, 12, 18, 255))

        prod_h = int(H * 0.60)
        if content.product_image:
            prod = self._cover_fill(content.product_image, W, prod_h)
            prod_rgba = prod.convert("RGBA")
            fade = Image.new("RGBA", (W, prod_h), (0, 0, 0, 0))
            fd = ImageDraw.Draw(fade)
            fade_start = int(prod_h * 0.65)
            for row in range(fade_start, prod_h):
                t = (row - fade_start) / (prod_h - fade_start)
                fd.line([(0, row), (W, row)], fill=(12, 12, 18, int(255 * t)))
            prod_rgba = Image.alpha_composite(prod_rgba, fade)
            canvas.paste(prod_rgba, (0, 0))

        draw = ImageDraw.Draw(canvas)

        div_y = int(H * 0.62)
        line_w = int(W * 0.4)
        draw.line([((W - line_w) // 2, div_y), ((W + line_w) // 2, div_y)],
                  fill=(*accent, 160), width=1)

        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=50, padding=30, backing=False)
        draw = ImageDraw.Draw(canvas)

        brand_display = content.brand_name.replace("_", " ").upper()
        brand_font = self._font("body", 12)
        canvas = self._draw_spaced_text(canvas, None, brand_display, brand_font,
                                        fill=(*accent, 200), spacing=5,
                                        center_x=W // 2, y_pos=div_y + 14)
        draw = ImageDraw.Draw(canvas)

        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("elegant", 36)
        lines = self._word_wrap(draw, headline, head_font, W - 140)
        y = div_y + 40
        for line in lines[:2]:
            tw, _ = self._text_size(draw, line, head_font)
            draw.text(((W - tw) // 2, y), line, fill=(240, 240, 245, 255), font=head_font)
            y += 44

        y += 10
        feat_font = self._font("body", 13)
        feat_text = "  \u2022  ".join(content.features[:3])
        ftw, _ = self._text_size(draw, feat_text, feat_font)
        draw.text(((W - ftw) // 2, y), feat_text, fill=(170, 170, 180, 190), font=feat_font)
        y += 26

        cta = self._get_cta_text(content)
        cta_font = self._font("headline", 18)
        ctw, _ = self._text_size(draw, cta, cta_font)
        btn_w = max(ctw + 56, 180)
        canvas, _ = self._draw_cta_button(canvas, cta,
                                          ((W - btn_w) // 2, min(y + 10, H - 60)),
                                          accent, style="outline", font_size=18)

        return canvas

    # === THEME 4: Side-by-Side ===
    def _render_split_layout(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)
        split_x = int(W * 0.52)

        darker = tuple(max(0, c - 50) for c in accent)
        canvas = Image.new("RGBA", (W, H), (*accent, 255))
        draw = ImageDraw.Draw(canvas)
        for row in range(H):
            t = row / H
            rgb = tuple(int(accent[i] + (darker[i] - accent[i]) * t) for i in range(3))
            draw.line([(split_x, row), (W, row)], fill=(*rgb, 255))

        if content.product_image:
            prod = self._cover_fill(content.product_image, split_x, H)
            canvas.paste(prod.convert("RGBA"), (0, 0))
        else:
            lighter = tuple(min(255, c + 60) for c in accent)
            for row in range(H):
                t = row / H
                rgb = tuple(int(lighter[i] * (1 - t * 0.3)) for i in range(3))
                draw.line([(0, row), (split_x, row)], fill=(*rgb, 255))

        blend_w = 40
        blend = Image.new("RGBA", (blend_w, H), (0, 0, 0, 0))
        bd = ImageDraw.Draw(blend)
        for bx in range(blend_w):
            t = bx / blend_w
            alpha = int(255 * (t ** 0.8))
            bd.line([(bx, 0), (bx, H)], fill=(*accent, alpha))
        canvas.paste(blend, (split_x - blend_w, 0), blend)

        draw = ImageDraw.Draw(canvas)
        lum = accent[0] * 0.299 + accent[1] * 0.587 + accent[2] * 0.114
        text_color = (255, 255, 255, 255) if lum < 140 else (20, 20, 30, 255)
        margin_r = split_x + 35
        text_max_w = W - margin_r - 35

        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=45, padding=25, backing=False)
        draw = ImageDraw.Draw(canvas)

        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("headline", 32)
        lines = self._word_wrap(draw, headline, head_font, text_max_w)
        y = int(H * 0.28)
        for line in lines[:2]:
            draw.text((margin_r, y), line, fill=text_color, font=head_font)
            y += 40

        y += 10
        draw.rounded_rectangle([margin_r, y, margin_r + 45, y + 3], radius=2,
                               fill=(*text_color[:3], 180))
        y += 24

        feat_font = self._font("body", 14)
        for feat in content.features[:3]:
            draw.text((margin_r, y), f"\u2022 {feat}",
                      fill=(*text_color[:3], 200), font=feat_font)
            y += 24

        y += 14
        btn_color = (255, 255, 255) if lum < 140 else accent
        canvas, _ = self._draw_cta_button(canvas, self._get_cta_text(content),
                                          (margin_r, min(y, H - 80)), btn_color,
                                          style="rounded", font_size=16)

        return canvas

    # === THEME 5: Product Showcase ===
    def _render_card_float(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)

        canvas = Image.new("RGBA", (W, H), (*accent, 255))
        draw = ImageDraw.Draw(canvas)
        darker = tuple(max(0, c - 50) for c in accent)
        for row in range(H):
            t = row / H
            rgb = tuple(int(accent[i] + (darker[i] - accent[i]) * t * 0.5) for i in range(3))
            draw.line([(0, row), (W, row)], fill=(*rgb, 255))

        prod_h = int(H * 0.62)
        if content.product_image:
            prod = self._cover_fill(content.product_image, W, prod_h)
            canvas.paste(prod.convert("RGBA"), (0, 0))

        wave_y = prod_h - 18
        canvas = self._draw_wave_divider(canvas, wave_y, (255, 255, 255, 200),
                                          amplitude=22, thickness=3,
                                          fill_below=(*accent, 252))

        draw = ImageDraw.Draw(canvas)
        margin = 60

        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=50, padding=30, backing=False)
        draw = ImageDraw.Draw(canvas)

        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("headline", 34)
        lum = accent[0] * 0.299 + accent[1] * 0.587 + accent[2] * 0.114
        head_color = (255, 255, 255, 255) if lum < 140 else (20, 20, 30, 255)
        text_w = W - 2 * margin
        lines = self._word_wrap(draw, headline, head_font, text_w)
        y = wave_y + 48
        for line in lines[:2]:
            tw, _ = self._text_size(draw, line, head_font)
            draw.text(((W - tw) // 2, y), line, fill=head_color, font=head_font)
            y += 42

        y += 10
        feat_font = self._font("body", 13)
        feat_color = (*head_color[:3], 190)
        feat_text = "  \u2022  ".join(content.features[:3])
        ftw, _ = self._text_size(draw, feat_text, feat_font)
        draw.text(((W - ftw) // 2, y), feat_text, fill=feat_color, font=feat_font)
        y += 26

        cta = self._get_cta_text(content)
        cta_font = self._font("headline", 17)
        ctw, _ = self._text_size(draw, cta, cta_font)
        btn_w = max(ctw + 50, 170)
        btn_color = (255, 255, 255) if lum < 140 else accent
        canvas, _ = self._draw_cta_button(canvas, cta,
                                          ((W - btn_w) // 2, min(y + 10, H - 60)),
                                          btn_color, style="rounded", font_size=17)

        return canvas

    # === THEME 6: Vibrant Mesh ===
    def _render_gradient_mesh(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)
        secondary = self._hex_to_rgb(content.secondary_color)

        canvas = Image.new("RGBA", (W, H), (20, 20, 35, 255))
        mesh_arr = np.zeros((H, W, 4), dtype=float)
        Y_grid, X_grid = np.ogrid[:H, :W]
        for cx, cy, color, rf in [
            (W * 0.2, H * 0.78, accent, 0.5),
            (W * 0.8, H * 0.85, secondary, 0.45),
            (W * 0.5, H * 0.92, tuple(max(0, int(c * 0.8)) for c in accent), 0.55),
        ]:
            radius = max(W, H) * rf
            dist = np.sqrt((X_grid - cx) ** 2 + (Y_grid - cy) ** 2)
            falloff = np.clip(1 - dist / radius, 0, 1) ** 2
            for c_idx in range(3):
                mesh_arr[:, :, c_idx] += color[c_idx] * falloff * 0.4
            mesh_arr[:, :, 3] += 120 * falloff * 0.5
        mesh_arr = np.clip(mesh_arr, 0, 255).astype(np.uint8)
        canvas = Image.alpha_composite(canvas, Image.fromarray(mesh_arr, "RGBA"))

        prod_h = int(H * 0.65)
        if content.product_image:
            prod = self._cover_fill(content.product_image, W, prod_h)
            canvas.paste(prod.convert("RGBA"), (0, 0))

        wave_y = prod_h - 18
        canvas = self._draw_wave_divider(canvas, wave_y, (*accent, 180),
                                          amplitude=25, thickness=3,
                                          fill_below=(20, 20, 35, 235))

        draw = ImageDraw.Draw(canvas)
        margin = 60

        canvas = self._place_logo(canvas, content.logo_image, "top-left",
                                  max_height=50, backing=False)
        draw = ImageDraw.Draw(canvas)

        headline = content.headline or content.tagline or content.brand_name.replace("_", " ")
        head_font = self._font("headline", 36)
        lines = self._word_wrap(draw, headline, head_font, W - 2 * margin)
        y = wave_y + 48
        for line in lines[:2]:
            self._draw_text_shadow(draw, (margin, y), line, head_font,
                                   fill=(255, 255, 255, 255))
            y += 44

        y += 10
        feat_font = self._font("body", 14)
        feat_text = "  \u2022  ".join(content.features[:3])
        draw.text((margin, y), feat_text, fill=(200, 200, 215, 200), font=feat_font)
        y += 28

        canvas, _ = self._draw_cta_button(canvas, self._get_cta_text(content),
                                          (margin, min(y + 6, H - 60)), accent,
                                          style="rounded", font_size=18)

        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, H - 4, W, H], fill=(*accent, 220))

        return canvas

    # === THEME 7: Realistic Pamphlet ===
    def _render_realistic_pamphlet(self, content) -> Image.Image:
        W, H = self.WIDTH, self.HEIGHT
        accent = self._hex_to_rgb(content.accent_color)
        secondary = self._hex_to_rgb(content.secondary_color)
        base_top = tuple(min(255, int(0.76 * c + 0.24 * 245)) for c in secondary)
        base_mid = tuple(min(255, int(0.60 * c + 0.40 * 232)) for c in secondary)
        base_bottom = tuple(max(0, int(0.55 * c + 0.45 * 28)) for c in accent)

        canvas = Image.new("RGBA", (W, H), (*base_bottom, 255))
        draw = ImageDraw.Draw(canvas)
        for y in range(H):
            t = y / max(1, H - 1)
            if t < 0.58:
                tt = t / 0.58
                rgb = tuple(int(base_top[i] * (1 - tt) + base_mid[i] * tt) for i in range(3))
            else:
                tt = (t - 0.58) / 0.42
                rgb = tuple(int(base_mid[i] * (1 - tt) + base_bottom[i] * tt) for i in range(3))
            draw.line([(0, y), (W, y)], fill=(*rgb, 255))

        # Soft studio lighting only; keep it realistic and uncluttered.
        light = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(light)
        ld.ellipse([int(W * 0.18), int(H * 0.20), int(W * 0.82), int(H * 0.84)], fill=(*secondary, 92))
        ld.ellipse([int(W * 0.34), int(H * 0.30), int(W * 0.66), int(H * 0.70)], fill=(255, 245, 230, 78))
        light = light.filter(ImageFilter.GaussianBlur(42))
        canvas = Image.alpha_composite(canvas, light)

        # Keep background cinematic and clean: avoid decorative geometry/circles.
        mood = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        md = ImageDraw.Draw(mood)
        md.rectangle([0, 0, W, int(H * 0.30)], fill=(20, 14, 10, 36))
        md.rectangle([0, int(H * 0.76), W, H], fill=(20, 14, 10, 62))
        mood = mood.filter(ImageFilter.GaussianBlur(24))
        canvas = Image.alpha_composite(canvas, mood)

        if content.logo_image:
            canvas = self._place_logo(canvas, content.logo_image, "top-center", max_height=54, padding=28, backing=False)

        draw = ImageDraw.Draw(canvas)
        sample_text = " ".join([content.brand_name or "", content.headline or "", content.tagline or "", content.cta_text or ""])
        self._script = self._detect_script(sample_text)

        brand = (content.brand_name or "").replace("_", " ").upper()
        brand_font = self._font("body", 14)
        if brand:
            bw, _ = self._text_size(draw, brand, brand_font)
            bx = (W - bw) // 2
            draw.text((bx, 66), brand, font=brand_font, fill=(66, 44, 28, 210))

        headline = content.headline or content.tagline or brand or "Premium Product"
        head_font = self._font("headline", 78)
        lines = self._word_wrap(draw, headline, head_font, int(W * 0.86))
        y = 118
        for line in lines[:2]:
            tw, th = self._text_size(draw, line, head_font)
            x = (W - tw) // 2
            draw.text((x, y + 3), line, font=head_font, fill=(58, 34, 20, 125))
            draw.text((x, y), line, font=head_font, fill=(255, 246, 232, 242))
            y += th + 6

        sub_font = self._font("body", 28)
        subtitle = content.tagline or ""
        if subtitle:
            lines = self._word_wrap(draw, subtitle, sub_font, int(W * 0.78))
            if lines:
                tw, _ = self._text_size(draw, lines[0], sub_font)
                x = (W - tw) // 2
                y += 8
                draw.text((x, y + 2), lines[0], font=sub_font, fill=(58, 34, 20, 110))
                draw.text((x, y), lines[0], font=sub_font, fill=(244, 229, 202, 225))
                y += 42

        # Product hero centered with realistic shadow.
        if content.product_image:
            prod = content.product_image.copy().convert("RGBA")
            if prod.mode != "RGBA":
                prod = prod.convert("RGBA")
            pw, ph = prod.size
            max_w = int(W * 0.62)
            max_h = int(H * 0.42)
            scale = min(max_w / max(1, pw), max_h / max(1, ph), 1.0)
            prod = prod.resize((max(1, int(pw * scale)), max(1, int(ph * scale))), Image.LANCZOS)
            px = (W - prod.width) // 2
            py = int(H * 0.50) - prod.height // 2
            py = max(270, min(py, 610))

            shadow = Image.new("RGBA", (prod.width, prod.height), (0, 0, 0, 0))
            sa = prod.split()[-1].filter(ImageFilter.GaussianBlur(14))
            shadow.putalpha(sa.point(lambda p: min(p, 72)))
            canvas.alpha_composite(shadow, (px, py + 18))
            canvas.alpha_composite(prod, (px, py))

        # Bottom copy and CTA.
        feat_font = self._font("body", 16)
        feat_text = "  \u2022  ".join(content.features[:3]) if content.features else ""
        if feat_text:
            ftw, _ = self._text_size(draw, feat_text, feat_font)
            fx = (W - ftw) // 2
            draw.text((fx, 858), feat_text, font=feat_font, fill=(244, 228, 201, 230))

        cta = self._get_cta_text(content)
        canvas, _ = self._draw_cta_button(canvas, cta, ((W - 240) // 2, 912), accent, style="rounded", font_size=20)

        return canvas


PamphletComposer = ProAdDesigner  # backward compatibility
