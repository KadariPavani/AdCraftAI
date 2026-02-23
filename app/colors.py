# KMeans-based dominant color extraction from ad images.

from collections import Counter
from typing import List

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


class ColorExtractor:
    RESIZE_DIM = 100

    def extract_from_images(self, image_paths: List[str], n_colors: int = 5) -> List[str]:
        all_pixels = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                img_small = img.resize((self.RESIZE_DIM, self.RESIZE_DIM))
                pixels = np.array(img_small).reshape(-1, 3)
                all_pixels.append(pixels)
            except Exception:
                continue

        if not all_pixels:
            return ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560"]

        combined = np.vstack(all_pixels)
        if len(combined) > 30000:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(combined), 30000, replace=False)
            combined = combined[indices]

        km = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
        km.fit(combined)

        colors_rgb = km.cluster_centers_.astype(int)
        counts = Counter(km.labels_)
        sorted_labels = sorted(counts.items(), key=lambda x: -x[1])

        hex_colors = []
        for label, _ in sorted_labels:
            c = colors_rgb[label]
            hex_colors.append(f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}")
        return hex_colors

    @staticmethod
    def get_accent_color(hex_colors: List[str]) -> str:
        best = hex_colors[0] if hex_colors else "#1a1a2e"
        best_score = -1
        for h in hex_colors:
            r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            sat = (max_c - min_c) / max_c if max_c > 0 else 0
            brightness = (r + g + b) / 3
            if 30 < brightness < 220:
                score = sat * 100 + (1 - abs(brightness - 128) / 128) * 30
            else:
                score = sat * 20
            if score > best_score:
                best_score = score
                best = h
        return best

    @staticmethod
    def lighten_color(hex_color: str, factor: float = 0.92) -> str:
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def darken_color(hex_color: str, factor: float = 0.6) -> str:
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"
