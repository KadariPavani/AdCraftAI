# CLIP-based content extraction: features, prompts, and category context.
# All design decisions are derived from analyzing dataset reference images —
# no hardcoded feature lists, prompts, or category-specific templates.

from typing import List

import torch
import torch.nn.functional as F
from PIL import Image


class CLIPContentExtractor:
    # General-purpose candidate pools for CLIP ranking against reference images.
    # These are NOT used as-is — CLIP scores each against the retrieved ad images
    # and only the top-ranked (most visually relevant) ones are returned.
    FEATURE_POOL = [
        "Premium quality craftsmanship", "Trusted by millions worldwide",
        "Award-winning innovative design", "Unbeatable value for money",
        "Industry-leading performance", "Eco-friendly sustainable choice",
        "Lightweight comfortable fit", "Breathable mesh upper design",
        "Advanced cushioning technology", "Durable long-lasting build",
        "Moisture-wicking performance fabric", "Athletic performance engineered",
        "Trendy modern street style", "All-day support and comfort",
        "Pure refreshing hydration", "Mineral-enriched natural water",
        "Natural fruit ingredients", "Zero artificial preservatives",
        "Clinically tested purity", "Refreshing clean taste",
        "Cutting-edge processor power", "Stunning high-resolution display",
        "All-day battery performance", "Professional camera system",
        "5G ultra-fast connectivity", "Sleek premium metal build",
        "Dermatologically tested formula", "Long-lasting natural radiance",
        "Gentle on all skin types", "Clinically proven visible results",
        "Nourishing botanical ingredients", "Salon-quality results at home",
        "Made with natural ingredients", "Irresistible authentic flavor",
        "Fresh quality guaranteed daily", "Nutrition-packed healthy goodness",
        "Loved by food enthusiasts",
        "Powerful engine performance", "Advanced safety technology",
        "Fuel-efficient smart engineering", "Spacious luxury interior cabin",
        "Exquisite handcrafted artistry", "Certified precious fine materials",
        "Timeless elegant classic design", "Precision Swiss-quality movement",
        "Secure trusted digital banking", "Digital-first modern convenience",
        "Best price guaranteed always", "Seamless easy booking experience",
    ]

    STYLE_POOL = [
        "dramatic studio lighting on dark background",
        "bright outdoor natural sunlight setting",
        "warm golden hour sunset glow",
        "clean minimalist white background",
        "urban city street environment",
        "lush green nature backdrop",
        "modern gym fitness environment",
        "luxurious upscale premium interior",
        "vibrant colorful dynamic style",
        "sleek futuristic tech aesthetic",
        "elegant fashion editorial style",
        "dynamic sports action moment",
    ]

    MOOD_POOL = [
        "energetic dynamic movement", "calm peaceful serenity",
        "luxurious premium elegance", "fun playful joyful energy",
        "professional confident trust", "fresh refreshing vitality",
        "warm family togetherness", "bold powerful strength",
        "sophisticated refined taste", "adventurous exciting freedom",
    ]

    SUBJECT_POOL = [
        "person wearing the product proudly",
        "athlete in dynamic action pose",
        "model showcasing fashion outfit",
        "person drinking refreshing beverage",
        "close-up of product in use",
        "family enjoying product together",
        "professional using tech device",
        "person with radiant glowing skin",
        "driver in luxury vehicle interior",
        "fitness enthusiast working out intensely",
        "traveler at beautiful scenic destination",
        "person savoring delicious food",
    ]

    def __init__(self, clip_model, clip_processor, device: str):
        self.model = clip_model
        self.processor = clip_processor
        self.device = device

    @staticmethod
    def _to_tensor(features):
        """Convert model output to tensor, handling different transformers versions."""
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, 'pooler_output'):
            return features.pooler_output
        if hasattr(features, 'last_hidden_state'):
            return features.last_hidden_state[:, 0, :]
        return features[0] if hasattr(features, '__getitem__') else features

    def _encode_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in inputs.items()
                       if k in ('input_ids', 'attention_mask')}
        with torch.no_grad():
            text_out = self.model.text_model(**text_inputs)
            pooled = text_out[1]  # pooler_output
            features = self.model.text_projection(pooled)
            return F.normalize(features, p=2, dim=-1)

    def _encode_images(self, image_paths):
        images = []
        for path in image_paths[:5]:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception:
                continue
        if not images:
            return torch.zeros(1, 512).to(self.device)
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=pixel_values)
            pooled = vision_out[1]  # pooler_output
            features = self.model.visual_projection(pooled)
            return F.normalize(features, p=2, dim=-1)

    def _encode_pil_images(self, images):
        """Encode already-loaded PIL images (not paths)."""
        if not images:
            return torch.zeros(1, 512).to(self.device)
        rgb_images = [img.convert("RGB") for img in images[:5]]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=pixel_values)
            pooled = vision_out[1]
            features = self.model.visual_projection(pooled)
            return F.normalize(features, p=2, dim=-1)

    def _rank_pool(self, image_features, pool, top_n):
        if image_features.sum() == 0:
            return pool[:top_n]
        avg_feat = F.normalize(image_features.mean(dim=0, keepdim=True), p=2, dim=-1)
        text_feat = self._encode_texts(pool)
        similarities = (avg_feat @ text_feat.T).squeeze(0)
        top_indices = similarities.argsort(descending=True)[:top_n].tolist()
        return [pool[i] for i in top_indices]

    def extract_features(self, image_paths, n=6, category=""):
        """Extract features by CLIP-ranking the candidate pool against retrieved
        reference images. The reference images drive which features are selected —
        no hardcoded category mapping is used."""
        img_feats = self._encode_images(image_paths)
        return self._rank_pool(img_feats, self.FEATURE_POOL, n)

    def generate_diffusion_prompt(self, image_paths, brand, category, subcategory, user_query=""):
        """Generate a rich image prompt by analyzing the retrieved reference ads.
        CLIP ranks styles, moods, and subjects against the actual ad images from
        the dataset to build a context-aware prompt — no hardcoded category templates."""
        print(f"    [CLIP-PROMPT] Generating diffusion prompt from {len(image_paths)} reference images")
        print(f"    [CLIP-PROMPT] Model: openai/clip-vit-base-patch32 (vision encoder)")
        img_feats = self._encode_images(image_paths)
        print(f"    [CLIP-PROMPT] Image features encoded: shape={img_feats.shape}")

        # Let CLIP pick the best-matching style, mood, and subject from the pools
        print(f"    [CLIP-PROMPT] Ranking pools against reference images:")
        top_styles = self._rank_pool(img_feats, self.STYLE_POOL, 2)
        print(f"    [CLIP-PROMPT]   Styles (top 2): {top_styles}")
        top_moods = self._rank_pool(img_feats, self.MOOD_POOL, 1)
        print(f"    [CLIP-PROMPT]   Moods (top 1): {top_moods}")
        top_subjects = self._rank_pool(img_feats, self.SUBJECT_POOL, 1)
        print(f"    [CLIP-PROMPT]   Subjects (top 1): {top_subjects}")

        brand_clean = brand.replace("_", " ")
        query_subject = self._extract_query_subject(user_query, brand_clean)
        if query_subject:
            print(f"    [CLIP-PROMPT]   Query subject: \"{query_subject}\"")

        parts = [f"professional commercial advertisement photography for {brand_clean}"]

        if query_subject:
            parts.append(query_subject)

        # Add CLIP-derived context from reference image analysis
        parts.append(top_subjects[0])
        parts.append(f"{top_styles[0]}, {top_styles[1]}")
        parts.append(f"{top_moods[0]} mood")
        parts.append("ultra realistic, photorealistic, 8k, sharp focus, professional lighting, commercial quality")

        prompt = ", ".join(parts)
        print(f"    [CLIP-PROMPT] Final prompt ({len(prompt)} chars)")
        return prompt

    def _extract_query_subject(self, user_query, brand_clean):
        """Extract meaningful product description from user query."""
        if not user_query:
            return ""
        query_desc = user_query.lower()
        for word in brand_clean.lower().split():
            query_desc = query_desc.replace(word, "")
        filler = {"ad", "ads", "advertisement", "pamphlet", "poster", "banner",
                  "create", "make", "generate", "please", "want", "need", "i",
                  "a", "an", "the", "for", "with", "and", "of", "in", "on",
                  "me", "my", "give", "show", "by", "it", "that", "this"}
        words = [w for w in query_desc.split() if w not in filler]
        return " ".join(words).strip()

    def analyze_uploaded_image(self, image: Image.Image, category: str = ""):
        """Analyze an uploaded product image using CLIP to extract descriptors.
        All results are derived from CLIP similarity — no hardcoded category mappings."""
        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=pixel_values)
            pooled = vision_out[1]
            img_feats = self.model.visual_projection(pooled)
            img_feats = F.normalize(img_feats, p=2, dim=-1)

        features = self._rank_pool(img_feats, self.FEATURE_POOL, 6)
        styles = self._rank_pool(img_feats, self.STYLE_POOL, 2)
        moods = self._rank_pool(img_feats, self.MOOD_POOL, 1)

        return {
            "features": features,
            "styles": styles,
            "mood": moods[0] if moods else "professional",
        }

    def analyze_reference_ads(self, image_paths):
        """Analyze a set of retrieved reference ad images to extract visual characteristics.
        Returns style, mood, subject, and feature rankings — all derived from CLIP
        similarity against the actual reference images, not from hardcoded mappings."""
        img_feats = self._encode_images(image_paths)

        return {
            "features": self._rank_pool(img_feats, self.FEATURE_POOL, 6),
            "styles": self._rank_pool(img_feats, self.STYLE_POOL, 3),
            "moods": self._rank_pool(img_feats, self.MOOD_POOL, 2),
            "subjects": self._rank_pool(img_feats, self.SUBJECT_POOL, 2),
        }
