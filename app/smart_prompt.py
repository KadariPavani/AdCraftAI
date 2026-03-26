# Smart Prompt Parser — Extracts structured product catalog data from free-text input.
# Uses AI (Pollinations) to parse a single prompt into Amazon-like catalog fields,
# then identifies which required fields are missing based on product category.

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.content_gen import PollinationsTextGenerator
from app.brands import BrandMatcher


# ---------------------------------------------------------------------------
# Category-specific required fields (Amazon catalog style)
# ---------------------------------------------------------------------------

# Each category defines:
#   "required": fields that MUST be present to generate a good ad
#   "optional": fields that improve quality but aren't mandatory
#   "display_name": human-readable category name

CATEGORY_FIELDS = {
    "footwear": {
        "display_name": "Footwear",
        "required": {
            "brand": "Brand name (e.g., Nike, Adidas, Puma)",
            "product_name": "Product name (e.g., Air Max 90, Ultraboost)",
            "product_type": "Type of footwear (e.g., running shoes, sneakers, sandals)",
            "target_audience": "Target audience (e.g., men, women, kids, unisex)",
            "size_range": "Available sizes (e.g., UK 6-12, US 7-13)",
        },
        "optional": {
            "material": "Material (e.g., mesh, leather, synthetic)",
            "color": "Color/colorway (e.g., black/white, triple red)",
            "price": "Price or price range",
            "key_features": "Key features (e.g., cushioning, waterproof, lightweight)",
            "occasion": "Occasion (e.g., running, casual, formal, sports)",
        },
    },
    "electronics": {
        "display_name": "Electronics",
        "required": {
            "brand": "Brand name (e.g., Samsung, Apple, Sony)",
            "product_name": "Product model name (e.g., Galaxy S24, iPhone 16)",
            "product_type": "Product type (e.g., smartphone, laptop, earbuds)",
            "key_features": "Key features/specs (e.g., 200MP camera, 5000mAh battery)",
        },
        "optional": {
            "price": "Price or price range",
            "storage": "Storage/memory (e.g., 256GB, 8GB RAM)",
            "display_size": "Display/screen size (e.g., 6.7 inch, 15.6 inch)",
            "color": "Available colors",
            "target_audience": "Target audience",
            "manufacturer": "Manufacturer details",
            "warranty": "Warranty information",
        },
    },
    "clothing": {
        "display_name": "Clothing & Apparel",
        "required": {
            "brand": "Brand name (e.g., Zara, H&M, Levi's)",
            "product_name": "Product name/style",
            "product_type": "Type (e.g., t-shirt, jeans, dress, jacket)",
            "target_audience": "Target audience (e.g., men, women, kids)",
            "size_range": "Available sizes (e.g., S/M/L/XL, 28-36)",
        },
        "optional": {
            "material": "Fabric/material (e.g., cotton, polyester, silk)",
            "color": "Colors available",
            "price": "Price",
            "occasion": "Occasion (e.g., casual, formal, party)",
            "care_instructions": "Care instructions (e.g., machine wash, dry clean)",
        },
    },
    "beverages": {
        "display_name": "Beverages",
        "required": {
            "brand": "Brand name (e.g., Coca-Cola, Bisleri, Red Bull)",
            "product_name": "Product name",
            "product_type": "Type (e.g., mineral water, soft drink, energy drink, juice)",
            "pack_size": "Pack size/volume (e.g., 500ml, 1L, 330ml can)",
        },
        "optional": {
            "flavor": "Flavor/variant (e.g., original, lime, mango)",
            "price": "Price",
            "ingredients": "Key ingredients or nutritional highlights",
            "target_audience": "Target audience",
            "manufacturer": "Manufacturer/bottler details",
        },
    },
    "jewelry": {
        "display_name": "Jewelry & Watches",
        "required": {
            "brand": "Brand name (e.g., Tanishq, Kalyan, Titan)",
            "product_name": "Product name/collection",
            "product_type": "Type (e.g., gold necklace, diamond ring, bangles, watch)",
            "material": "Material (e.g., 22K gold, platinum, sterling silver, diamond)",
        },
        "optional": {
            "weight": "Weight (e.g., 10g, 5 carat)",
            "size_range": "Size details (e.g., ring size 6-12, wrist size)",
            "price": "Price range",
            "occasion": "Occasion (e.g., wedding, daily wear, festive)",
            "target_audience": "Target audience (e.g., women, men, couples)",
            "certification": "Certification (e.g., BIS hallmark, GIA certified)",
        },
    },
    "automotive": {
        "display_name": "Automotive",
        "required": {
            "brand": "Brand name (e.g., Mahindra, Toyota, BMW)",
            "product_name": "Model name (e.g., Thar, Fortuner, 3 Series)",
            "product_type": "Vehicle type (e.g., SUV, sedan, hatchback, motorcycle)",
            "key_features": "Key features (e.g., 4x4, sunroof, diesel engine)",
        },
        "optional": {
            "price": "Price / ex-showroom price",
            "engine": "Engine specs (e.g., 2.0L turbo, 150 HP)",
            "mileage": "Fuel efficiency (e.g., 15 km/l)",
            "color": "Available colors",
            "seating": "Seating capacity",
            "target_audience": "Target audience",
        },
    },
    "food": {
        "display_name": "Food & Snacks",
        "required": {
            "brand": "Brand name (e.g., Lays, Amul, Maggi)",
            "product_name": "Product name",
            "product_type": "Type (e.g., chips, chocolate, instant noodles, cheese)",
            "pack_size": "Pack size/weight (e.g., 100g, 500g, family pack)",
        },
        "optional": {
            "flavor": "Flavor/variant",
            "price": "Price",
            "ingredients": "Key ingredients",
            "dietary_info": "Dietary info (e.g., vegetarian, gluten-free)",
            "manufacturer": "Manufacturer details",
        },
    },
    "personal_care": {
        "display_name": "Personal Care & Beauty",
        "required": {
            "brand": "Brand name (e.g., Dove, L'Oreal, Nivea)",
            "product_name": "Product name",
            "product_type": "Type (e.g., shampoo, face cream, lipstick, perfume)",
            "target_audience": "Target audience (e.g., men, women, all)",
        },
        "optional": {
            "size": "Size/quantity (e.g., 200ml, 50g)",
            "key_features": "Key benefits (e.g., anti-dandruff, moisturizing)",
            "ingredients": "Key ingredients (e.g., aloe vera, vitamin E)",
            "skin_type": "Suitable skin/hair type",
            "price": "Price",
        },
    },
    "home_appliances": {
        "display_name": "Home Appliances",
        "required": {
            "brand": "Brand name (e.g., LG, Samsung, Dyson)",
            "product_name": "Product model name",
            "product_type": "Type (e.g., washing machine, air conditioner, vacuum)",
            "key_features": "Key features/specs",
        },
        "optional": {
            "capacity": "Capacity (e.g., 7kg, 1.5 ton)",
            "energy_rating": "Energy rating (e.g., 5 star, A+++)",
            "price": "Price",
            "color": "Available colors",
            "warranty": "Warranty period",
            "dimensions": "Dimensions/size",
        },
    },
    "general": {
        "display_name": "General Product",
        "required": {
            "brand": "Brand name",
            "product_name": "Product name",
            "product_type": "What kind of product is this",
        },
        "optional": {
            "target_audience": "Target audience",
            "key_features": "Key features or benefits",
            "price": "Price",
            "size_range": "Size/dimensions",
            "material": "Material/composition",
            "color": "Color options",
            "manufacturer": "Manufacturer details",
        },
    },
}


# ---------------------------------------------------------------------------
# Smart Prompt Parser
# ---------------------------------------------------------------------------

class SmartPromptParser:
    """Parses free-text product descriptions into structured catalog data."""

    def __init__(self, pipeline=None):
        self.text_gen = PollinationsTextGenerator()
        self.pipeline = pipeline  # Set later via set_pipeline()
        print(f"    [SMART-PROMPT] SmartPromptParser initialized")

    def set_pipeline(self, pipeline):
        """Connect to the pipeline for brand/dataset awareness."""
        self.pipeline = pipeline

    def get_category_fields(self, category: str) -> Dict[str, Any]:
        """Return the field definitions for a given category."""
        cat_key = self._normalize_category(category)
        fields = CATEGORY_FIELDS.get(cat_key, CATEGORY_FIELDS["general"])
        return {
            "category": cat_key,
            "display_name": fields["display_name"],
            "required": fields["required"],
            "optional": fields["optional"],
        }

    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse a free-text prompt into structured product data.

        Returns:
            {
                "extracted": { field: value, ... },
                "category": "detected_category",
                "missing_required": { field: description, ... },
                "all_required": { field: description, ... },
                "all_optional": { field: description, ... },
                "completeness": 0.0-1.0,
                "rich_prompt": "enhanced prompt with all data",
                "brand_status": "known|unknown",
                "dataset_brand": "Matched_Brand or null",
                "dataset_category": "category from dataset or null",
                "related_brands": ["Brand1", "Brand2", ...],
            }
        """
        print(f"    [SMART-PROMPT] Parsing prompt: \"{prompt[:100]}...\"")

        # Step 1: Use AI to extract structured data from the prompt
        extracted = self._extract_fields_via_ai(prompt)
        if not extracted:
            print(f"    [SMART-PROMPT] AI extraction failed, using regex fallback")
            extracted = self._extract_fields_regex(prompt)

        # Step 2: Normalize brand name if present
        if extracted.get("brand"):
            extracted["brand"] = BrandMatcher.normalize_brand(extracted["brand"])

        # Step 3: Detect category from extracted data or AI
        category = extracted.get("category", "")
        if not category:
            category = self._detect_category(prompt, extracted)
        extracted["category"] = category

        cat_key = self._normalize_category(category)
        cat_fields = CATEGORY_FIELDS.get(cat_key, CATEGORY_FIELDS["general"])

        # Step 4: Check brand against dataset
        brand_status = "unknown"
        dataset_brand = None
        dataset_category = None
        related_brands = []

        if self.pipeline and extracted.get("brand"):
            brand_info = self._check_brand_in_dataset(extracted["brand"], prompt)
            brand_status = brand_info["status"]
            dataset_brand = brand_info.get("matched_brand")
            dataset_category = brand_info.get("category")
            related_brands = brand_info.get("related_brands", [])

            # Use dataset category if AI category is too generic
            if dataset_category and cat_key == "general":
                cat_key = self._normalize_category(dataset_category)
                cat_fields = CATEGORY_FIELDS.get(cat_key, CATEGORY_FIELDS["general"])
                extracted["category"] = dataset_category
                print(f"    [SMART-PROMPT] Using dataset category: {dataset_category}")

        # Step 5: Find missing required fields
        missing_required = {}
        for field_name, field_desc in cat_fields["required"].items():
            value = extracted.get(field_name, "")
            if not value or str(value).strip().lower() in ("", "unknown", "n/a", "none", "not specified"):
                missing_required[field_name] = field_desc

        # Step 6: Calculate completeness
        total_required = len(cat_fields["required"])
        filled_required = total_required - len(missing_required)
        completeness = filled_required / total_required if total_required > 0 else 1.0

        # Step 7: Build enriched prompt from all extracted data
        rich_prompt = self._build_rich_prompt(prompt, extracted, cat_key)

        result = {
            "extracted": extracted,
            "category": cat_key,
            "category_display": cat_fields["display_name"],
            "missing_required": missing_required,
            "all_required": cat_fields["required"],
            "all_optional": cat_fields["optional"],
            "completeness": round(completeness, 2),
            "rich_prompt": rich_prompt,
            "brand_status": brand_status,
            "dataset_brand": dataset_brand,
            "dataset_category": dataset_category,
            "related_brands": related_brands,
        }

        print(f"    [SMART-PROMPT] Category: {cat_key} ({cat_fields['display_name']})")
        print(f"    [SMART-PROMPT] Extracted {len(extracted)} fields")
        print(f"    [SMART-PROMPT] Missing required: {list(missing_required.keys())}")
        print(f"    [SMART-PROMPT] Completeness: {completeness:.0%}")
        print(f"    [SMART-PROMPT] Brand status: {brand_status} | Dataset brand: {dataset_brand}")
        if related_brands:
            print(f"    [SMART-PROMPT] Related brands in dataset: {related_brands[:5]}")

        return result

    def _check_brand_in_dataset(self, brand: str, query: str) -> Dict[str, Any]:
        """Check if the brand exists in the FAISS dataset and find related brands."""
        result = {"status": "unknown", "related_brands": []}

        if not self.pipeline:
            return result

        try:
            # Try matching the brand via the brand matcher
            brand_match = self.pipeline.brand_matcher.match(query)
            if brand_match.matched_brand and brand_match.confidence > 0.5:
                result["status"] = "known"
                result["matched_brand"] = brand_match.matched_brand
                result["confidence"] = brand_match.confidence
                result["category"] = brand_match.category
                result["subcategory"] = brand_match.subcategory
                result["image_count"] = brand_match.image_count
                print(f"    [SMART-PROMPT] Brand FOUND in dataset: {brand_match.matched_brand} "
                      f"(confidence={brand_match.confidence:.2f}, images={brand_match.image_count})")
            else:
                result["status"] = "unknown"
                print(f"    [SMART-PROMPT] Brand NOT in dataset: \"{brand}\"")

            # Find related brands in the same category
            detected_cat = result.get("category", "")
            if detected_cat:
                related = []
                for bname, binfo in self.pipeline.brand_matcher.brand_index.items():
                    if binfo["category"] == detected_cat and bname != result.get("matched_brand"):
                        related.append({
                            "brand": bname,
                            "count": binfo["count"],
                        })
                # Sort by image count, take top 5
                related.sort(key=lambda x: x["count"], reverse=True)
                result["related_brands"] = [r["brand"].replace("_", " ") for r in related[:5]]
            elif result["status"] == "unknown":
                # For unknown brands, try to find similar brands by name
                import difflib
                canonical = brand.lower().replace("_", "").replace(" ", "")
                all_brands_lower = self.pipeline.brand_matcher.brand_names_lower
                close = difflib.get_close_matches(canonical, all_brands_lower, n=5, cutoff=0.3)
                if close:
                    for match_lower in close:
                        idx = all_brands_lower.index(match_lower)
                        bname = self.pipeline.brand_matcher.brand_names[idx]
                        result["related_brands"].append(bname.replace("_", " "))

        except Exception as e:
            print(f"    [SMART-PROMPT] Brand check error: {e}")

        return result

    def _extract_fields_via_ai(self, prompt: str) -> Optional[Dict[str, str]]:
        """Use AI to extract structured product fields from free text."""
        system_prompt = (
            "You are a product data extraction expert. Given a product description, "
            "extract ALL available information into structured JSON fields.\n\n"
            "Extract these fields if mentioned (output empty string if not found):\n"
            "{\n"
            '  "brand": "brand/company name",\n'
            '  "product_name": "specific product name or model",\n'
            '  "product_type": "type/category of product (e.g., running shoes, smartphone)",\n'
            '  "category": "broad category (footwear/electronics/clothing/beverages/jewelry/automotive/food/personal_care/home_appliances/general)",\n'
            '  "target_audience": "intended audience (e.g., men, women, kids, professionals)",\n'
            '  "size_range": "sizes available",\n'
            '  "material": "material/fabric/composition",\n'
            '  "color": "color options",\n'
            '  "price": "price or price range",\n'
            '  "key_features": "main features, specs, or selling points",\n'
            '  "pack_size": "package size/weight/volume",\n'
            '  "occasion": "usage occasion",\n'
            '  "manufacturer": "manufacturer or maker details",\n'
            '  "weight": "product weight",\n'
            '  "warranty": "warranty information",\n'
            '  "flavor": "flavor/variant if applicable",\n'
            '  "scene_description": "any visual/scene description for the ad"\n'
            "}\n\n"
            "IMPORTANT:\n"
            "- Extract ONLY what is explicitly mentioned in the text\n"
            "- Do NOT guess or make up information\n"
            "- Output empty string for fields not found in the text\n"
            "- Output ONLY valid JSON, nothing else"
        )

        try:
            result = self.text_gen.generate_json(system_prompt, prompt)
            if result and isinstance(result, dict):
                # Clean up empty values
                cleaned = {}
                for k, v in result.items():
                    if v and str(v).strip() and str(v).strip().lower() not in ("", "n/a", "none", "not specified", "unknown", "not mentioned"):
                        cleaned[k] = str(v).strip()
                print(f"    [SMART-PROMPT] AI extracted {len(cleaned)} fields: {list(cleaned.keys())}")
                return cleaned
        except Exception as e:
            print(f"    [SMART-PROMPT] AI extraction error: {e}")

        return None

    def _extract_fields_regex(self, prompt: str) -> Dict[str, str]:
        """Fallback regex-based field extraction."""
        extracted = {}
        text = prompt.lower()

        # Price patterns
        price_match = re.search(r'(?:rs\.?|inr|usd|\$|price[:\s]+)\s*[\d,]+(?:\.\d{2})?', text, re.I)
        if price_match:
            extracted["price"] = price_match.group().strip()

        # Size patterns
        size_match = re.search(r'(?:size[s]?[:\s]+)([\w\s,/-]+?)(?:\.|,\s*\w|\n|$)', text, re.I)
        if size_match:
            extracted["size_range"] = size_match.group(1).strip()

        # Weight/volume
        weight_match = re.search(r'\b(\d+(?:\.\d+)?\s*(?:kg|g|gm|ml|l|liter|oz|lb))\b', text, re.I)
        if weight_match:
            extracted["pack_size"] = weight_match.group(1).strip()

        return extracted

    def _detect_category(self, prompt: str, extracted: Dict) -> str:
        """Detect product category from prompt and extracted data."""
        try:
            result = self.text_gen.generate(
                system_prompt=(
                    "Given a product description, respond with ONLY one of these categories "
                    "(exactly as written, lowercase):\n"
                    "footwear, electronics, clothing, beverages, jewelry, automotive, "
                    "food, personal_care, home_appliances, general\n\n"
                    "Output ONLY the category word, nothing else."
                ),
                user_prompt=prompt,
            )
            if result:
                cat = result.strip().lower().replace(" ", "_")
                if cat in CATEGORY_FIELDS:
                    return cat
        except Exception as e:
            print(f"    [SMART-PROMPT] Category detection failed: {e}")

        return "general"

    def _normalize_category(self, category: str) -> str:
        """Normalize category string to a known key."""
        if not category:
            return "general"
        cat = category.strip().lower().replace(" ", "_")

        # Map common aliases
        aliases = {
            "shoes": "footwear", "sneakers": "footwear", "boots": "footwear",
            "sandals": "footwear", "slippers": "footwear",
            "phone": "electronics", "smartphone": "electronics", "laptop": "electronics",
            "tablet": "electronics", "earbuds": "electronics", "headphones": "electronics",
            "tv": "electronics", "television": "electronics", "camera": "electronics",
            "clothes": "clothing", "apparel": "clothing", "fashion": "clothing",
            "shirt": "clothing", "jeans": "clothing", "dress": "clothing",
            "water": "beverages", "drink": "beverages", "juice": "beverages",
            "soda": "beverages", "coffee": "beverages", "tea": "beverages",
            "watches": "jewelry", "watch": "jewelry", "jewellery": "jewelry",
            "necklace": "jewelry", "ring": "jewelry", "bangles": "jewelry",
            "earrings": "jewelry", "bracelet": "jewelry",
            "car": "automotive", "bike": "automotive", "motorcycle": "automotive",
            "scooter": "automotive", "vehicle": "automotive", "suv": "automotive",
            "snacks": "food", "chips": "food", "chocolate": "food",
            "noodles": "food", "biscuits": "food", "cookies": "food",
            "shampoo": "personal_care", "cream": "personal_care", "skincare": "personal_care",
            "cosmetics": "personal_care", "perfume": "personal_care", "soap": "personal_care",
            "makeup": "personal_care", "beauty": "personal_care",
            "washing_machine": "home_appliances", "refrigerator": "home_appliances",
            "ac": "home_appliances", "air_conditioner": "home_appliances",
            "microwave": "home_appliances", "vacuum": "home_appliances",
            "stationery": "general", "toys": "general", "furniture": "general",
        }

        if cat in CATEGORY_FIELDS:
            return cat
        return aliases.get(cat, "general")

    def _build_rich_prompt(self, original_prompt: str, extracted: Dict, category: str) -> str:
        """Use LLM to build a comprehensive ad image generation prompt.

        The prompt instructs the image model to render a complete advertisement
        with ALL visual elements (headline, CTA button, tagline, features, price,
        brand name) baked directly into the generated image.
        """
        # Gather all product details for the LLM
        product_info_parts = []
        if extracted.get("brand"):
            product_info_parts.append(f"Brand: {extracted['brand']}")
        if extracted.get("product_name"):
            product_info_parts.append(f"Product: {extracted['product_name']}")
        if extracted.get("product_type"):
            product_info_parts.append(f"Type: {extracted['product_type']}")

        detail_fields = ["material", "color", "key_features", "target_audience",
                         "occasion", "pack_size", "size_range", "flavor", "price"]
        for field in detail_fields:
            val = extracted.get(field, "")
            if val:
                product_info_parts.append(f"{field.replace('_', ' ').title()}: {val}")

        if extracted.get("scene_description"):
            product_info_parts.append(f"Scene: {extracted['scene_description']}")

        product_info = "\n".join(product_info_parts)

        # Use LLM to craft a rich image generation prompt
        try:
            system_prompt = (
                "You are an expert at writing image generation prompts for creating complete advertisement images. "
                "Given product details, write a SINGLE detailed prompt that will generate a COMPLETE, READY-TO-USE "
                "advertisement image. The generated image MUST include ALL of the following elements rendered "
                "as part of the image itself (not as overlays):\n\n"
                "1. The product shown prominently (photorealistic or stylized based on category)\n"
                "2. A bold, catchy HEADLINE text rendered clearly in the image\n"
                "3. A CTA button (e.g., 'Shop Now', 'Buy Now', 'Order Today') with visible button shape\n"
                "4. Brand name displayed prominently\n"
                "5. Key selling points or features as text in the image\n"
                "6. Price if provided\n"
                "7. Professional ad layout with proper typography, colors, and composition\n\n"
                "The prompt should describe a polished, professional advertisement poster/banner that looks "
                "like it was designed by a professional graphic designer. Include specific details about:\n"
                "- Layout and composition\n"
                "- Typography style (bold, modern, elegant, etc.)\n"
                "- Color scheme\n"
                "- Visual hierarchy\n"
                "- Background style\n\n"
                "Output ONLY the image generation prompt, nothing else. Make it 3-5 sentences."
            )

            user_prompt = (
                f"Create an image generation prompt for this product ad:\n\n"
                f"{product_info}\n\n"
                f"Original user request: {original_prompt}\n\n"
                f"Write the prompt to generate a COMPLETE advertisement image with all text, "
                f"buttons, and visual elements included in the image."
            )

            result = self.text_gen.generate(system_prompt, user_prompt)
            if result and len(result) > 50:
                print(f"    [SMART-PROMPT] LLM-enhanced rich prompt generated ({len(result)} chars)")
                return result.strip()
        except Exception as e:
            print(f"    [SMART-PROMPT] LLM rich prompt generation failed: {e}, using structured fallback")

        # Fallback: structured prompt with all details
        parts = []
        if extracted.get("brand"):
            parts.append(extracted['brand'])
        if extracted.get("product_name"):
            parts.append(extracted['product_name'])
        if extracted.get("product_type"):
            parts.append(extracted['product_type'])
        for field in detail_fields:
            val = extracted.get(field, "")
            if val:
                parts.append(f"{field.replace('_', ' ')}: {val}")
        if extracted.get("scene_description"):
            parts.append(extracted["scene_description"])

        if parts:
            return f"{' | '.join(parts)}. {original_prompt}"
        return original_prompt

    def merge_fields(self, parsed_data: Dict, additional_fields: Dict[str, str]) -> Dict[str, Any]:
        """Merge additional user-provided fields into parsed data and re-evaluate completeness."""
        extracted = dict(parsed_data.get("extracted", {}))

        # Merge new fields
        for field, value in additional_fields.items():
            if value and str(value).strip():
                extracted[field] = str(value).strip()

        # Re-detect category if changed
        category = extracted.get("category", parsed_data.get("category", "general"))
        cat_key = self._normalize_category(category)
        cat_fields = CATEGORY_FIELDS.get(cat_key, CATEGORY_FIELDS["general"])

        # Recalculate missing
        missing_required = {}
        for field_name, field_desc in cat_fields["required"].items():
            value = extracted.get(field_name, "")
            if not value or str(value).strip().lower() in ("", "unknown", "n/a", "none"):
                missing_required[field_name] = field_desc

        total_required = len(cat_fields["required"])
        filled_required = total_required - len(missing_required)
        completeness = filled_required / total_required if total_required > 0 else 1.0

        # Rebuild rich prompt
        rich_prompt = self._build_rich_prompt(
            parsed_data.get("rich_prompt", ""),
            extracted, cat_key
        )

        return {
            "extracted": extracted,
            "category": cat_key,
            "category_display": cat_fields["display_name"],
            "missing_required": missing_required,
            "all_required": cat_fields["required"],
            "all_optional": cat_fields["optional"],
            "completeness": round(completeness, 2),
            "rich_prompt": rich_prompt,
        }
