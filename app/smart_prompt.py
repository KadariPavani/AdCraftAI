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

    def parse_prompt(self, prompt: str, use_ai: bool = False) -> Dict[str, Any]:
        """Parse a free-text prompt into structured product data.

        Local extraction is always run first (<20ms). If use_ai=True, also
        calls Pollinations AI to refine/correct the extraction (adds ~3-8s).
        Falls back to local-only if AI is unavailable or times out.

        Args:
            prompt: Free-text product description.
            use_ai: If True, also use AI refinement for better accuracy.

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
        t0 = time.time()
        mode = "hybrid" if use_ai else "local"
        print(f"    [SMART-PROMPT] Parsing prompt ({mode}): \"{prompt[:100]}...\"")

        # Step 1: Fast LOCAL extraction — regex + keyword parsing (<20ms)
        extracted = self._extract_fields_local(prompt)

        # Step 2: Optional AI refinement — correct local errors with compact prompt
        if use_ai:
            ai_refined = self._refine_with_ai(prompt, extracted)
            if ai_refined:
                extracted = ai_refined

        # Step 2: Normalize brand name if present
        if extracted.get("brand"):
            extracted["brand"] = BrandMatcher.normalize_brand(extracted["brand"])

        # Step 3: Detect category locally
        category = extracted.get("category", "")
        if not category:
            category = self._detect_category_local(prompt, extracted)
        extracted["category"] = category

        cat_key = self._normalize_category(category)
        cat_fields = CATEGORY_FIELDS.get(cat_key, CATEGORY_FIELDS["general"])

        # Step 4: Check brand against dataset (local, instant)
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

            # Use dataset category if local detection was too generic
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

        # Step 7: Build rich prompt locally
        rich_prompt = self._build_rich_prompt_local(prompt, extracted, cat_key)

        elapsed_ms = (time.time() - t0) * 1000

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
        print(f"    [SMART-PROMPT] Total parse time: {elapsed_ms:.1f}ms")
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

    # ----- Known brand names with category hints for local detection -----
    # Maps brand name -> default category (used when no product type is detected)
    BRAND_CATEGORY_MAP = {
        # Footwear
        "Nike": "footwear", "Adidas": "footwear", "Puma": "footwear",
        "Reebok": "footwear", "New Balance": "footwear", "Skechers": "footwear",
        "Fila": "footwear", "Under Armour": "footwear", "Asics": "footwear",
        "Converse": "footwear", "Vans": "footwear", "Jordan": "footwear",
        "Bata": "footwear", "Sparx": "footwear", "Campus": "footwear",
        "Woodland": "footwear", "Red Chief": "footwear", "Crocs": "footwear",
        # Electronics
        "Samsung": "electronics", "Apple": "electronics", "Sony": "electronics",
        "LG": "electronics", "OnePlus": "electronics", "Xiaomi": "electronics",
        "Realme": "electronics", "Oppo": "electronics", "Vivo": "electronics",
        "Google": "electronics", "Microsoft": "electronics", "Dell": "electronics",
        "HP": "electronics", "Lenovo": "electronics", "Asus": "electronics",
        "Acer": "electronics", "Canon": "electronics", "Nikon": "electronics",
        "GoPro": "electronics", "JBL": "electronics", "Bose": "electronics",
        "Sennheiser": "electronics", "Marshall": "electronics", "Motorola": "electronics",
        "Nokia": "electronics", "Nothing": "electronics", "iQOO": "electronics",
        "Poco": "electronics", "Redmi": "electronics", "Mi": "electronics",
        # Clothing
        "Zara": "clothing", "H&M": "clothing", "Levi's": "clothing", "Levis": "clothing", "Levi": "clothing",
        "Raymond": "clothing", "Allen Solly": "clothing", "Peter England": "clothing",
        "Louis Philippe": "clothing", "Monte Carlo": "clothing", "Biba": "clothing",
        "Fabindia": "clothing", "Flying Machine": "clothing", "HRX": "clothing",
        "Tommy Hilfiger": "clothing", "Gucci": "clothing", "Louis Vuitton": "clothing",
        "Prada": "clothing", "Versace": "clothing", "Burberry": "clothing", "Coach": "clothing",
        "US Polo": "clothing", "Wrangler": "clothing", "Lee": "clothing",
        "Jack & Jones": "clothing", "Only": "clothing", "Mango": "clothing",
        # Beverages
        "Coca-Cola": "beverages", "Coca Cola": "beverages", "Pepsi": "beverages",
        "Bisleri": "beverages", "Red Bull": "beverages", "Sprite": "beverages",
        "Fanta": "beverages", "Thums Up": "beverages", "Mountain Dew": "beverages",
        "7Up": "beverages", "Tropicana": "beverages", "Frooti": "beverages",
        "Sting": "beverages", "Monster": "beverages", "Gatorade": "beverages",
        "Paper Boat": "beverages", "Real": "beverages", "Appy Fizz": "beverages",
        # Jewelry & Watches
        "Tanishq": "jewelry", "Kalyan Jewellers": "jewelry", "Kalyan": "jewelry",
        "Malabar Gold and Diamonds": "jewelry", "Malabar Gold": "jewelry",
        "Titan": "jewelry", "Fastrack": "jewelry", "Casio": "jewelry",
        "Rolex": "jewelry", "Ray-Ban": "jewelry", "Oakley": "jewelry",
        "Fossil": "jewelry", "Daniel Wellington": "jewelry", "Joyalukkas": "jewelry",
        "Sonata": "jewelry", "Swarovski": "jewelry", "Pandora": "jewelry",
        # Automotive
        "Mahindra": "automotive", "Toyota": "automotive", "BMW": "automotive",
        "Mercedes": "automotive", "Audi": "automotive", "Hyundai": "automotive",
        "Maruti": "automotive", "Suzuki": "automotive", "Maruti Suzuki": "automotive",
        "Honda": "automotive", "Tata": "automotive", "Kia": "automotive",
        "MG": "automotive", "Volvo": "automotive", "Jeep": "automotive", "Ford": "automotive",
        "Hero": "automotive", "TVS": "automotive", "Royal Enfield": "automotive",
        "Yamaha": "automotive", "KTM": "automotive", "Ducati": "automotive",
        "Bajaj": "automotive", "MRF": "automotive", "CEAT": "automotive",
        "Apollo": "automotive", "Bridgestone": "automotive", "Michelin": "automotive",
        # Food
        "Lays": "food", "Lay's": "food", "Amul": "food", "Maggi": "food",
        "Nestle": "food", "Cadbury": "food", "Parle": "food", "Britannia": "food",
        "Haldiram": "food", "Kurkure": "food", "Oreo": "food", "KitKat": "food",
        "Dairy Milk": "food", "5 Star": "food", "Bournvita": "food",
        "Horlicks": "food", "Kelloggs": "food", "MTR": "food", "Lijjat": "food",
        # Personal Care
        "Dove": "personal_care", "L'Oreal": "personal_care", "Loreal": "personal_care",
        "Nivea": "personal_care", "Lakme": "personal_care", "Maybelline": "personal_care",
        "Garnier": "personal_care", "Pantene": "personal_care",
        "Head & Shoulders": "personal_care", "Colgate": "personal_care",
        "Pepsodent": "personal_care", "Dettol": "personal_care", "Lifebuoy": "personal_care",
        "Patanjali": "personal_care", "Himalaya": "personal_care",
        "Mamaearth": "personal_care", "Biotique": "personal_care",
        "Forest Essentials": "personal_care", "Lux": "personal_care",
        "Sunsilk": "personal_care", "Clinic Plus": "personal_care",
        "Lotus Herbals": "personal_care", "Nykaa": "personal_care",
        # Home Appliances
        "Dyson": "home_appliances", "Philips": "home_appliances", "Bosch": "home_appliances",
        "Whirlpool": "home_appliances", "Godrej": "home_appliances",
        "Havells": "home_appliances", "IFB": "home_appliances",
        "Voltas": "home_appliances", "Blue Star": "home_appliances",
        "Kent": "home_appliances", "Eureka Forbes": "home_appliances",
        "Crompton": "home_appliances", "Prestige": "home_appliances",
        "Bajaj Electricals": "home_appliances", "Orient": "home_appliances",
        # Others
        "Ikea": "general", "Urban Ladder": "general", "Pepperfry": "general",
        "Sleepwell": "general", "Duroflex": "general",
        "Jio": "general", "Airtel": "general", "Vodafone": "general", "BSNL": "general",
        "Amazon": "general", "Flipkart": "general", "Myntra": "general", "Ajio": "general",
    }

    # Pre-sorted for matching (longest first to match "Coca-Cola" before "Coca")
    KNOWN_BRANDS_SORTED = sorted(BRAND_CATEGORY_MAP.keys(), key=lambda b: -len(b))

    # Conversational noise prefixes to strip before extraction
    _NOISE_PATTERNS = re.compile(
        r'^(?:i\s+want\s+(?:to\s+)?|please\s+|can\s+you\s+|'
        r'(?:create|generate|make|design|build|show)\s+(?:me\s+)?'
        r'(?:a(?:n)?\s+)?(?:ad|advertisement|poster|banner|pamphlet|flyer|creative|image|photo)?\s*'
        r'(?:for|of|about|featuring|with|showing)?\s*)',
        re.I
    )

    # ── AI Refinement — compact prompt for fast correction ──

    _AI_SYSTEM = (
        "Extract product data from the user's ad description as JSON. "
        "Fields: brand, product_name, product_type, category (footwear/electronics/"
        "clothing/beverages/jewelry/automotive/food/personal_care/home_appliances/general), "
        "target_audience, size_range, material, color, price, key_features, occasion, "
        "pack_size, flavor, weight, warranty, scene_description. "
        "Return ONLY JSON. Use null for absent fields. Be precise, don't fabricate."
    )

    def _refine_with_ai(self, prompt: str, local_extracted: Dict) -> Optional[Dict[str, str]]:
        """Use Pollinations AI to accurately extract fields — optimized for speed.

        Makes a direct API call with:
        - Compact system prompt (minimal tokens)
        - max_tokens=400 (limits response length = faster generation)
        - temperature=0 (deterministic, no sampling overhead)
        - 15s timeout (fail fast, fall back to local)

        Returns refined dict or None on failure (caller keeps local extraction).
        """
        import requests as _req

        try:
            t0 = time.time()
            payload = {
                "messages": [
                    {"role": "system", "content": self._AI_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "model": "openai",
                "seed": int(time.time()) % 10000,
                "jsonMode": True,
                "max_tokens": 400,
                "temperature": 0,
            }
            print(f"    [SMART-PROMPT] AI refinement request (max_tokens=400, temp=0)")
            resp = _req.post("https://text.pollinations.ai/", json=payload, timeout=15)
            elapsed = time.time() - t0
            print(f"    [SMART-PROMPT] AI refinement took {elapsed:.1f}s (status={resp.status_code})")

            if resp.status_code != 200 or len(resp.text) < 10:
                print(f"    [SMART-PROMPT] AI refinement failed, keeping local extraction")
                return None

            text = resp.text.strip()
            if text.startswith("```"):
                text = re.sub(r'^```(?:json)?\s*', '', text)
                text = re.sub(r'\s*```$', '', text)
            result = json.loads(text)

            if not isinstance(result, dict):
                return None

            # Build merged result: AI fields override local, but keep local extras
            refined = {}
            for key, val in result.items():
                if val is not None and str(val).strip() and str(val).strip().lower() not in ("null", "none", "n/a", "not specified", "not mentioned"):
                    refined[key] = str(val).strip()

            # Supplement with local extraction for fields AI missed
            for key, val in local_extracted.items():
                if key not in refined or not refined[key]:
                    refined[key] = val

            # For precise numeric fields, prefer local regex (better formatting)
            for key in ("price", "pack_size", "size_range", "weight"):
                if local_extracted.get(key) and refined.get(key):
                    local_val = local_extracted[key]
                    if any(c in local_val for c in "₹$€£") or re.search(r'\d+\s*(?:ml|l|g|kg|mm|cm|inch)', local_val, re.I):
                        refined[key] = local_val

            print(f"    [SMART-PROMPT] AI refined {len(refined)} fields: {list(refined.keys())}")
            return refined

        except _req.Timeout:
            elapsed = time.time() - t0
            print(f"    [SMART-PROMPT] AI refinement TIMEOUT ({elapsed:.1f}s), keeping local extraction")
            return None
        except (json.JSONDecodeError, Exception) as e:
            print(f"    [SMART-PROMPT] AI refinement error: {e}, keeping local extraction")
            return None

    def _extract_fields_regex(self, prompt: str) -> Dict[str, str]:
        """Basic regex extraction — used only as inner helper."""
        extracted = {}
        text = prompt.lower()

        price_match = re.search(r'(?:rs\.?|inr|usd|\$|price[:\s]+)\s*[\d,]+(?:\.\d{2})?', text, re.I)
        if price_match:
            extracted["price"] = price_match.group().strip()

        size_match = re.search(r'(?:size[s]?[:\s]+)([\w\s,/-]+?)(?:\.|,\s*\w|\n|$)', text, re.I)
        if size_match:
            extracted["size_range"] = size_match.group(1).strip()

        weight_match = re.search(r'\b(\d+(?:\.\d+)?\s*(?:kg|g|gm|ml|l|liter|oz|lb))\b', text, re.I)
        if weight_match:
            extracted["pack_size"] = weight_match.group(1).strip()

        return extracted

    def _clean_prompt(self, prompt: str) -> str:
        """Strip conversational noise while preserving product info.

        Handles: 'create an ad for Nike Air Max', 'I want to generate a poster
        of Samsung Galaxy', 'please make me an advertisement featuring Coca-Cola'
        """
        cleaned = self._NOISE_PATTERNS.sub('', prompt).strip()
        # If we stripped too much (e.g., only noise was entered), keep original
        if len(cleaned) < 3:
            return prompt.strip()
        return cleaned

    def _extract_fields_local(self, prompt: str) -> Dict[str, str]:
        """Full local field extraction using regex + keyword matching.

        Zero network calls. Uses prompt cleaning, brand-category knowledge,
        longest-match-first product types, possessive handling, and multi-strategy
        product name extraction. Runs in <10ms.
        """
        extracted = {}
        # Clean conversational noise but keep original for scene/feature extraction
        original_text = prompt
        text = self._clean_prompt(prompt)
        text_lower = text.lower()
        # Also handle possessives: "Nike's" -> match "Nike"
        text_lower_noposs = re.sub(r"['']s\b", "", text_lower)

        # ── EARLY: Extract scene description and create product-only text ──
        # This prevents scene elements (e.g., "saree" in "bride in red silk saree")
        # from being wrongly detected as the product type.
        scene_text = ""
        product_text_lower = text_lower  # text with scene masked out
        scene_patterns_early = [
            r'(?:show(?:ing)?|depict(?:ing)?|featuring?|scene[:\s]+|setting[:\s]+|background[:\s]+|backdrop[:\s]+)(.*?)$',
            r'(?:at\s+(?:a\s+)?)((?:luxury|premium|modern|elegant|beautiful|outdoor|indoor|urban|rustic|tropical|mountain|city|beach|studio|nature|street|rooftop|garden)[\w\s,]+?)$',
            r'(?:with\s+(?:a\s+)?)((?:gradient|bokeh|blurred|neon|sunset|sunrise|golden\s*hour|dramatic|cinematic|minimalist|dark|bright|colorful)[\w\s]+?(?:background|backdrop|lighting|setting))$',
        ]
        for pat in scene_patterns_early:
            m = re.search(pat, original_text, re.I)
            if m:
                scene_text = m.group(0).strip().rstrip(".,")
                if len(scene_text) > 15:
                    extracted["scene_description"] = scene_text[:200]
                    # Mask scene from product detection text
                    product_text_lower = text_lower[:m.start()].strip().lower() if m.start() > 0 else text_lower
                    print(f"    [SMART-PROMPT] Scene extracted early, masked from product detection")
                break

        # ── Brand detection ──
        # 1) Check known brands list — longest first, ALWAYS use word boundaries
        #    to prevent "casio" matching inside "occasion", "hp" in "shampoo", etc.
        for brand in self.KNOWN_BRANDS_SORTED:
            brand_lower = brand.lower()
            if re.search(r'\b' + re.escape(brand_lower) + r'\b', text_lower_noposs):
                extracted["brand"] = brand
                break

        # 2) If pipeline has brand matcher, also try fuzzy match
        if not extracted.get("brand") and self.pipeline:
            try:
                bm = self.pipeline.brand_matcher.match(text)
                if bm.matched_brand and bm.confidence > 0.6:
                    extracted["brand"] = bm.matched_brand.replace("_", " ")
            except Exception:
                pass

        # 3) Heuristic: first capitalized/CamelCase word as brand — unknown brands
        #    Only take the FIRST word to avoid absorbing model names
        #    (e.g., "TerraMotors EcoRide" → brand="TerraMotors", not "TerraMotors EcoRide")
        if not extracted.get("brand"):
            m = re.search(r'\b([A-Z][a-zA-Z&\'-]{1,20})\b', text)
            if m:
                candidate = m.group().strip()
                skip = {"The", "For", "And", "With", "New", "Best", "Top", "Buy",
                        "Available", "Show", "Create", "Make", "Get", "All", "Our",
                        "Please", "Want", "Need", "Generate", "Design", "Professional",
                        "Premium", "Modern", "Luxury", "Bold", "Dynamic", "Classic",
                        "Pack", "Set", "Box", "Fresh", "Pure", "Natural", "Super",
                        "Ultra", "Pro", "Max", "Plus", "Mini", "Lite", "Smart"}
                if candidate not in skip and len(candidate) > 1:
                    extracted["brand"] = candidate

        # ── Use brand knowledge for category hint ──
        brand_category_hint = ""
        if extracted.get("brand"):
            brand_category_hint = self.BRAND_CATEGORY_MAP.get(extracted["brand"], "")

        # ── Product Type (what kind of product) — LONGEST MATCH FIRST ──
        product_types = {
            "footwear": [
                "running shoes", "sports shoes", "casual shoes", "formal shoes",
                "flip flops", "high heels", "ankle boots",
                "sneakers", "sandals", "boots", "loafers", "heels",
                "slippers", "shoes", "trainers", "cleats", "moccasins",
            ],
            "electronics": [
                "mobile phone", "smart phone", "smartphone", "smart watch", "smartwatch",
                "power bank", "wireless earbuds", "bluetooth speaker",
                "laptop", "tablet", "earbuds", "headphones", "speaker",
                "television", "tv", "monitor", "camera", "charger",
                "earphone", "soundbar", "projector", "drone", "router",
            ],
            "clothing": [
                "t-shirt", "tshirt", "polo shirt", "kurta set", "co-ord set",
                "shirt", "jeans", "trousers", "pants", "dress",
                "jacket", "hoodie", "sweater", "kurta", "saree", "sari", "lehenga",
                "blazer", "suit", "shorts", "skirt", "top", "blouse",
                "tracksuit", "joggers", "sweatshirt", "co-ord", "palazzo",
                "romper", "onesie", "bodysuit", "dungaree", "jumpsuit",
                "cardigan", "poncho", "shrug", "cape", "overcoat", "parka",
            ],
            "beverages": [
                "mineral water", "soft drink", "energy drink", "cold drink",
                "sparkling water", "carbonated water", "iced tea", "iced coffee",
                "juice", "cola", "soda", "coffee", "tea", "milkshake", "smoothie",
                "lemonade", "lassi", "buttermilk", "sharbat", "drink", "water",
            ],
            "jewelry": [
                "gold necklace", "gold bangles", "diamond ring", "diamond necklace",
                "bangles set", "bangles",
                "necklace", "ring", "bracelet", "earrings", "pendant",
                "chain", "mangalsutra", "anklet", "brooch",
                "watch", "watches", "wristwatch",
                "nose ring", "toe ring", "cufflinks", "tiara",
            ],
            "automotive": [
                "sports car", "electric car", "electric vehicle",
                "suv", "sedan", "hatchback", "motorcycle", "scooter", "bike",
                "ev", "truck", "van", "mpv", "crossover", "coupe",
                "car", "auto", "vehicle",
            ],
            "food": [
                "ice cream", "instant noodles", "dark chocolate", "milk chocolate",
                "chips", "chocolate", "noodles", "biscuits", "cookies", "cheese",
                "butter", "candy", "snacks", "cereal", "bread", "cake",
                "namkeen", "papad", "pickle", "jam", "honey", "ghee",
            ],
            "personal_care": [
                "face cream", "face wash", "body lotion", "hair oil",
                "body wash", "hand cream", "eye cream", "lip balm",
                "shampoo", "conditioner", "moisturizer", "moisturiser",
                "lipstick", "foundation", "perfume", "cologne", "deodorant",
                "sunscreen", "serum", "soap",
                "toothpaste", "mouthwash", "razor", "trimmer",
            ],
            "home_appliances": [
                "washing machine", "vacuum cleaner", "mixer grinder",
                "air conditioner", "water purifier", "air purifier",
                "refrigerator", "microwave", "oven",
                "dishwasher", "iron", "fan", "induction", "geyser", "heater",
                "chimney", "cooler", "freezer",
            ],
        }
        # Build a flat list of (product_type, category) sorted longest first globally.
        # This ensures "water purifier" (home_appliances) is checked before "water" (beverages).
        all_product_types: List[Tuple[str, str]] = []
        for cat, types in product_types.items():
            for pt in types:
                all_product_types.append((pt, cat))
        all_product_types.sort(key=lambda x: -len(x[0]))

        for pt, cat in all_product_types:
            # Always use word boundary to prevent substring false positives
            # Use product_text_lower (scene-masked) to avoid scene elements as products
            if re.search(r'\b' + re.escape(pt) + r's?\b', product_text_lower):
                extracted["product_type"] = pt.title()
                if not extracted.get("category"):
                    extracted["category"] = cat
                break

        # If no product type found but brand has category hint, use it
        if not extracted.get("category") and brand_category_hint:
            extracted["category"] = brand_category_hint
            print(f"    [SMART-PROMPT] Category from brand knowledge: {brand_category_hint}")

        # ── Product Name — multi-strategy extraction ──
        if not extracted.get("product_name"):
            # Strategy 1: Quoted product names — highest confidence
            quoted = re.search(r'["\u201c\u201d\'`]([^"\'`\u201c\u201d]{3,50})["\u201c\u201d\'`]', text)
            if quoted:
                extracted["product_name"] = quoted.group(1).strip()

            # Strategy 2: Explicit "product/model: XYZ" patterns
            if not extracted.get("product_name"):
                name_pat = re.search(
                    r'(?:product|model|name|series|variant|edition)[:\s]+([A-Za-z0-9][\w\s.-]{1,40}?)(?:\.|,|\n|$|\s+(?:for|with|in|is|at|price))',
                    text, re.I)
                if name_pat:
                    extracted["product_name"] = name_pat.group(1).strip()

            # Strategy 3: Brand followed by model/product name
            if not extracted.get("product_name") and extracted.get("brand"):
                brand_lower = extracted["brand"].lower()
                # Handle possessives: "Nike's Air Max" -> find after "nike's"
                brand_poss = re.compile(
                    re.escape(brand_lower) + r"(?:[''`]s)?\s*",
                    re.I
                )
                m = brand_poss.search(text_lower_noposs if "'" not in text_lower else text_lower)
                if not m:
                    m = brand_poss.search(text_lower)
                if m:
                    after = text[m.end():].strip()
                    # Take the next 1-5 words as product name, stop at stop words
                    chunks = re.split(r'[,.]', after)[0].strip().split()
                    if chunks:
                        name_words = []
                        stop_words = {
                            "for", "with", "in", "is", "are", "the", "and", "or",
                            "a", "an", "men", "men's", "mens", "women", "women's",
                            "womens", "kids", "kid's", "unisex", "boys", "girls",
                            "at", "price", "priced", "available", "featuring", "new",
                            "on", "from", "by",
                        }
                        pt_lower = (extracted.get("product_type") or "").lower()
                        pt_last_word = pt_lower.split()[-1] if pt_lower else ""
                        passed_product_type = False
                        for w in chunks[:8]:
                            wl = w.lower().rstrip("'s")
                            if wl in stop_words:
                                break
                            # Skip single lowercase-char leftovers (e.g., "s" from "Levi's")
                            if len(w.strip()) == 1 and w.strip().islower():
                                continue
                            name_words.append(w)
                            # Once we've included the product type, allow 1 more word then stop
                            # e.g., "Air Max 90 running shoes" -> stop after "shoes"
                            # e.g., "collection 22K gold bangles set" -> include "set" then stop
                            if pt_last_word and wl == pt_last_word:
                                passed_product_type = True
                            elif passed_product_type:
                                break
                        # Cap at 6 words max to avoid very long product names
                        if name_words:
                            name = " ".join(name_words[:6])
                            # Don't use product type alone as product name
                            if name.lower() != pt_lower:
                                extracted["product_name"] = name

        # Fallback: if no product_name but we have product_type, use it as product_name
        if not extracted.get("product_name") and extracted.get("product_type"):
            extracted["product_name"] = extracted["product_type"]

        # ── Price ──
        price_patterns = [
            r'(?:Rs\.?|INR|₹)\s*[\d,]+(?:\.\d{1,2})?(?:\s*[-–to]+\s*(?:Rs\.?|INR|₹)?\s*[\d,]+(?:\.\d{1,2})?)?',
            r'\$\s*[\d,]+(?:\.\d{1,2})?(?:\s*[-–to]+\s*\$?\s*[\d,]+(?:\.\d{1,2})?)?',
            r'(?:price[d]?\s*(?:at|is|:)?\s*(?:Rs\.?|INR|₹|\$)?\s*)[\d,]+(?:\.\d{1,2})?',
            r'(?:MRP|mrp)[:\s]*(?:Rs\.?|₹|\$)?\s*[\d,]+(?:\.\d{1,2})?',
            r'(?:under|below|above|around|about|approximately|approx)\s*(?:Rs\.?|INR|₹|\$)\s*[\d,]+',
        ]
        for pat in price_patterns:
            m = re.search(pat, text, re.I)
            if m:
                extracted["price"] = m.group().strip().rstrip(",.")
                break

        # ── Size / Size Range ──
        size_patterns = [
            # "UK 6-12", "US 7 to 13", "EU 38-44"
            r'\b((?:UK|US|EU)\s*\d+\s*[-–to]+\s*\d+)\b',
            # "sizes: S/M/L/XL" or "S, M, L, XL"
            r'\b((?:S|M|L|XL|XXL)(?:\s*[/,]\s*(?:S|M|L|XL|XXL|XXXL))+)\b',
            # "sizes 28-36" or "size 28 to 36" (only numeric ranges)
            r'\bsizes?\s+([\d]+\s*[-–to]+\s*[\d]+)\b',
            # "sizes: UK 6-12, S/M/L" (explicit label + value)
            r'(?:sizes?\s*(?:range)?[:\s]+)((?:UK|US|EU)\s*\d[\w\s,/\-–]+?)(?:\.|,\s*[a-z]|\n|$)',
            # "size 8" or "size 42"
            r'\bsize\s+(\d{2,})\b',
            # Bangle/ring sizes like "2.4 to 2.8" after "sizes"
            r'\bsizes?\s+(\d+(?:\.\d+)?\s*(?:to|-|–)\s*\d+(?:\.\d+)?)\b',
        ]
        for pat in size_patterns:
            m = re.search(pat, product_text_lower, re.I)
            if m:
                val = m.group(1).strip() if m.lastindex else m.group().strip()
                if len(val) >= 2:  # avoid single-char garbage
                    extracted["size_range"] = val
                    break

        # ── Weight / Pack Size / Volume ──
        vol_m = re.search(r'\b(\d+(?:\.\d+)?\s*(?:kg|g|gm|grams?|ml|l|liter|litre|oz|lb|lbs|cc|cl))\b', text_lower)
        if vol_m:
            extracted["pack_size"] = vol_m.group(1).strip()

        pack_m = re.search(r'\b(?:pack|set|box|combo)\s*(?:of\s*)?\s*(\d+)\b', text_lower)
        if pack_m and not extracted.get("pack_size"):
            extracted["pack_size"] = f"Pack of {pack_m.group(1)}"

        # ── Weight (standalone) ──
        weight_m = re.search(r'\bweight\s*[:\s]*(\d+(?:\.\d+)?\s*(?:kg|g|gm|grams?|oz|lb|lbs|carat))\b', text_lower)
        if weight_m:
            extracted["weight"] = weight_m.group(1).strip()
        elif not extracted.get("weight"):
            wt = re.search(r'\b(\d+(?:\.\d+)?\s*(?:carat|grams?))\b', text_lower)
            if wt:
                extracted["weight"] = wt.group(1).strip()

        # ── Color — multi-word colors first ──
        multi_word_colors = [
            "rose gold", "midnight blue", "sky blue", "baby blue", "royal blue",
            "forest green", "lime green", "olive green", "sage green",
            "hot pink", "baby pink", "dusty pink", "blush pink",
            "deep red", "cherry red", "wine red", "burgundy red",
            "pearl white", "off white", "snow white",
            "matte black", "jet black", "charcoal grey", "space grey",
            "ocean blue", "coral orange", "sunset orange",
            "champagne gold", "brushed silver",
        ]
        single_colors = [
            "black", "white", "red", "blue", "green", "yellow", "orange", "pink",
            "purple", "violet", "grey", "gray", "silver", "gold", "golden", "brown",
            "beige", "navy", "teal", "cyan", "magenta", "maroon", "cream", "ivory",
            "titanium", "coral", "lavender", "turquoise", "burgundy", "copper",
            "bronze", "peach", "khaki", "indigo", "aqua", "charcoal",
        ]
        found_colors = []
        for c in multi_word_colors:
            if re.search(r'\b' + re.escape(c) + r'\b', product_text_lower):
                found_colors.append(c.title())
        for c in single_colors:
            if re.search(r'\b' + re.escape(c) + r'\b', product_text_lower):
                # Avoid duplicates from multi-word matches
                c_title = c.title()
                if not any(c_title in fc for fc in found_colors):
                    found_colors.append(c_title)
        colorway_m = re.search(r'\b([\w]+(?:\s*/\s*[\w]+)+)\s*(?:color(?:way)?|colour)', product_text_lower)
        if colorway_m:
            found_colors.append(colorway_m.group(1).strip())
        if found_colors:
            extracted["color"] = ", ".join(dict.fromkeys(found_colors))

        # ── Material — multi-word first ──
        multi_word_materials = [
            "organic cotton", "stainless steel", "sterling silver", "carbon fiber",
            "carbon fibre", "22k gold", "24k gold", "18k gold", "14k gold",
            "faux leather", "vegan leather", "patent leather", "full grain leather",
            "memory foam", "tempered glass", "gorilla glass",
        ]
        single_materials = [
            "leather", "mesh", "synthetic", "cotton", "polyester", "silk", "wool",
            "nylon", "rubber", "suede", "canvas", "denim", "linen", "satin",
            "velvet", "titanium", "platinum", "gold", "diamond", "ceramic",
            "glass", "aluminum", "aluminium", "plastic", "wood",
            "bamboo", "jute", "gore-tex", "fleece", "lycra", "spandex",
            "rayon", "cashmere", "chiffon", "georgette", "crepe", "tweed",
        ]
        found_mats = []
        for mat in multi_word_materials:
            if re.search(r'\b' + re.escape(mat) + r'\b', product_text_lower):
                found_mats.append(mat.title())
        for mat in single_materials:
            if re.search(r'\b' + re.escape(mat) + r'\b', product_text_lower):
                mat_title = mat.title()
                if not any(mat_title in fm for fm in found_mats):
                    found_mats.append(mat_title)
        if found_mats:
            extracted["material"] = ", ".join(dict.fromkeys(found_mats))

        # ── Target Audience — expanded patterns ──
        audience_patterns = [
            (r'\bfor\s+(men|women|kids|children|boys|girls|unisex|adults|teens|teenagers|babies|toddlers|couples|ladies|gents|youth|seniors|professionals|athletes|students)\b', 1),
            (r"\b(men'?s|women'?s|kid'?s|boy'?s|girl'?s|unisex|ladies'?|gentlemen'?s?)\b", 1),
            (r'\b(men|women|male|female)\s+(?:running|casual|sports|formal|daily|fashion)', 1),
            (r'\b(young|teen|adult|senior|professional|sporty|fitness)\s+(?:men|women|people|adults|users)\b', 0),
            (r'\bage\s*(?:group)?\s*[:\s]*(\d+\s*[-–to]+\s*\d+)\b', None),  # age range
        ]
        if not extracted.get("target_audience"):
            for pat, grp in audience_patterns:
                m = re.search(pat, text_lower)
                if m:
                    if grp is None:
                        extracted["target_audience"] = f"Age {m.group(1).strip()}"
                    else:
                        val = m.group(grp).strip()
                        # Normalize possessives and variants
                        val = re.sub(r"[''`]s?$", "", val).title()
                        if val == "Male":
                            val = "Men"
                        elif val == "Female":
                            val = "Women"
                        extracted["target_audience"] = val
                    break

        # ── Key Features — expanded extraction ──
        feature_indicators = [
            r'(?:features?|specs?|specifications?|highlights?|benefits?)[:\s]+(.*?)(?:\.|$)',
            r'(?:with|includes?|equipped with|comes with|offers?|provides?|has)\s+([\w\s,]+(?:,\s*[\w\s]+){1,})',
            r'(?:known for|famous for|best for)\s+(.*?)(?:\.|,\s*(?:and|the)|$)',
        ]
        for pat in feature_indicators:
            m = re.search(pat, original_text, re.I)
            if m:
                feat = m.group(1).strip()[:200]
                if len(feat) > 5:  # Avoid tiny matches
                    extracted["key_features"] = feat
                    break

        # Also collect standalone tech specs as features
        tech_specs = []
        spec_patterns = [
            r'\b(\d+\s*MP\s*camera)\b', r'\b(\d+\s*mAh\s*battery)\b',
            r'\b(\d+\s*GB?\s*(?:RAM|storage|ROM|SSD|HDD))\b',
            r'\b(\d+(?:\.\d+)?\s*inch\s*(?:display|screen)?)\b',
            r'\b(Snapdragon\s*\d+[^\s,]*)\b', r'\b(MediaTek\s*\w+\s*\d+[^\s,]*)\b',
            r'\b(Exynos\s*\d+[^\s,]*)\b', r'\b(Apple\s*[AM]\d+[^\s,]*)\b',
            r'\b(AMOLED|Super\s*AMOLED|OLED|IPS|LCD|Retina|Mini[- ]?LED)\s*(?:display)?\b',
            r'\b(4[GX]|5G|WiFi\s*\d*[a-z]?|Bluetooth\s*[\d.]*|NFC|USB[-\s]?C|Thunderbolt\s*\d?)\b',
            r'\b(waterproof|water[- ]?resistant|IP\d{2}|dust[- ]?proof)\b',
            r'\b(BIS\s*hallmark(?:ed)?|GIA\s*certified|ISO\s*certified|CE\s*certified)\b',
            r'\b(lightweight|breathable|cushion(?:ing|ed)?|ergonomic|anti[- ]?slip|shock[- ]?proof)\b',
            r'\b(fast[- ]?charging|wireless[- ]?charging|quick[- ]?charge|turbo[- ]?charge)\b',
            r'\b(\d+(?:\.\d+)?\s*(?:HP|hp|BHP|bhp|cc|PS)\b)',
            r'\b(\d+\s*(?:km/?l|kmpl|miles?\s*per\s*gallon))\b',
            r'\b(4WD|AWD|4x4|front[- ]?wheel[- ]?drive|rear[- ]?wheel[- ]?drive)\b',
            r'\b(turbo(?:charged)?|supercharged|hybrid|electric[- ]?motor)\b',
            r'\b(SPF\s*\d+|paraben[- ]?free|sulfate[- ]?free|cruelty[- ]?free|vegan|organic|natural)\b',
            r'\b(anti[- ]?aging|anti[- ]?wrinkle|anti[- ]?dandruff|anti[- ]?acne|whitening|brightening)\b',
        ]
        for pat in spec_patterns:
            m = re.search(pat, text, re.I)
            if m:
                spec = m.group(1).strip() if m.lastindex else m.group().strip()
                if spec not in tech_specs:
                    tech_specs.append(spec)
        if tech_specs and not extracted.get("key_features"):
            extracted["key_features"] = ", ".join(tech_specs)
        elif tech_specs and extracted.get("key_features"):
            # Append only specs not already mentioned
            existing = extracted["key_features"].lower()
            new_specs = [s for s in tech_specs if s.lower() not in existing]
            if new_specs:
                extracted["key_features"] += ", " + ", ".join(new_specs)

        # ── Flavor ──
        flavor_m = re.search(r'\b(?:flavor|flavour|variant|taste)[:\s]+([\w\s]+?)(?:\.|,|\n|$)', text_lower)
        if flavor_m:
            extracted["flavor"] = flavor_m.group(1).strip().title()
        else:
            # Only auto-detect flavors for food/beverage categories
            detected_cat = extracted.get("category", "")
            if detected_cat in ("food", "beverages", "personal_care", ""):
                flavors = [
                    "original", "lime", "lemon", "mango", "strawberry", "vanilla",
                    "chocolate", "mint", "peppermint", "orange", "grape", "apple",
                    "cherry", "masala", "classic", "spicy", "tangy", "tomato",
                    "cheese", "peri peri", "barbecue", "bbq", "salted", "caramel",
                    "blueberry", "raspberry", "tropical", "mixed fruit", "pineapple",
                    "butter", "garlic", "honey", "ginger", "cardamom", "saffron",
                ]
                flavors.sort(key=lambda x: -len(x))
                for f in flavors:
                    if re.search(r'\b' + re.escape(f) + r'\b', text_lower):
                        extracted["flavor"] = f.title()
                        break

        # ── Occasion — expanded ──
        occasions = [
            "date night", "daily wear", "everyday wear",
            "wedding", "festive", "festival", "party", "casual", "formal",
            "office", "sports", "running", "gym", "workout", "training",
            "outdoor", "travel", "beach", "bridal", "engagement",
            "anniversary", "birthday", "christmas", "diwali", "eid",
            "valentines", "housewarming", "graduation", "interview",
            "trekking", "hiking", "camping", "yoga", "meditation",
        ]
        occasions.sort(key=lambda x: -len(x))
        for occ in occasions:
            if occ in text_lower:
                extracted["occasion"] = occ.title()
                break

        # ── Warranty ──
        warranty_m = re.search(r'\b(\d+\s*(?:year|yr|month|mon)s?\s*warranty)\b', text_lower)
        if warranty_m:
            extracted["warranty"] = warranty_m.group(1).strip().title()

        # ── Manufacturer ──
        mfr_m = re.search(r'(?:manufactured?\s*by|made\s*by|produced\s*by|from)\s+([A-Z][\w\s&]+?)(?:\.|,|\n|$)', original_text)
        if mfr_m:
            extracted["manufacturer"] = mfr_m.group(1).strip()

        # ── Scene Description — fallback if early extraction missed it ──
        if not extracted.get("scene_description"):
            scene_patterns = [
                r'(?:show(?:ing)?|depict(?:ing)?|featuring?|scene[:\s]+|setting[:\s]+|background[:\s]+|backdrop[:\s]+)(.*?)(?:\.|$)',
                r'(?:at\s+(?:a\s+)?|in\s+(?:a\s+)?)((?:luxury|premium|modern|elegant|beautiful|outdoor|indoor|urban|rustic|tropical|mountain|city|beach|studio|nature|street|rooftop|garden)[\w\s,]+?)(?:\.|,\s*\w|\n|$)',
                r'(?:with\s+(?:a\s+)?)((?:gradient|bokeh|blurred|neon|sunset|sunrise|golden\s*hour|dramatic|cinematic|minimalist|dark|bright|colorful)[\w\s]+?(?:background|backdrop|lighting|setting))(?:\.|,|\n|$)',
            ]
            for pat in scene_patterns:
                m = re.search(pat, original_text, re.I)
                if m:
                    scene = m.group(0).strip().rstrip(".,")
                    if len(scene) > 15:
                        extracted["scene_description"] = scene[:200]
                        break

        print(f"    [SMART-PROMPT] Local extraction: {len(extracted)} fields: {list(extracted.keys())}")
        return extracted

    def _detect_category_local(self, prompt: str, extracted: Dict) -> str:
        """Detect product category locally using keyword matching + brand knowledge.

        Priority: extracted product_type > keyword scoring (weighted by specificity)
        > brand-category map. Instant, no network.
        """
        # Check extracted product_type first (most reliable signal)
        product_type = (extracted.get("product_type") or "").lower().replace(" ", "_")
        if product_type:
            cat = self._normalize_category(product_type)
            if cat != "general":
                return cat

        # Scan prompt for category keywords — weighted by specificity
        text = prompt.lower()
        keyword_scores: Dict[str, float] = {}

        # (keyword, weight) — multi-word/specific keywords score higher
        category_keywords = {
            "footwear": [
                ("running shoe", 3), ("sports shoe", 3), ("casual shoe", 3),
                ("shoe", 2), ("shoes", 2), ("sneaker", 2), ("boot", 1.5),
                ("sandal", 2), ("slipper", 2), ("footwear", 3), ("loafer", 2), ("heel", 1),
                ("trainer", 1.5), ("cleats", 2),
            ],
            "electronics": [
                ("smartphone", 3), ("mobile phone", 3), ("smart watch", 3),
                ("phone", 2), ("laptop", 3), ("tablet", 2), ("earbuds", 3),
                ("headphone", 2), ("tv", 1.5), ("television", 2), ("camera", 2),
                ("smartwatch", 3), ("speaker", 1.5), ("charger", 2), ("monitor", 2),
                ("powerbank", 3), ("soundbar", 3), ("projector", 2),
            ],
            "clothing": [
                ("t-shirt", 3), ("tshirt", 3), ("polo shirt", 3),
                ("shirt", 2), ("jeans", 2), ("dress", 1.5), ("jacket", 2),
                ("trouser", 2), ("kurta", 3), ("saree", 3), ("apparel", 2),
                ("clothing", 2), ("sweater", 2), ("hoodie", 2), ("blazer", 2),
            ],
            "beverages": [
                ("energy drink", 3), ("soft drink", 3), ("cold drink", 3),
                ("mineral water", 3), ("sparkling water", 3),
                ("drink", 1), ("juice", 2), ("soda", 2), ("coffee", 1.5),
                ("tea", 1), ("cola", 2), ("beverage", 2), ("milkshake", 3),
                ("water", 0.5),  # low weight — too generic
            ],
            "jewelry": [
                ("gold necklace", 3), ("diamond ring", 3),
                ("necklace", 2), ("ring", 1.5), ("bangle", 2), ("bracelet", 2),
                ("earring", 2), ("pendant", 2), ("chain", 1), ("watch", 1.5),
                ("gold", 1), ("diamond", 1.5), ("platinum", 1), ("jewel", 2),
                ("mangalsutra", 3), ("wristwatch", 3),
            ],
            "automotive": [
                ("sports car", 3), ("electric car", 3), ("electric vehicle", 3),
                ("car", 2), ("bike", 1.5), ("motorcycle", 3), ("scooter", 2),
                ("suv", 3), ("sedan", 3), ("hatchback", 3), ("truck", 2),
                ("vehicle", 1.5), ("tyre", 2), ("tire", 2),
            ],
            "food": [
                ("ice cream", 3), ("instant noodle", 3),
                ("chips", 2), ("chocolate", 2), ("noodle", 2), ("biscuit", 2),
                ("cookie", 2), ("cheese", 1.5), ("butter", 1), ("snack", 2),
                ("cereal", 2), ("bread", 1.5), ("candy", 2), ("cake", 1.5),
            ],
            "personal_care": [
                ("face cream", 3), ("face wash", 3), ("body lotion", 3),
                ("hair oil", 3), ("body wash", 3),
                ("shampoo", 3), ("cream", 1), ("lotion", 2), ("perfume", 3),
                ("soap", 2), ("lipstick", 3), ("foundation", 2), ("skincare", 3),
                ("deodorant", 3), ("sunscreen", 3), ("serum", 2), ("moisturizer", 3),
            ],
            "home_appliances": [
                ("washing machine", 3), ("air conditioner", 3), ("water purifier", 3),
                ("mixer grinder", 3), ("vacuum cleaner", 3),
                ("refrigerator", 3), ("microwave", 3), ("vacuum", 2),
                ("mixer", 1.5), ("oven", 2), ("dishwasher", 3), ("iron", 1),
                ("fan", 1), ("purifier", 2), ("geyser", 3), ("chimney", 2),
            ],
        }

        for cat, kw_weights in category_keywords.items():
            for kw, weight in kw_weights:
                if kw in text:
                    keyword_scores[cat] = keyword_scores.get(cat, 0) + weight

        if keyword_scores:
            best = max(keyword_scores, key=keyword_scores.get)
            print(f"    [SMART-PROMPT] Local category detection: {best} (score={keyword_scores[best]:.1f})")
            return best

        # Fallback: use brand-category map
        brand = extracted.get("brand", "")
        if brand:
            cat = self.BRAND_CATEGORY_MAP.get(brand, "")
            if cat:
                print(f"    [SMART-PROMPT] Category from brand map: {cat}")
                return cat

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

    def _build_rich_prompt_local(self, original_prompt: str, extracted: Dict, category: str) -> str:
        """Build a rich ad image generation prompt locally — no API call.

        Used as fallback when the combined AI call already provides a rich_prompt,
        or when the AI call fails entirely.
        """
        detail_fields = ["material", "color", "key_features", "target_audience",
                         "occasion", "pack_size", "size_range", "flavor", "price"]

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

        # Build a structured prompt that works well with image generation
        product_desc = " | ".join(parts) if parts else original_prompt

        # Category-aware style hints
        style_hints = {
            "footwear": "dynamic sports photography, clean studio backdrop",
            "electronics": "sleek tech product shot, gradient background, modern",
            "clothing": "fashion editorial style, clean backdrop, lifestyle",
            "beverages": "refreshing product shot, condensation, vibrant colors",
            "jewelry": "luxury studio lighting, velvet backdrop, elegant",
            "automotive": "dramatic angle, open road, cinematic lighting",
            "food": "appetizing food photography, warm lighting, fresh",
            "personal_care": "clean beauty shot, soft lighting, minimal",
            "home_appliances": "modern kitchen/home setting, clean product shot",
        }
        style = style_hints.get(category, "professional product photography, clean layout")

        rich = (
            f"Professional advertisement poster for {product_desc}. "
            f"{style}. "
            f"Bold headline text, prominent brand name, 'Shop Now' CTA button, "
            f"key features displayed with modern typography and professional layout. "
            f"{original_prompt}"
        )
        print(f"    [SMART-PROMPT] Local rich prompt built ({len(rich)} chars)")
        return rich

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

        # Rebuild rich prompt locally (instant, no API call)
        rich_prompt = self._build_rich_prompt_local(
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
