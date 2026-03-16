"""
AdCraft AI - FastAPI Backend
Production-ready web server for ad generation.
Fully local — no external APIs for image/text generation.
"""

import io
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.pipeline import (
    AdCraftPipeline, SUPPORTED_LANGUAGES, OUTPUT_DIR, UPLOAD_DIR
)
from app.smart_prompt import SmartPromptParser, CATEGORY_FIELDS

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AdCraft AI",
    description="AI-powered ad generation platform. Fully local image generation.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


# Lazy pipeline loading
_pipeline = None
_smart_parser = None

def get_pipeline() -> AdCraftPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AdCraftPipeline()
    return _pipeline

def get_smart_parser() -> SmartPromptParser:
    global _smart_parser
    if _smart_parser is None:
        _smart_parser = SmartPromptParser()
    # Connect to pipeline if available (for brand/dataset awareness)
    try:
        pipeline = get_pipeline()
        _smart_parser.set_pipeline(pipeline)
    except Exception:
        pass  # Pipeline not ready yet, parser works without it
    return _smart_parser


# ---------------------------------------------------------------------------
# Frontend Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>AdCraft AI</h1><p>Frontend not found.</p>")


@app.get("/dataset", response_class=HTMLResponse)
async def serve_dataset_explorer():
    """Serve the dataset explorer UI page."""
    ds_path = STATIC_DIR / "dataset.html"
    if ds_path.exists():
        return HTMLResponse(content=ds_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dataset Explorer</h1><p>Page not found.</p>")


@app.get("/hub/{product_id}", response_class=HTMLResponse)
async def serve_product_hub(product_id: str):
    """Serve product hub page - public shareable link."""
    pipeline = get_pipeline()
    product = pipeline.db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    content_list = pipeline.db.get_product_content(product_id)
    analytics = pipeline.db.get_analytics(product_id)

    # Build a simple product hub page
    latest_content = content_list[0] if content_list else None
    pamphlet_url = ""
    if latest_content and latest_content.get("pamphlet_path"):
        pamphlet_url = f"/file?path={latest_content['pamphlet_path']}"

    brand = product.get("brand", product.get("name", "Product"))
    description = product.get("description", "")
    if latest_content and latest_content.get("content_json"):
        cj = latest_content["content_json"]
        if isinstance(cj, dict):
            description = cj.get("product_description", description)

    hub_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{brand} - Product</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="max-w-lg mx-auto py-8 px-4">
        <div class="bg-white rounded-2xl shadow-lg overflow-hidden">
            {"<img src='" + pamphlet_url + "' class='w-full' alt='Product'>" if pamphlet_url else ""}
            <div class="p-6">
                <h1 class="text-2xl font-bold text-gray-900">{brand}</h1>
                <p class="text-gray-600 mt-2">{description}</p>
                {f"<p class='text-xl font-semibold text-green-600 mt-3'>{product.get('price', '')}</p>" if product.get('price') else ""}
                <div class="mt-6 space-y-3">
                    <a href="#" onclick="track('whatsapp')"
                       class="block w-full text-center bg-green-500 text-white py-3 rounded-xl font-semibold hover:bg-green-600 transition">
                        WhatsApp
                    </a>
                    <a href="#" onclick="track('instagram')"
                       class="block w-full text-center bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 rounded-xl font-semibold hover:opacity-90 transition">
                        Instagram
                    </a>
                    <a href="#" onclick="track('website')"
                       class="block w-full text-center bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition">
                        Visit Website
                    </a>
                </div>
            </div>
        </div>
        <p class="text-center text-gray-400 text-sm mt-6">Powered by AdCraft AI</p>
    </div>
    <script>
        function track(platform) {{
            fetch('/api/track/{product_id}?platform=' + platform, {{method: 'POST'}});
        }}
    </script>
</body>
</html>"""
    return HTMLResponse(content=hub_html)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    print(f"\n[API] GET /api/health")
    return {"status": "ok", "version": "2.1.0", "api_keys_required": False}


@app.get("/api/languages")
async def list_languages():
    print(f"\n[API] GET /api/languages -> {len(SUPPORTED_LANGUAGES)} languages")
    return {"languages": SUPPORTED_LANGUAGES}


@app.get("/api/stats")
async def get_stats():
    print(f"\n[API] GET /api/stats")
    pipeline = get_pipeline()
    stats = {
        "total_vectors": pipeline.index.ntotal,
        "total_brands": len(pipeline.brand_matcher.brand_names),
        "device": pipeline.device,
        "products": len(pipeline.db.list_products()),
    }
    print(f"[API] Stats: vectors={stats['total_vectors']:,} | brands={stats['total_brands']} | device={stats['device']} | products={stats['products']}")
    return stats


@app.get("/api/dataset-summary")
async def dataset_summary():
    """Full summary of dataset: all brands, categories, subcategories,
    newly generated brands, and per-brand breakdowns."""
    import math
    from collections import Counter, defaultdict
    from urllib.parse import quote
    from app.brands import BRAND_TAGLINES, BrandMatcher

    print(f"\n[API] GET /api/dataset-summary")
    pipeline = get_pipeline()

    def _clean(val, fallback="Unknown"):
        """Sanitize metadata values — drop NaN, None, empty strings."""
        if val is None:
            return fallback
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return fallback
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", "unknown", ""):
            return fallback
        return s

    # ── Aggregate from FAISS metadata ──
    brand_counter = Counter()
    category_counter = Counter()
    subcategory_counter = Counter()
    language_counter = Counter()
    ad_type_counter = Counter()
    source_counter = Counter()

    brand_detail = defaultdict(lambda: {
        "image_count": 0,
        "categories": Counter(),
        "subcategories": Counter(),
        "ad_types": Counter(),
        "languages": Counter(),
        "generated_count": 0,
        "sample_images": [],
    })

    for _idx, meta in pipeline.id_to_metadata.items():
        brand_raw = _clean(meta.get("brand"), "Unknown")
        brand = BrandMatcher.normalize_brand(brand_raw) if brand_raw != "Unknown" else "Unknown"
        category = _clean(meta.get("category"), "Unknown")
        subcategory = _clean(meta.get("subcategory"), "Unknown")
        language = _clean(meta.get("language"), "unknown")
        ad_type = _clean(meta.get("ad_type"), "unknown")
        source = _clean(meta.get("source"), "unknown")

        brand_counter[brand] += 1
        category_counter[category] += 1
        subcategory_counter[subcategory] += 1
        language_counter[language] += 1
        ad_type_counter[ad_type] += 1
        source_counter[source] += 1

        bd = brand_detail[brand]
        bd["image_count"] += 1
        bd["categories"][category] += 1
        bd["subcategories"][subcategory] += 1
        bd["ad_types"][ad_type] += 1
        bd["languages"][language] += 1
        if ad_type == "generated":
            bd["generated_count"] += 1
        # Collect ALL images per brand with timestamp for sorting
        img_path = meta.get("image_path", "")
        if img_path and isinstance(img_path, str) and img_path.strip():
            encoded = quote(img_path, safe="/\\:")
            ts = meta.get("timestamp", "")
            bd["sample_images"].append({
                "url": f"/file?path={encoded}",
                "timestamp": ts,
                "ad_type": ad_type,
            })

    # Track unlabeled count, then remove from brand list
    unlabeled_count = brand_counter.get("Unknown", 0)
    brand_detail.pop("Unknown", None)
    brand_counter.pop("Unknown", None)

    # Case-insensitive tagline lookup
    tagline_lookup = {k.lower(): v for k, v in BRAND_TAGLINES.items()}

    # ── Build per-brand list sorted by image count ──
    brands_list = []
    for brand_name in sorted(brand_detail, key=lambda b: brand_detail[b]["image_count"], reverse=True):
        bd = brand_detail[brand_name]
        # Sort newest first, keep only 8 for fast loading
        sorted_images = sorted(
            bd["sample_images"],
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )
        brands_list.append({
            "brand": brand_name,
            "image_count": bd["image_count"],
            "generated_count": bd["generated_count"],
            "categories": dict(bd["categories"].most_common()),
            "subcategories": dict(bd["subcategories"].most_common()),
            "ad_types": dict(bd["ad_types"].most_common()),
            "tagline": tagline_lookup.get(brand_name.lower(), ""),
            "sample_images": [img["url"] for img in sorted_images[:8]],
        })

    # ── Newly generated brands (only have generated images) ──
    generated_brands = [
        b for b in brands_list
        if b["generated_count"] > 0 and b["ad_types"].get("generated", 0) == b["image_count"]
    ]

    # ── Category → subcategory tree ──
    cat_subcat_tree = defaultdict(Counter)
    for _idx, meta in pipeline.id_to_metadata.items():
        cat = _clean(meta.get("category"), "Unknown")
        sub = _clean(meta.get("subcategory"), "Unknown")
        cat_subcat_tree[cat][sub] += 1
    category_tree = {
        cat: dict(subs.most_common())
        for cat, subs in sorted(cat_subcat_tree.items())
    }

    # ── Products from database ──
    db_products = pipeline.db.list_products()
    db_brands = Counter(p.get("brand", "") for p in db_products if p.get("brand"))
    db_categories = Counter(p.get("category", "") for p in db_products if p.get("category"))

    summary = {
        "total_vectors": pipeline.index.ntotal,
        "total_brands": len(brand_counter),
        "total_labeled_images": pipeline.index.ntotal - unlabeled_count,
        "total_unlabeled_images": unlabeled_count,
        "total_categories": len(category_counter),
        "total_subcategories": len(subcategory_counter),
        "total_generated_images": ad_type_counter.get("generated", 0),
        "total_products_in_db": len(db_products),

        "brands": brands_list,
        "newly_created_brands": generated_brands,

        "categories": dict(category_counter.most_common()),
        "subcategories": dict(subcategory_counter.most_common()),
        "category_tree": category_tree,

        "languages": dict(language_counter.most_common()),
        "ad_types": dict(ad_type_counter.most_common()),
        "sources": dict(source_counter.most_common()),

        "db_brands": dict(db_brands.most_common()),
        "db_categories": dict(db_categories.most_common()),
    }

    print(f"[API] Dataset summary: {summary['total_brands']} brands | "
          f"{summary['total_categories']} categories | "
          f"{summary['total_subcategories']} subcategories | "
          f"{summary['total_vectors']:,} vectors | "
          f"{summary['total_generated_images']} generated | "
          f"{len(generated_brands)} new brands")
    return JSONResponse(content=summary)


# ---------------------------------------------------------------------------
# Smart Prompt — Parse & Validate Product Data
# ---------------------------------------------------------------------------

@app.get("/api/category-fields")
async def get_category_fields(category: str = Query("general")):
    """Get required and optional fields for a product category."""
    print(f"\n[API] GET /api/category-fields?category={category}")
    parser = get_smart_parser()
    fields = parser.get_category_fields(category)
    return JSONResponse(content=fields)


@app.get("/api/categories")
async def list_categories():
    """List all supported product categories with their required fields."""
    print(f"\n[API] GET /api/categories")
    cats = {}
    for key, val in CATEGORY_FIELDS.items():
        cats[key] = {
            "display_name": val["display_name"],
            "required_count": len(val["required"]),
            "required_fields": list(val["required"].keys()),
        }
    return JSONResponse(content={"categories": cats})


@app.post("/api/parse-prompt")
async def parse_prompt(prompt: str = Form(...)):
    """Parse a free-text product prompt into structured catalog data.

    Uses AI to extract product details (brand, name, type, size, material, etc.)
    and identifies which required fields are missing based on the detected category.

    Returns extracted fields, missing required fields, and completeness score.
    """
    import time as _time
    t0 = _time.time()
    print(f"\n{'#' * 70}")
    print(f"[API] POST /api/parse-prompt")
    print(f"[API] Prompt: \"{prompt[:200]}\"")
    print(f"{'#' * 70}")

    parser = get_smart_parser()
    result = parser.parse_prompt(prompt)

    elapsed = _time.time() - t0
    print(f"[API] Parse complete in {elapsed:.2f}s | Category: {result['category']} | "
          f"Completeness: {result['completeness']:.0%} | "
          f"Missing: {list(result['missing_required'].keys())}")

    return JSONResponse(content=result)


@app.post("/api/validate-fields")
async def validate_fields(
    parsed_data: str = Form(...),
    additional_fields: str = Form("{}"),
):
    """Validate and merge additional fields into previously parsed data.

    - parsed_data: JSON string of previously parsed prompt data
    - additional_fields: JSON string of new field values to merge
    """
    print(f"\n[API] POST /api/validate-fields")
    try:
        parsed = json.loads(parsed_data)
        additional = json.loads(additional_fields)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    parser = get_smart_parser()
    result = parser.merge_fields(parsed, additional)

    print(f"[API] Validation complete | Completeness: {result['completeness']:.0%} | "
          f"Missing: {list(result['missing_required'].keys())}")
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Generate Ad Creative
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def generate_ad(
    prompt: str = Form(...),
    languages: str = Form("en"),
    brand: str = Form(""),
    image: Optional[UploadFile] = File(None),
    product_metadata: str = Form(""),
):
    """Generate ad creative from text prompt and optional image.

    - prompt: Text description (e.g., "Bisleri water bottle pamphlet")
    - languages: Comma-separated language codes (e.g., "en,hi,ta,bn")
    - brand: Brand name for logo fetching (e.g., "Nike", "Samsung")
    - image: Optional product image upload
    - product_metadata: JSON string of structured product data from Smart Prompt
    """
    import time as _time
    api_start = _time.time()
    print(f"\n{'#' * 70}")
    print(f"[API] POST /api/generate")
    print(f"[API] Prompt: \"{prompt}\"")
    print(f"[API] Languages: {languages}")
    print(f"[API] Brand: {brand or 'auto-detect'}")
    print(f"[API] Image uploaded: {image.filename if image and image.filename else 'None'}")
    print(f"[API] Product metadata: {'YES' if product_metadata else 'NO'}")
    print(f"{'#' * 70}")

    pipeline = get_pipeline()

    # Parse product metadata if provided (from Smart Prompt)
    metadata = {}
    if product_metadata:
        try:
            metadata = json.loads(product_metadata)
            print(f"[API] Parsed product metadata: {list(metadata.keys())}")
            # Use brand from metadata if not explicitly provided
            if not brand.strip() and metadata.get("brand"):
                brand = metadata["brand"]
                print(f"[API] Brand from metadata: {brand}")
            # Enrich prompt with metadata
            if metadata.get("rich_prompt"):
                prompt = metadata["rich_prompt"]
                print(f"[API] Using enriched prompt from metadata")
        except json.JSONDecodeError:
            print(f"[API] WARNING: Invalid product_metadata JSON, ignoring")

    # Normalize brand name to prevent case-variant duplicates
    if brand.strip():
        from app.brands import BrandMatcher as _BM
        brand = _BM.normalize_brand(brand)
        print(f"[API] Normalized brand: {brand}")

    # Parse languages
    lang_list = [l.strip() for l in languages.split(",") if l.strip()]
    if not lang_list:
        lang_list = ["en"]
    print(f"[API] Parsed languages: {lang_list}")

    # Handle uploaded image
    uploaded_image = None
    if image and image.filename:
        try:
            contents = await image.read()
            uploaded_image = Image.open(io.BytesIO(contents)).convert("RGB")
            # Save uploaded image
            img_filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}"
            img_path = UPLOAD_DIR / img_filename
            uploaded_image.save(str(img_path))
            print(f"[API] Uploaded image saved: {img_path} | Size: {uploaded_image.size}")
        except Exception as e:
            print(f"[API] ERROR: Invalid image upload: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    # Run pipeline
    result = pipeline.generate(
        query=prompt,
        languages=lang_list,
        uploaded_image=uploaded_image,
        brand_override=brand.strip() if brand.strip() else None,
    )

    # Build response
    response = {
        "success": len(result.errors) == 0,
        "query": result.query,
        "brand": result.brand_match,
        "timings": result.stage_timings,
        "content": {
            "product_title": result.product_title,
            "product_description": result.product_description,
            "instagram_caption": result.instagram_caption,
            "whatsapp_copy": result.whatsapp_copy,
            "hashtags": result.hashtags,
        },
        "translations": result.translations,
        "languages_generated": result.languages_generated,
        "pamphlet_url": f"/file?path={result.pamphlet_path}" if result.pamphlet_path else None,
        "product_image_url": f"/file?path={result.product_image_path}" if result.product_image_path else None,
        "retrieved_ads": result.retrieved_ads[:3],
        "colors": result.extracted_colors,
        "dataset_paths": result.dataset_paths,
        "errors": result.errors,
    }

    api_elapsed = _time.time() - api_start
    print(f"\n[API] Response ready | Success: {response['success']} | Total API time: {api_elapsed:.2f}s")
    print(f"[API] Image model used: {result.image_generator_used}")
    print(f"[API] Languages generated: {result.languages_generated}")
    if result.errors:
        print(f"[API] Errors: {result.errors}")

    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# Product CRUD
# ---------------------------------------------------------------------------

@app.post("/api/products")
async def create_product(
    name: str = Form(...),
    description: str = Form(""),
    price: str = Form(""),
    category: str = Form(""),
    brand: str = Form(""),
    images: List[UploadFile] = File(None),
):
    """Create a new product entry."""
    print(f"\n[API] POST /api/products")
    print(f"[API] Name: {name} | Brand: {brand} | Category: {category} | Price: {price}")
    pipeline = get_pipeline()

    image_paths = []
    if images:
        for img_file in images[:3]:  # Max 3 images
            if img_file and img_file.filename:
                try:
                    contents = await img_file.read()
                    img = Image.open(io.BytesIO(contents)).convert("RGB")
                    filename = f"product_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{img_file.filename}"
                    path = str(UPLOAD_DIR / filename)
                    img.save(path, quality=95)
                    image_paths.append(path)
                    print(f"[API] Product image saved: {path} | Size: {img.size}")
                except Exception as e:
                    print(f"[API] Failed to save product image: {e}")
                    continue

    product_id = pipeline.db.create_product(
        name=name, description=description, price=price,
        category=category, brand=brand, image_paths=image_paths,
    )

    print(f"[API] Product created: {product_id} | Images: {len(image_paths)}")
    return {"product_id": product_id, "message": "Product created successfully"}


@app.get("/api/products")
async def list_products():
    pipeline = get_pipeline()
    products = pipeline.db.list_products()
    # Add hub URLs
    for p in products:
        p["hub_url"] = f"/hub/{p['id']}"
        p["image_urls"] = [f"/file?path={ip}" for ip in p.get("image_paths", [])]
    return {"products": products}


@app.get("/api/products/{product_id}")
async def get_product(product_id: str):
    pipeline = get_pipeline()
    product = pipeline.db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    content = pipeline.db.get_product_content(product_id)
    analytics = pipeline.db.get_analytics(product_id)

    product["hub_url"] = f"/hub/{product_id}"
    product["image_urls"] = [f"/file?path={ip}" for ip in product.get("image_paths", [])]

    return {
        "product": product,
        "content": content,
        "analytics": analytics,
    }


@app.delete("/api/products/{product_id}")
async def delete_product(product_id: str):
    pipeline = get_pipeline()
    product = pipeline.db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    pipeline.db.delete_product(product_id)
    return {"message": "Product deleted"}


# ---------------------------------------------------------------------------
# Generate for existing product
# ---------------------------------------------------------------------------

@app.post("/api/products/{product_id}/generate")
async def generate_for_product(
    product_id: str,
    prompt: str = Form(""),
    languages: str = Form("en"),
):
    """Generate ad content for an existing product."""
    print(f"\n[API] POST /api/products/{product_id}/generate")
    print(f"[API] Prompt: \"{prompt or 'auto-generated'}\" | Languages: {languages}")
    pipeline = get_pipeline()
    product = pipeline.db.get_product(product_id)
    if not product:
        print(f"[API] ERROR: Product {product_id} not found")
        raise HTTPException(status_code=404, detail="Product not found")

    print(f"[API] Product: {product['name']} | Brand: {product.get('brand', 'N/A')}")
    query = prompt or f"{product['brand'] or product['name']} {product['category'] or 'product'} advertisement"
    lang_list = [l.strip() for l in languages.split(",") if l.strip()] or ["en"]

    uploaded_image = None
    if product["image_paths"]:
        try:
            uploaded_image = Image.open(product["image_paths"][0]).convert("RGB")
            print(f"[API] Using product image: {product['image_paths'][0]} | Size: {uploaded_image.size}")
        except Exception as e:
            print(f"[API] Failed to load product image: {e}")

    result = pipeline.generate(
        query=query, languages=lang_list,
        uploaded_image=uploaded_image, product_id=product_id,
    )

    print(f"[API] Generation for product {product_id} complete | Success: {len(result.errors) == 0}")
    return {
        "success": len(result.errors) == 0,
        "content": {
            "product_title": result.product_title,
            "product_description": result.product_description,
            "instagram_caption": result.instagram_caption,
            "whatsapp_copy": result.whatsapp_copy,
            "hashtags": result.hashtags,
        },
        "translations": result.translations,
        "pamphlet_url": f"/file?path={result.pamphlet_path}" if result.pamphlet_path else None,
        "product_image_url": f"/file?path={result.product_image_path}" if result.product_image_path else None,
        "dataset_paths": result.dataset_paths,
        "timings": result.stage_timings,
        "errors": result.errors,
    }


# ---------------------------------------------------------------------------
# Image Enhancement
# ---------------------------------------------------------------------------

@app.post("/api/enhance")
async def enhance_image(image: UploadFile = File(...)):
    """Enhance a product image (brightness, contrast, sharpness, color)."""
    print(f"\n[API] POST /api/enhance")
    print(f"[API] Image: {image.filename}")
    pipeline = get_pipeline()
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        print(f"[API] Original image: {img.size} | Mode: {img.mode}")
    except Exception:
        print(f"[API] ERROR: Invalid image upload")
        raise HTTPException(status_code=400, detail="Invalid image")

    enhanced = pipeline.enhance_image(img)
    filename = f"enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = str(OUTPUT_DIR / filename)
    enhanced.save(path, quality=95)
    print(f"[API] Enhanced image saved: {path} | Size: {enhanced.size}")

    return {
        "enhanced_image_url": f"/file?path={path}",
        "original_size": img.size,
        "enhanced_size": enhanced.size,
    }


# ---------------------------------------------------------------------------
# Description & Caption Generation
# ---------------------------------------------------------------------------

@app.post("/api/describe")
async def generate_description(
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    """Generate product description from prompt and optional image."""
    print(f"\n[API] POST /api/describe")
    print(f"[API] Prompt: \"{prompt}\"")
    print(f"[API] Image: {image.filename if image and image.filename else 'None'}")
    pipeline = get_pipeline()

    uploaded_image = None
    if image and image.filename:
        try:
            contents = await image.read()
            uploaded_image = Image.open(io.BytesIO(contents)).convert("RGB")
            print(f"[API] Uploaded image: {uploaded_image.size}")
        except Exception as e:
            print(f"[API] Image load failed: {e}")

    result = pipeline.generate_description(prompt, uploaded_image)
    print(f"[API] Description generated: {len(result)} fields")
    return result


@app.post("/api/captions")
async def generate_captions(
    prompt: str = Form(...),
    languages: str = Form("en"),
):
    """Generate marketing captions in multiple languages."""
    print(f"\n[API] POST /api/captions")
    print(f"[API] Prompt: \"{prompt}\" | Languages: {languages}")
    pipeline = get_pipeline()
    lang_list = [l.strip() for l in languages.split(",") if l.strip()] or ["en"]
    result = pipeline.generate_captions(prompt, lang_list)
    print(f"[API] Captions generated for {len(result)} languages: {list(result.keys())}")
    return result


# ---------------------------------------------------------------------------
# Analytics & Tracking
# ---------------------------------------------------------------------------

@app.post("/api/track/{product_id}")
async def track_click(product_id: str, platform: str = Query("direct"), source: str = Query("")):
    """Track a click on a product hub link."""
    print(f"\n[API] POST /api/track/{product_id} | platform={platform} | source={source}")
    pipeline = get_pipeline()
    pipeline.db.track_click(product_id, platform, source)
    return {"tracked": True}


@app.get("/api/analytics/{product_id}")
async def get_analytics(product_id: str):
    print(f"\n[API] GET /api/analytics/{product_id}")
    pipeline = get_pipeline()
    analytics = pipeline.db.get_analytics(product_id)
    print(f"[API] Analytics: total_clicks={analytics['total_clicks']} | platforms={analytics['by_platform']}")
    return analytics


# ---------------------------------------------------------------------------
# File Serving
# ---------------------------------------------------------------------------

@app.get("/file")
async def serve_file(path: str):
    """Serve a generated file by path."""
    from app.models import BASE_DIR

    file_path = Path(path)
    # Resolve relative paths against project root
    if not file_path.is_absolute():
        file_path = BASE_DIR / file_path

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Security: only serve files from allowed directories
    allowed_dirs = [OUTPUT_DIR, UPLOAD_DIR]
    is_allowed = any(
        str(file_path.resolve()).startswith(str(d.resolve()))
        for d in allowed_dirs
    )

    if not is_allowed:
        # Also allow data/images for thumbnails
        data_dir = BASE_DIR / "data"
        if str(file_path.resolve()).startswith(str(data_dir.resolve())):
            is_allowed = True

    if not is_allowed:
        raise HTTPException(status_code=403, detail="Access denied")

    media_type = "image/png"
    suffix = file_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif suffix == ".json":
        media_type = "application/json"

    return FileResponse(str(file_path), media_type=media_type)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Pre-load pipeline on startup for faster first request."""
    print("\n" + "=" * 70)
    print("  MAdVerse AI Server Starting")
    print("=" * 70)
    print("  Frontend:  http://localhost:8000")
    print("  API docs:  http://localhost:8000/docs")
    print("  Health:    http://localhost:8000/api/health")
    print("  Languages: http://localhost:8000/api/languages")
    print("  Stats:     http://localhost:8000/api/stats")
    print("  Dataset API: http://localhost:8000/api/dataset-summary")
    print("  Dataset UI:  http://localhost:8000/dataset")
    print("=" * 70)
    print()
