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


def _build_gallery_html(image_urls: list, product_name: str) -> str:
    """Build a horizontal scrollable image gallery for the hub page."""
    if not image_urls or len(image_urls) < 2:
        return ""
    imgs = "".join(
        f"<img src='{url}' alt='{product_name}' class='w-32 h-32 object-cover rounded-lg flex-shrink-0 border border-gray-100'>"
        for url in image_urls[:6]
    )
    return f"""<div class="mt-4">
                <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Gallery</h3>
                <div class="flex space-x-2 overflow-x-auto pb-2 scrollbar-hide">{imgs}</div>
            </div>"""


def _build_content_card(title: str, content: str, card_id: str, icon_path: str) -> str:
    """Build a copyable content card for generated text (caption, whatsapp copy)."""
    if not content:
        return ""
    # Escape single quotes and backslashes for JS
    safe_content = content.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    return f"""<div class="mt-4">
                <div class="bg-white rounded-xl border border-gray-100 overflow-hidden card-hover">
                    <div class="flex items-center justify-between px-4 py-2.5 border-b border-gray-50">
                        <div class="flex items-center space-x-2">
                            <svg class="w-3.5 h-3.5 text-gray-400" fill="currentColor" viewBox="0 0 24 24"><path d="{icon_path}"/></svg>
                            <span class="text-xs font-semibold text-gray-500 uppercase tracking-wider">{title}</span>
                        </div>
                        <button onclick="copyText('{safe_content}')" class="text-xs text-gray-400 hover:text-gray-600 transition flex items-center space-x-1 px-2 py-1 rounded-md hover:bg-gray-50">
                            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"/></svg>
                            <span>Copy</span>
                        </button>
                    </div>
                    <p class="text-sm text-gray-600 leading-relaxed p-4">{content}</p>
                </div>
            </div>"""


@app.get("/hub/{product_id}", response_class=HTMLResponse)
async def serve_product_hub(product_id: str):
    """Serve product hub page - public shareable link."""
    pipeline = get_pipeline()
    product = pipeline.db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    content_list = pipeline.db.get_product_content(product_id)
    analytics = pipeline.db.get_analytics(product_id)

    # Build product hub page
    latest_content = content_list[0] if content_list else None
    pamphlet_url = ""
    product_image_url = ""
    if latest_content and latest_content.get("pamphlet_path"):
        pamphlet_url = f"/file?path={latest_content['pamphlet_path']}"
    if latest_content and latest_content.get("product_image_path"):
        product_image_url = f"/file?path={latest_content['product_image_path']}"

    brand = product.get("brand", product.get("name", "Product"))
    description = product.get("description", "")
    product_title = ""
    instagram_caption = ""
    whatsapp_copy = ""
    hashtags = ""
    if latest_content and latest_content.get("content_json"):
        cj = latest_content["content_json"]
        if isinstance(cj, dict):
            description = cj.get("product_description", description)
            product_title = cj.get("product_title", "")
            instagram_caption = cj.get("instagram_caption", "")
            whatsapp_copy = cj.get("whatsapp_copy", "")
            hashtags = cj.get("hashtags", "")

    product_name = product.get("name", brand)
    category = product.get("category", "")
    price = product.get("price", "")

    # Product images for gallery
    image_urls = [f"/file?path={ip}" for ip in product.get("image_paths", [])]
    # Use pamphlet as hero, fallback to first product image
    hero_image = pamphlet_url or product_image_url or (image_urls[0] if image_urls else "")

    # Analytics summary
    total_clicks = analytics.get("total_clicks", 0) if analytics else 0
    platform_clicks = analytics.get("platform_clicks", {}) if analytics else {}

    # Build hashtag pills
    hashtag_list = []
    if hashtags:
        if isinstance(hashtags, list):
            hashtag_list = hashtags
        elif isinstance(hashtags, str):
            hashtag_list = [h.strip() for h in hashtags.replace("#", "").split() if h.strip()]

    # Pre-build complex HTML sections
    category_html = f'<span class="text-xs text-gray-400 bg-gray-50 px-2.5 py-1 rounded-full">{category}</span>' if category else ''

    if hero_image:
        hero_html = f"<div class='relative hero-gradient'><img src='{hero_image}' class='w-full max-h-96 object-cover' alt='{product_name}'></div>"
    else:
        hero_html = "<div class='h-24 bg-gradient-to-br from-gray-100 to-gray-200'></div>"

    brand_badge = f'<span class="inline-block text-xs font-medium bg-gray-900 text-white px-2.5 py-0.5 rounded-full mb-2">{brand}</span>' if brand and brand != product_name else ''
    title_display = product_title or product_name
    subtitle_html = f'<p class="text-sm text-gray-400 mt-0.5">{product_name}</p>' if product_title and product_title != product_name else ''
    price_html = f'<span class="text-xl font-bold text-gray-900 flex-shrink-0 bg-gray-50 px-3 py-1 rounded-lg">{price}</span>' if price else ''
    desc_html = f'<p class="text-sm text-gray-600 mt-3 leading-relaxed">{description}</p>' if description else ''

    if hashtag_list:
        tag_pills = "".join(f"<span class='text-xs text-gray-500 bg-gray-50 px-2 py-0.5 rounded-full'>#{h}</span>" for h in hashtag_list[:8])
        hashtag_html = f"<div class='flex flex-wrap gap-1.5 mt-3'>{tag_pills}</div>"
    else:
        hashtag_html = ""

    gallery_html = _build_gallery_html(image_urls, product_name)
    ig_card_html = _build_content_card("Instagram Caption", instagram_caption, "instagram", "M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z")
    wa_card_html = _build_content_card("WhatsApp Message", whatsapp_copy, "whatsapp", "M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347z M12 0C5.373 0 0 5.373 0 12c0 2.625.846 5.059 2.284 7.034L.789 23.492l4.624-1.467A11.955 11.955 0 0012 24c6.627 0 12-5.373 12-12S18.627 0 12 0zm0 21.75c-2.115 0-4.093-.657-5.727-1.778l-.41-.253-2.742.87.908-2.686-.278-.432A9.713 9.713 0 012.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75z")

    if total_clicks > 0:
        stats_html = f'''<div class="mt-5 bg-white rounded-xl border border-gray-100 p-4">
                <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-2">
                        <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
                        <span class="text-sm text-gray-500">Total Engagement</span>
                    </div>
                    <span class="text-sm font-semibold text-gray-900">{total_clicks} clicks</span>
                </div>
            </div>'''
    else:
        stats_html = ""

    hub_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{product_name} - {brand}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        body {{ font-family: 'Inter', sans-serif; }}
        .hero-gradient {{ background: linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.03) 100%); }}
        .copy-toast {{ animation: fadeInUp 0.3s ease-out; }}
        @keyframes fadeInUp {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:translateY(0); }} }}
        .card-hover {{ transition: box-shadow 0.2s ease, transform 0.2s ease; }}
        .card-hover:hover {{ box-shadow: 0 4px 24px rgba(0,0,0,0.08); transform: translateY(-1px); }}
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <!-- Nav -->
    <nav class="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div class="max-w-2xl mx-auto px-4 h-14 flex items-center justify-between">
            <div class="flex items-center space-x-2">
                <svg class="w-4 h-4 text-gray-900" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                <span class="font-bold text-gray-900 text-sm tracking-tight">AdCraft AI</span>
            </div>
            <div class="flex items-center space-x-3">
                {category_html}
                <button onclick="shareHub()" class="text-gray-400 hover:text-gray-600 transition p-1" title="Share">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>
                </button>
            </div>
        </div>
    </nav>

    <div class="max-w-2xl mx-auto">
        <!-- Hero Section -->
        {hero_html}

        <div class="px-4 pb-8">
            <!-- Product Header -->
            <div class="bg-white rounded-xl shadow-sm border border-gray-100 -mt-6 relative z-10 p-5">
                <div class="flex items-start justify-between gap-4">
                    <div class="min-w-0 flex-1">
                        {brand_badge}
                        <h1 class="text-xl font-bold text-gray-900 leading-tight">{title_display}</h1>
                        {subtitle_html}
                    </div>
                    {price_html}
                </div>
                {desc_html}

                <!-- Hashtags -->
                {hashtag_html}
            </div>

            <!-- Image Gallery -->
            {gallery_html}

            <!-- Generated Content Cards -->
            {ig_card_html}

            {wa_card_html}

            <!-- Share Actions -->
            <div class="mt-5">
                <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Share Product</h3>
                <div class="grid grid-cols-3 gap-2.5 mb-2.5">
                    <a href="#" onclick="shareWhatsApp(); return false;"
                       class="card-hover flex flex-col items-center justify-center bg-white border border-gray-200 py-4 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-50 transition cursor-pointer">
                        <svg class="w-5 h-5 mb-1.5 text-green-600" fill="currentColor" viewBox="0 0 24 24"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347z"/><path d="M12 0C5.373 0 0 5.373 0 12c0 2.625.846 5.059 2.284 7.034L.789 23.492l4.624-1.467A11.955 11.955 0 0012 24c6.627 0 12-5.373 12-12S18.627 0 12 0zm0 21.75c-2.115 0-4.093-.657-5.727-1.778l-.41-.253-2.742.87.908-2.686-.278-.432A9.713 9.713 0 012.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75z"/></svg>
                        <span class="text-xs">WhatsApp</span>
                    </a>
                    <a href="#" onclick="shareInstagram(); return false;"
                       class="card-hover flex flex-col items-center justify-center bg-white border border-gray-200 py-4 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-50 transition cursor-pointer">
                        <svg class="w-5 h-5 mb-1.5 text-pink-600" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z"/></svg>
                        <span class="text-xs">Instagram</span>
                    </a>
                    <a href="#" onclick="shareFacebook(); return false;"
                       class="card-hover flex flex-col items-center justify-center bg-white border border-gray-200 py-4 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-50 transition cursor-pointer">
                        <svg class="w-5 h-5 mb-1.5 text-blue-700" fill="currentColor" viewBox="0 0 24 24"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
                        <span class="text-xs">Facebook</span>
                    </a>
                </div>
                <div class="grid grid-cols-3 gap-2.5 mb-2.5">
                    <a href="#" onclick="shareTwitter(); return false;"
                       class="card-hover flex flex-col items-center justify-center bg-white border border-gray-200 py-4 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-50 transition cursor-pointer">
                        <svg class="w-5 h-5 mb-1.5 text-gray-900" fill="currentColor" viewBox="0 0 24 24"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
                        <span class="text-xs">X (Twitter)</span>
                    </a>
                    <a href="#" onclick="shareLinkedIn(); return false;"
                       class="card-hover flex flex-col items-center justify-center bg-white border border-gray-200 py-4 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-50 transition cursor-pointer">
                        <svg class="w-5 h-5 mb-1.5 text-blue-600" fill="currentColor" viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
                        <span class="text-xs">LinkedIn</span>
                    </a>
                    <a href="#" onclick="shareTelegram(); return false;"
                       class="card-hover flex flex-col items-center justify-center bg-white border border-gray-200 py-4 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-50 transition cursor-pointer">
                        <svg class="w-5 h-5 mb-1.5 text-sky-500" fill="currentColor" viewBox="0 0 24 24"><path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/></svg>
                        <span class="text-xs">Telegram</span>
                    </a>
                </div>
                <div class="grid grid-cols-3 gap-2.5">
                    <a href="#" onclick="sharePinterest(); return false;"
                       class="card-hover flex flex-col items-center justify-center bg-white border border-gray-200 py-4 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-50 transition cursor-pointer">
                        <svg class="w-5 h-5 mb-1.5 text-red-600" fill="currentColor" viewBox="0 0 24 24"><path d="M12.017 0C5.396 0 .029 5.367.029 11.987c0 5.079 3.158 9.417 7.618 11.162-.105-.949-.199-2.403.041-3.439.219-.937 1.406-5.957 1.406-5.957s-.359-.72-.359-1.781c0-1.668.967-2.914 2.171-2.914 1.023 0 1.518.769 1.518 1.69 0 1.029-.655 2.568-.994 3.995-.283 1.194.599 2.169 1.777 2.169 2.133 0 3.772-2.249 3.772-5.495 0-2.873-2.064-4.882-5.012-4.882-3.414 0-5.418 2.561-5.418 5.207 0 1.031.397 2.138.893 2.738a.36.36 0 0 1 .083.345l-.333 1.36c-.053.22-.174.267-.402.161-1.499-.698-2.436-2.889-2.436-4.649 0-3.785 2.75-7.262 7.929-7.262 4.163 0 7.398 2.967 7.398 6.931 0 4.136-2.607 7.464-6.227 7.464-1.216 0-2.359-.631-2.75-1.378l-.748 2.853c-.271 1.043-1.002 2.35-1.492 3.146C9.57 23.812 10.763 24 12.017 24c6.624 0 11.99-5.367 11.99-11.988C24.007 5.367 18.641 0 12.017 0z"/></svg>
                        <span class="text-xs">Pinterest</span>
                    </a>
                    <a href="#" onclick="track('website'); return false;"
                       class="card-hover flex flex-col items-center justify-center bg-white border border-gray-200 py-4 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-50 transition cursor-pointer">
                        <svg class="w-5 h-5 mb-1.5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"/></svg>
                        <span class="text-xs">Website</span>
                    </a>
                    <a href="#" onclick="shareEmail(); return false;"
                       class="card-hover flex flex-col items-center justify-center bg-white border border-gray-200 py-4 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-50 transition cursor-pointer">
                        <svg class="w-5 h-5 mb-1.5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
                        <span class="text-xs">Email</span>
                    </a>
                </div>
            </div>

            <!-- Engagement Stats -->
            {stats_html}

            <!-- Footer -->
            <div class="mt-8 pb-4 text-center">
                <div class="flex items-center justify-center space-x-1.5 text-gray-400">
                    <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                    <span class="text-xs">Powered by Adcraft AI</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Copy Toast -->
    <div id="copyToast" class="fixed bottom-6 left-1/2 -translate-x-1/2 bg-gray-900 text-white text-sm px-4 py-2.5 rounded-xl shadow-lg hidden copy-toast z-50">
        <div class="flex items-center space-x-2">
            <svg class="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
            <span>Copied to clipboard</span>
        </div>
    </div>

    <script>
        const hubUrl = window.location.href;
        const productName = '{product_name}';
        const shareText = '{product_name} - Check out this product!';

        function track(platform) {{
            fetch('/api/track/{product_id}?platform=' + platform, {{method: 'POST'}});
        }}

        function copyText(text) {{
            navigator.clipboard.writeText(text).then(() => {{
                const toast = document.getElementById('copyToast');
                toast.classList.remove('hidden');
                setTimeout(() => toast.classList.add('hidden'), 2000);
            }});
        }}

        function shareHub() {{
            if (navigator.share) {{
                navigator.share({{ title: productName, url: hubUrl }});
            }} else {{
                copyText(hubUrl);
            }}
        }}

        function shareWhatsApp() {{
            track('whatsapp');
            window.open('https://api.whatsapp.com/send?text=' + encodeURIComponent(shareText + '\\n' + hubUrl), '_blank');
        }}

        function shareInstagram() {{
            track('instagram');
            copyText(hubUrl);
            const toast = document.getElementById('copyToast');
            toast.querySelector('span').textContent = 'Link copied! Paste in Instagram';
            toast.classList.remove('hidden');
            setTimeout(() => {{ toast.classList.add('hidden'); toast.querySelector('span').textContent = 'Copied to clipboard'; }}, 3000);
        }}

        function shareFacebook() {{
            track('facebook');
            window.open('https://www.facebook.com/sharer/sharer.php?u=' + encodeURIComponent(hubUrl), '_blank', 'width=600,height=400');
        }}

        function shareTwitter() {{
            track('twitter');
            window.open('https://twitter.com/intent/tweet?text=' + encodeURIComponent(shareText) + '&url=' + encodeURIComponent(hubUrl), '_blank', 'width=600,height=400');
        }}

        function shareLinkedIn() {{
            track('linkedin');
            window.open('https://www.linkedin.com/sharing/share-offsite/?url=' + encodeURIComponent(hubUrl), '_blank', 'width=600,height=400');
        }}

        function shareTelegram() {{
            track('telegram');
            window.open('https://t.me/share/url?url=' + encodeURIComponent(hubUrl) + '&text=' + encodeURIComponent(shareText), '_blank');
        }}

        function sharePinterest() {{
            track('pinterest');
            const imgUrl = document.querySelector('img')?.src || '';
            window.open('https://pinterest.com/pin/create/button/?url=' + encodeURIComponent(hubUrl) + '&media=' + encodeURIComponent(imgUrl) + '&description=' + encodeURIComponent(shareText), '_blank', 'width=600,height=400');
        }}

        function shareEmail() {{
            track('email');
            window.location.href = 'mailto:?subject=' + encodeURIComponent(productName) + '&body=' + encodeURIComponent(shareText + '\\n\\n' + hubUrl);
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
async def parse_prompt(prompt: str = Form(...), use_ai: str = Form("false")):
    """Parse a free-text product prompt into structured catalog data.

    Fast local extraction (<20ms) by default. Set use_ai=true for AI-enhanced
    accuracy via Pollinations (adds ~3-8s).

    Returns extracted fields, missing required fields, and completeness score.
    """
    import time as _time
    t0 = _time.time()
    ai_mode = use_ai.lower() in ("true", "1", "yes")
    print(f"\n{'#' * 70}")
    print(f"[API] POST /api/parse-prompt (ai={ai_mode})")
    print(f"[API] Prompt: \"{prompt[:200]}\"")
    print(f"{'#' * 70}")

    parser = get_smart_parser()
    result = parser.parse_prompt(prompt, use_ai=ai_mode)

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
# Save Generated Ad as Product
# ---------------------------------------------------------------------------

@app.post("/api/save-generated")
async def save_generated_as_product(
    name: str = Form(...),
    description: str = Form(""),
    price: str = Form(""),
    category: str = Form(""),
    brand: str = Form(""),
    content_json: str = Form("{}"),
    pamphlet_path: str = Form(""),
    product_image_path: str = Form(""),
    languages_generated: str = Form("en"),
):
    """Save a generated ad result as a product with all its content.

    Called after ad generation to store the result in the product catalog
    so it can be shared via hub pages and social media.
    """
    import time as _time
    t0 = _time.time()
    print(f"\n{'#' * 70}")
    print(f"[API] POST /api/save-generated")
    print(f"[API] Name: {name} | Brand: {brand} | Category: {category}")
    print(f"{'#' * 70}")

    pipeline = get_pipeline()

    # Parse content JSON
    try:
        content = json.loads(content_json)
    except json.JSONDecodeError:
        content = {}

    # Build image paths list
    image_paths = []
    if product_image_path:
        image_paths.append(product_image_path)
    if pamphlet_path and pamphlet_path != product_image_path:
        image_paths.append(pamphlet_path)

    # Create product
    product_id = pipeline.db.create_product(
        name=name,
        description=description or content.get("product_description", ""),
        price=price,
        category=category,
        brand=brand,
        image_paths=image_paths,
    )

    # Update enhanced_image_path to the pamphlet
    if pamphlet_path:
        pipeline.db.update_product(product_id, enhanced_image_path=pamphlet_path)

    # Save generated content for each language
    lang_list = [l.strip() for l in languages_generated.split(",") if l.strip()] or ["en"]
    for lang in lang_list:
        lang_content = dict(content)
        # If translations exist, merge translated fields for non-English
        if lang != "en" and "translations" in content:
            trans = content.get("translations", {}).get(lang, {})
            if trans:
                lang_content.update(trans)
        pipeline.db.save_generated_content(
            product_id=product_id,
            content_type="ad",
            language=lang,
            content=lang_content,
            pamphlet_path=pamphlet_path,
            product_image_path=product_image_path,
        )

    elapsed = _time.time() - t0
    hub_url = f"/hub/{product_id}"
    print(f"[API] Product saved: {product_id} | Hub: {hub_url} | Time: {elapsed:.2f}s")

    return {
        "product_id": product_id,
        "hub_url": hub_url,
        "message": "Ad saved as product successfully",
    }


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
