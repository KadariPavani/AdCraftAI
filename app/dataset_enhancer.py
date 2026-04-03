# Dataset Enhancement — copies generated pamphlets into the existing brand
# dataset folder, generates CLIP embeddings, and incrementally updates the
# FAISS index so future queries can retrieve them.

import csv
import pickle
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.models import BASE_DIR, FAISS_DIR
from app.brands import BrandMatcher

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_ENHANCEMENT_ENABLED = True
METADATA_CSV = BASE_DIR / "processed" / "metadata" / "madverse_metadata.csv"
# Fallback root when brand has no existing folder in the dataset
DEFAULT_IMAGES_ROOT = BASE_DIR / "data" / "images" / "Advert_Gallery" / "NewsPaperAds" / "Advert_Gallery"


class DatasetEnhancer:
    """Copies generated pamphlets into the brand's existing dataset folder and
    incrementally updates the FAISS index so future queries can retrieve them."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        id_to_metadata: Dict[int, dict],
        brand_matcher: Any,
        clip_model: Any,
        clip_processor: Any,
        device: str = "cpu",
    ):
        self.index = faiss_index
        self.id_to_metadata = id_to_metadata
        self.brand_matcher = brand_matcher
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def enhance(
        self,
        pamphlet_path: str,
        brand: str,
        category: str,
        subcategory: str,
        query: str = "",
        timestamp: str = "",
    ) -> Optional[Dict[str, str]]:
        """Copy pamphlet into the brand's dataset folder, update FAISS.
        Returns dataset paths dict on success, None on failure. Never raises."""
        if not DATASET_ENHANCEMENT_ENABLED:
            print(f"    [DATASET-ENH] Dataset enhancement DISABLED")
            return None

        # Normalize brand to Title_Case (prevents duplicate folders)
        original_brand = brand
        brand = BrandMatcher.normalize_brand(brand)
        if original_brand != brand:
            print(f"    [DATASET-ENH] Brand normalized: \"{original_brand}\" -> \"{brand}\"")

        print(f"    [DATASET-ENH] Starting dataset enhancement...")
        print(f"    [DATASET-ENH] Source: {pamphlet_path}")
        print(f"    [DATASET-ENH] Brand: {brand} | Category: {category}/{subcategory}")

        try:
            ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

            # Step 1 — save image into the brand's existing dataset folder
            print(f"    [DATASET-ENH] Step 1: Copying to brand dataset folder...")
            dest_path, source_folder, is_new_folder = self._save_to_dataset(
                pamphlet_path, brand, ts,
            )
            if dest_path is None:
                print(f"    [DATASET-ENH] Step 1 FAILED: Could not save to dataset")
                return None
            print(f"    [DATASET-ENH] Step 1 OK: Saved to {dest_path}")

            # Step 2 — generate CLIP embedding
            print(f"    [DATASET-ENH] Step 2: Generating CLIP embedding...")
            print(f"    [DATASET-ENH] Model: openai/clip-vit-base-patch32 (vision encoder)")
            embedding = self._generate_embedding(dest_path)
            if embedding is None:
                print(f"    [DATASET-ENH] Step 2 FAILED: Embedding generation failed")
                print(f"    [DATASET-ENH] Image saved but not indexed")
                return {
                    "image_path": str(dest_path),
                    "folder": str(dest_path.parent),
                    "new_folder_created": is_new_folder,
                }
            print(f"    [DATASET-ENH] Step 2 OK: Embedding shape={embedding.shape}")

            # Step 3 — update index + metadata (thread-safe)
            print(f"    [DATASET-ENH] Step 3: Updating FAISS index (thread-safe)...")
            new_id = self._update_index(
                embedding, dest_path, brand, category, subcategory,
                source_folder, ts, query,
            )
            print(f"    [DATASET-ENH] Step 3 OK: New vector ID={new_id} | Total vectors: {self.index.ntotal}")

            # Step 4 — append row to metadata CSV
            print(f"    [DATASET-ENH] Step 4: Appending to metadata CSV...")
            self._append_to_csv(
                dest_path, brand, category, subcategory, source_folder, ts,
            )
            print(f"    [DATASET-ENH] Step 4 OK")

            # Step 5 — persist index to disk
            print(f"    [DATASET-ENH] Step 5: Persisting FAISS index to disk...")
            self._persist_index()
            print(f"    [DATASET-ENH] Step 5 OK: Index saved to {FAISS_DIR}")

            if is_new_folder:
                print(f"    [DATASET-ENH] COMPLETE: Created NEW brand folder -> {dest_path.parent}")
            else:
                print(f"    [DATASET-ENH] COMPLETE: Added to EXISTING folder -> {dest_path.parent}")
            print(f"    [DATASET-ENH] File: {dest_path.name} | ID: {new_id} | Brand: {brand} | {category}/{subcategory}")
            return {
                "image_path": str(dest_path),
                "folder": str(dest_path.parent),
                "new_folder_created": is_new_folder,
                "index_id": str(new_id),
                "brand": brand,
                "category": category,
                "subcategory": subcategory,
            }

        except Exception as exc:
            print(f"    [DATASET-ENH] Non-blocking ERROR: {type(exc).__name__}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_brand_name(self, brand: str) -> str:
        """Resolve a brand name to its canonical display name in the index.
        Handles case variations like 'kalyan jewellers' -> 'Kalyan_Jewellers'."""
        # Direct match
        if brand in self.brand_matcher.brand_index:
            return brand
        # Canonical lookup
        canonical = BrandMatcher._canonical(brand)
        existing = self.brand_matcher._canonical_to_name.get(canonical)
        if existing and existing in self.brand_matcher.brand_index:
            return existing
        return brand

    def _find_brand_folder(self, brand: str) -> Optional[Path]:
        """Look up an existing image for this brand in id_to_metadata and
        return its parent directory so we can place new images alongside it."""
        resolved = self._resolve_brand_name(brand)
        if resolved in self.brand_matcher.brand_index:
            indices = self.brand_matcher.brand_index[resolved]["indices"]
            for idx in indices:
                meta = self.id_to_metadata.get(idx)
                if meta:
                    img_path = Path(meta.get("image_path", ""))
                    # Handle relative paths (stored relative to BASE_DIR)
                    if not img_path.is_absolute():
                        img_path = BASE_DIR / img_path
                    if img_path.parent.exists():
                        return img_path.parent
        return None

    def _get_source_folder(self, brand: str) -> str:
        """Return the source_folder value from an existing entry for this brand."""
        resolved = self._resolve_brand_name(brand)
        if resolved in self.brand_matcher.brand_index:
            indices = self.brand_matcher.brand_index[resolved]["indices"]
            for idx in indices:
                meta = self.id_to_metadata.get(idx)
                if meta and meta.get("source_folder"):
                    return meta["source_folder"]
        return "Advert_Gallery"

    def _save_to_dataset(
        self, src_path: str, brand: str, ts: str,
    ) -> tuple:
        """Copy pamphlet into the brand's existing dataset folder.
        Returns (dest_path, source_folder, is_new_folder) or (None, None, None)."""
        src = Path(src_path)
        if not src.exists():
            return None, None, None

        # Always use normalized Title_Case brand for consistent folder names
        safe_brand = BrandMatcher.normalize_brand(brand)

        # Try to find the brand's existing folder in the dataset
        dest_dir = self._find_brand_folder(brand)
        source_folder = self._get_source_folder(brand)
        is_new_folder = False

        if dest_dir is None:
            # Brand not in dataset yet — create a new folder alongside others
            dest_dir = DEFAULT_IMAGES_ROOT / safe_brand
            dest_dir.mkdir(parents=True, exist_ok=True)
            source_folder = "Advert_Gallery"
            is_new_folder = True

        filename = f"{safe_brand}_gen_{ts}{src.suffix}"
        dest = dest_dir / filename
        shutil.copy2(str(src), str(dest))
        return dest, source_folder, is_new_folder

    def _generate_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """Generate a CLIP embedding for the image at *image_path*."""
        try:
            img = Image.open(str(image_path)).convert("RGB")
            inputs = self.clip_processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                image_out = self.clip_model.vision_model(pixel_values=pixel_values)
                pooled = image_out[1]
                features = self.clip_model.visual_projection(pooled)
                features = F.normalize(features, p=2, dim=-1)
            return features.cpu().numpy().flatten().astype(np.float32)
        except Exception as exc:
            print(f"  [DatasetEnhancer] Embedding error: {exc}")
            return None

    def _update_index(
        self,
        embedding: np.ndarray,
        image_path: Path,
        brand: str,
        category: str,
        subcategory: str,
        source_folder: str,
        ts: str,
        query: str,
    ) -> int:
        """Add vector to FAISS and update id_to_metadata + brand_matcher (thread-safe)."""
        # Ensure brand is normalized before storing
        brand = BrandMatcher.normalize_brand(brand)
        with self._lock:
            new_id = self.index.ntotal
            self.index.add(np.array([embedding]))

            rel_path = str(image_path)
            self.id_to_metadata[new_id] = {
                "image_path": rel_path,
                "image_id": image_path.stem,
                "brand": brand,
                "category": category,
                "subcategory": subcategory,
                "language": "english",
                "ad_type": "generated",
                "source": "adgal",
                "source_folder": source_folder,
                "image_filename": image_path.name,
                "original_path": rel_path,
                "timestamp": ts,
                "query": query,
            }

            # Update brand_matcher so the new image is retrievable by brand
            if brand and brand.lower() not in ("", "nan", "unknown"):
                canonical = brand.strip().lower().replace(" ", "_")
                # Find existing display name for this canonical key
                existing_name = self.brand_matcher._canonical_to_name.get(canonical)
                if existing_name and existing_name in self.brand_matcher.brand_index:
                    # Merge into existing brand (case-insensitive)
                    self.brand_matcher.brand_index[existing_name]["indices"].append(new_id)
                    self.brand_matcher.brand_index[existing_name]["count"] += 1
                elif brand in self.brand_matcher.brand_index:
                    # Exact match already exists
                    self.brand_matcher.brand_index[brand]["indices"].append(new_id)
                    self.brand_matcher.brand_index[brand]["count"] += 1
                else:
                    # Completely new brand
                    brand_lower = canonical.replace("_", "")
                    self.brand_matcher.brand_names.append(brand)
                    self.brand_matcher.brand_names_lower.append(brand_lower)
                    self.brand_matcher.brand_names_with_spaces.append(
                        brand.lower().replace("_", " ")
                    )
                    self.brand_matcher._canonical_to_name[canonical] = brand
                    self.brand_matcher.brand_index[brand] = {
                        "indices": [new_id],
                        "category": category,
                        "subcategory": subcategory,
                        "count": 1,
                    }

        return new_id

    def _append_to_csv(
        self, image_path: Path, brand: str, category: str, subcategory: str,
        source_folder: str, ts: str,
    ) -> None:
        """Append a row to the metadata CSV, matching existing column format."""
        if not METADATA_CSV.exists():
            return
        # Normalize brand for consistent CSV entries
        brand = BrandMatcher.normalize_brand(brand)
        row = {
            "source": "adgal",
            "source_folder": source_folder,
            "image_id": image_path.stem,
            "image_filename": image_path.name,
            "category": category,
            "subcategory": subcategory,
            "brand": brand,
            "language": "english",
            "ad_type": "generated",
            "original_path": str(image_path),
            "image_path": str(image_path),
        }
        with self._lock:
            with open(str(METADATA_CSV), "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)

    def _persist_index(self) -> None:
        """Write FAISS index + id_to_metadata to disk atomically.
        Uses temp files + rename for atomic writes to prevent partial persistence."""
        with self._lock:
            try:
                # Write to temporary files first
                temp_faiss = FAISS_DIR / "madverse_index.faiss.tmp"
                temp_meta = FAISS_DIR / "id_to_metadata.pkl.tmp"
                
                # Write FAISS index
                faiss.write_index(self.index, str(temp_faiss))
                
                # Write metadata
                with open(str(temp_meta), "wb") as f:
                    pickle.dump(self.id_to_metadata, f)
                
                # Atomic rename (both or neither - prevents partial writes)
                final_faiss = FAISS_DIR / "madverse_index.faiss"
                final_meta = FAISS_DIR / "id_to_metadata.pkl"
                
                temp_faiss.replace(final_faiss)
                temp_meta.replace(final_meta)
                
            except Exception as e:
                # Clean up temp files on error
                if temp_faiss.exists():
                    temp_faiss.unlink()
                if temp_meta.exists():
                    temp_meta.unlink()
                raise e
