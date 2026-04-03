"""
Image storage abstraction - cloud or local.
Copy this file to: app/storage/image_storage.py
"""

import os
from pathlib import Path
from typing import Union
from PIL import Image
from .cloudinary_manager import get_cloudinary_manager


class ImageStorage:
    """Unified storage: Cloudinary if enabled, else local files"""
    
    def __init__(self, use_cloud: bool = None):
        if use_cloud is None:
            use_cloud = os.getenv("USE_CLOUD_STORAGE", "false").lower() == "true"
        
        self.cloudinary = get_cloudinary_manager()
        self.use_cloud = use_cloud and self.cloudinary.enabled
        
        if self.use_cloud:
            print("[STORAGE] ✅ Using Cloudinary")
        else:
            print("[STORAGE] 📁 Using local storage")
    
    def save_image(self, image: Union[Image.Image, str, Path], filename: str, folder: str = "outputs") -> str:
        """Save image, return URL (cloud) or path (local)"""
        if self.use_cloud:
            try:
                public_id = Path(filename).stem
                result = self.cloudinary.upload_image(
                    image,
                    folder=f"madverse/{folder}"
                )
                return result['secure_url']
            except Exception as e:
                print(f"[STORAGE] Cloud failed: {e}, using local")
                return self._save_local(image, filename, folder)
        else:
            return self._save_local(image, filename, folder)
    
    def _save_local(self, image: Union[Image.Image, str, Path], filename: str, folder: str) -> str:
        """Save to local filesystem"""
        folders = {"outputs": "./outputs", "uploads": "./uploads", "dataset": "./data/images"}
        base_dir = Path(folders.get(folder, f"./{folder}"))
        base_dir.mkdir(parents=True, exist_ok=True)
        save_path = base_dir / filename
        
        if isinstance(image, (str, Path)):
            from shutil import copy2
            copy2(image, save_path)
        else:
            image.save(save_path)
        
        print(f"[STORAGE] 💾 Saved: {save_path}")
        return str(save_path)
    
    def load_image(self, path_or_url: str) -> Image.Image:
        """Load from URL or local path"""
        if path_or_url.startswith(('http://', 'https://')):
            return self.cloudinary.download_image(path_or_url)
        return Image.open(path_or_url)
    
    def delete_image(self, path_or_url: str) -> bool:
        """Delete from cloud or local"""
        if path_or_url.startswith(('http://', 'https://')):
            # Extract public_id from URL
            try:
                parts = path_or_url.split('/')
                upload_idx = parts.index('upload')
                public_id_parts = parts[upload_idx + 2:]
                public_id = '/'.join(public_id_parts).split('.')[0]
                return self.cloudinary.delete_image(public_id)
            except:
                return False
        else:
            try:
                Path(path_or_url).unlink(missing_ok=True)
                return True
            except:
                return False
