"""
Cloudinary integration for cloud image storage.
Copy this file to: app/storage/cloudinary_manager.py
"""

import os
import io
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import requests

try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
    from cloudinary.utils import cloudinary_url
    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False


class CloudinaryManager:
    """Manages image upload/download to/from Cloudinary"""
    
    def __init__(self):
        self.enabled = False
        self.cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "")
        self.api_key = os.getenv("CLOUDINARY_API_KEY", "")
        self.api_secret = os.getenv("CLOUDINARY_API_SECRET", "")
        
        if not CLOUDINARY_AVAILABLE:
            print("[CLOUDINARY] ❌ Package not installed. Run: pip install cloudinary")
            return
        
        if not all([self.cloud_name, self.api_key, self.api_secret]):
            print("[CLOUDINARY] ⚠️  Not configured - add to .env:")
            print("  CLOUDINARY_CLOUD_NAME=your-cloud-name")
            print("  CLOUDINARY_API_KEY=your-api-key")
            print("  CLOUDINARY_API_SECRET=your-api-secret")
            return
        
        cloudinary.config(
            cloud_name=self.cloud_name,
            api_key=self.api_key,
            api_secret=self.api_secret,
            secure=True
        )
        
        self.enabled = True
        print(f"[CLOUDINARY] ✅ Configured: {self.cloud_name}")
    
    def upload_image(self, image: Union[str, Path, Image.Image], folder: str = "madverse") -> dict:
        """Upload image, return {'url', 'secure_url', 'public_id'}"""
        if not self.enabled:
            raise ValueError("Cloudinary not enabled")
        
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            image_data = buffer
        else:
            image_data = str(image)
        
        result = cloudinary.uploader.upload(
            image_data,
            folder=folder,
            resource_type="image"
        )
        
        print(f"[CLOUDINARY] ✅ Uploaded: {result['secure_url']}")
        return {
            'url': result['url'],
            'secure_url': result['secure_url'],
            'public_id': result['public_id'],
            'format': result['format'],
            'width': result['width'],
            'height': result['height']
        }
    
    def download_image(self, url: str) -> Image.Image:
        """Download image from Cloudinary URL"""
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    
    def delete_image(self, public_id: str) -> bool:
        """Delete image from Cloudinary"""
        if not self.enabled:
            return False
        result = cloudinary.uploader.destroy(public_id)
        return result.get('result') == 'ok'


_cloudinary_manager = None

def get_cloudinary_manager() -> CloudinaryManager:
    global _cloudinary_manager
    if _cloudinary_manager is None:
        _cloudinary_manager = CloudinaryManager()
    return _cloudinary_manager
