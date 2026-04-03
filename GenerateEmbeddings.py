import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import pickle
import json
import gc
import os
import torch.nn.functional as F

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is not set — please set it before running.")

class CLIPEmbedderWorking:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        print("\n" + "="*80)
        print("CLIP EMBEDDING GENERATOR - ACTUALLY WORKING")
        print("="*80)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n📱 Device: {self.device.upper()}")

        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️  CPU mode (slower)")

        print(f"\n⏳ Loading model: {model_name}")

        print("   Step 1/2: Loading processor...", end=" ", flush=True)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("✓")

        print("   Step 2/2: Loading model...", end=" ", flush=True)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✓")

        print("\n✅ Model ready!")
        print("="*80)

    def test_single_image(self, image_path):
        """Test embedding a single image"""
        print(f"\n🧪 Testing with: {Path(image_path).name}")

        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            print(f"   ✓ Image loaded: {image.size}")

            # Process with CLIP
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding - FIXED VERSION
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # Extract tensor from the output object
                image_features = outputs if isinstance(outputs, torch.Tensor) else outputs.last_hidden_state

                # If still not a tensor, try accessing pooler_output
                if not isinstance(image_features, torch.Tensor):
                    image_features = outputs.pooler_output

                # Normalize
                image_features = F.normalize(image_features, p=2, dim=-1)

            embedding = image_features.cpu().numpy().flatten()
            print(f"   ✓ Embedding shape: {embedding.shape}")
            print(f"   ✓ Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")
            print(f"   ✓ Norm: {np.linalg.norm(embedding):.4f} (should be ~1.0)")

            return True

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def embed_single(self, image_path):
        """Embed single image"""
        try:
            image = Image.open(image_path).convert('RGB')

            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # Extract actual tensor
                image_features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
                # Normalize
                image_features = F.normalize(image_features, p=2, dim=-1)

            return image_features.cpu().numpy().flatten()

        except Exception as e:
            return None

    def process_dataset(self, metadata_csv, output_dir="./embeddings", save_every=1000):
        """Process all images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("LOADING METADATA")
        print("="*80)

        df = pd.read_csv(metadata_csv)
        print(f"\n✓ Loaded {len(df):,} images")

        # Test first image
        print("\n" + "="*80)
        print("TESTING MODEL")
        print("="*80)

        test_success = self.test_single_image(df.iloc[0]['image_path'])

        if not test_success:
            print("\n❌ Model test failed!")
            return {}

        print("\n✅ Test passed! Processing all images...")

        # Check checkpoint
        checkpoint_file = output_dir / "embeddings_checkpoint.pkl"
        start_idx = 0
        embeddings_dict = {}

        if checkpoint_file.exists():
            print(f"\n📂 Found checkpoint")
            response = input("   Resume? (y/n): ").lower()
            if response == 'y':
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    embeddings_dict = checkpoint['embeddings']
                    start_idx = checkpoint['last_index']
                print(f"   ✓ Resuming from {start_idx:,}")

        print("\n" + "="*80)
        print(f"PROCESSING {len(df):,} IMAGES")
        print("="*80 + "\n")

        success_count = len(embeddings_dict)
        failed_count = 0

        # Process with progress bar
        for idx in tqdm(range(start_idx, len(df)), desc="Embedding", unit="img", ncols=80):
            row = df.iloc[idx]

            embedding = self.embed_single(row['image_path'])

            if embedding is not None:
                embeddings_dict[row['image_id']] = {
                    'embedding': embedding.tolist(),
                    'image_path': row['image_path'],
                    'source': row['source'],
                    'category': row['category'],
                    'subcategory': row.get('subcategory', ''),
                    'brand': row['brand'],
                    'language': row['language'],
                    'ad_type': row.get('ad_type', 'unknown')
                }
                success_count += 1
            else:
                failed_count += 1

            # Checkpoint
            if (idx + 1) % save_every == 0:
                checkpoint = {
                    'embeddings': embeddings_dict,
                    'last_index': idx + 1
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint, f)

                rate = (success_count / (idx + 1)) * 100
                print(f"\n💾 {idx + 1:,}/{len(df):,} | ✅ {success_count:,} ({rate:.1f}%) | ❌ {failed_count}")

            # Memory cleanup
            if (idx + 1) % 100 == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # Final save
        print("\n\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)

        embedding_file = output_dir / "image_embeddings.pkl"
        print(f"\n💾 Saving {len(embeddings_dict):,} embeddings...")

        with open(embedding_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)

        print(f"✓ Saved to {embedding_file}")

        # Metadata
        metadata = {
            'total_embeddings': len(embeddings_dict),
            'total_attempted': len(df),
            'failed_images': failed_count,
            'embedding_dim': 512,
            'model_name': 'openai/clip-vit-base-patch32',
            'success_rate': f"{len(embeddings_dict)/len(df)*100:.2f}%"
        }

        metadata_file = output_dir / "embedding_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Cleanup
        if checkpoint_file.exists():
            checkpoint_file.unlink()

        print("\n" + "="*80)
        print("COMPLETE!")
        print("="*80)
        print(f"\n✅ Processed: {len(embeddings_dict):,} / {len(df):,}")
        print(f"❌ Failed: {failed_count:,}")
        print(f"📊 Success: {len(embeddings_dict)/len(df)*100:.1f}%")
        print(f"📁 File: {embedding_file}")
        print(f"💾 Size: {embedding_file.stat().st_size / 1024 / 1024:.1f} MB")
        print("="*80)

        return embeddings_dict

if __name__ == "__main__":
    import time
    start = time.time()

    embedder = CLIPEmbedderWorking()

    embeddings = embedder.process_dataset(
        metadata_csv="./processed/metadata/madverse_metadata.csv",
        output_dir="./embeddings",
        save_every=1000
    )

    elapsed = time.time() - start

    print(f"\n⏱️  Time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hrs)")

    if len(embeddings) > 0:
        rate = len(embeddings) / elapsed
        print(f"⚡ Speed: {rate:.2f} images/sec")

    print("\n" + "="*80)