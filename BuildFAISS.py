import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm

class FAISSIndexBuilder:
    def __init__(self):
        self.index = None
        self.id_to_metadata = {}
        self.embedding_dim = 512  # CLIP base-patch32

    def build_index(self, embeddings_pkl):
        """Build FAISS index from embeddings"""
        print("\n" + "="*80)
        print("BUILDING FAISS INDEX")
        print("="*80)

        print("\n📂 Loading embeddings...")
        with open(embeddings_pkl, 'rb') as f:
            embeddings_dict = pickle.load(f)

        print(f"✓ Loaded {len(embeddings_dict):,} embeddings")

        # Extract embeddings and metadata
        print("\n📊 Extracting vectors...")
        embeddings_list = []

        for idx, (image_id, data) in enumerate(tqdm(embeddings_dict.items(), desc="Processing", ncols=80)):
            embeddings_list.append(np.array(data['embedding'], dtype=np.float32))

            self.id_to_metadata[idx] = {
                'image_id': image_id,
                'image_path': data['image_path'],
                'source': data['source'],
                'category': data['category'],
                'subcategory': data.get('subcategory', ''),
                'brand': data['brand'],
                'language': data['language'],
                'ad_type': data.get('ad_type', 'unknown')
            }

        # Convert to numpy array
        embeddings_array = np.stack(embeddings_list)
        print(f"✓ Array shape: {embeddings_array.shape}")

        print("\n🔨 Creating FAISS index...")

        # Use IndexFlatL2 for exact search (best quality)
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        print("✓ Index created")

        print("\n📥 Adding vectors to index...")
        self.index.add(embeddings_array)

        print(f"✓ Added {self.index.ntotal:,} vectors")

        return self.index

    def save_index(self, output_dir="./embeddings/faiss_indexes"):
        """Save FAISS index and metadata"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("SAVING INDEX")
        print("="*80)

        # Save FAISS index
        index_file = output_dir / "madverse_index.faiss"
        print(f"\n💾 Saving FAISS index...")
        faiss.write_index(self.index, str(index_file))
        print(f"✓ Saved to {index_file}")

        # Save metadata
        metadata_file = output_dir / "id_to_metadata.pkl"
        print(f"\n💾 Saving metadata...")
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.id_to_metadata, f)
        print(f"✓ Saved to {metadata_file}")

        # Save stats
        stats = {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'index_type': 'FlatL2',
            'metadata_count': len(self.id_to_metadata)
        }

        stats_file = output_dir / "index_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("\n" + "="*80)
        print("COMPLETE!")
        print("="*80)
        print(f"\n✅ Index vectors: {self.index.ntotal:,}")
        print(f"✅ Metadata entries: {len(self.id_to_metadata):,}")
        print(f"✅ Embedding dimension: {self.embedding_dim}")
        print(f"📁 Files saved in: {output_dir}")
        print("="*80)

if __name__ == "__main__":
    import time
    start = time.time()

    print("\n" + "="*80)
    print("FAISS INDEX BUILDER")
    print("="*80)

    builder = FAISSIndexBuilder()

    embeddings_pkl = Path("./embeddings/image_embeddings.pkl")

    if not embeddings_pkl.exists():
        print(f"\n❌ Error: {embeddings_pkl} not found!")
        print("   Run 02_embeddings_FINAL.py first")
        exit(1)

    # Build index
    builder.build_index(embeddings_pkl)

    # Save
    builder.save_index()

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
    print("\n" + "="*80)