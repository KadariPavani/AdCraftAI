"""
Fix FAISS Index Mismatch
========================
This script fixes the 'key < ntotal' error by removing metadata entries
that don't have corresponding FAISS vectors.

Problem: Metadata has 61,630 entries but FAISS index only has 61,628 vectors
Solution: Remove metadata entries with keys >= 61,628
"""

import pickle
import faiss
from pathlib import Path
import shutil
from datetime import datetime

def fix_faiss_mismatch():
    print("\n" + "="*80)
    print("FIXING FAISS INDEX MISMATCH")
    print("="*80)
    
    # Paths
    index_path = Path("./embeddings/faiss_indexes/madverse_index.faiss")
    metadata_path = Path("./embeddings/faiss_indexes/id_to_metadata.pkl")
    
    # Backup original files
    backup_suffix = f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metadata_backup = metadata_path.with_suffix(metadata_path.suffix + backup_suffix)
    
    print(f"\n📂 Creating backup: {metadata_backup.name}")
    shutil.copy2(metadata_path, metadata_backup)
    print(f"✅ Backup created")
    
    # Load FAISS index
    print(f"\n📂 Loading FAISS index...")
    index = faiss.read_index(str(index_path))
    print(f"✅ FAISS index: {index.ntotal:,} vectors")
    
    # Load metadata
    print(f"\n📂 Loading metadata...")
    with open(metadata_path, 'rb') as f:
        id_to_metadata = pickle.load(f)
    print(f"✅ Metadata: {len(id_to_metadata):,} entries")
    
    # Find invalid entries
    print(f"\n🔍 Checking for invalid metadata keys...")
    invalid_keys = [k for k in id_to_metadata.keys() if k >= index.ntotal]
    
    if not invalid_keys:
        print(f"✅ No invalid keys found! Metadata is already consistent.")
        print(f"   All {len(id_to_metadata)} metadata keys < {index.ntotal}")
        return
    
    print(f"❌ Found {len(invalid_keys)} invalid metadata entries")
    print(f"   Invalid keys: {sorted(invalid_keys)}")
    
    # Show what will be removed
    print(f"\n📋 Metadata to be removed:")
    for key in sorted(invalid_keys):
        meta = id_to_metadata[key]
        brand = meta.get('brand', 'Unknown')
        category = meta.get('category', 'Unknown')
        print(f"   Key {key}: {brand} - {category}")
    
    # Remove invalid entries
    print(f"\n🔧 Removing invalid metadata entries...")
    for key in invalid_keys:
        del id_to_metadata[key]
    print(f"✅ Removed {len(invalid_keys)} entries")
    
    # Verify fix
    remaining_invalid = [k for k in id_to_metadata.keys() if k >= index.ntotal]
    if remaining_invalid:
        print(f"❌ ERROR: Still have {len(remaining_invalid)} invalid keys!")
        return
    
    print(f"✅ All remaining metadata keys are valid")
    print(f"   Metadata entries: {len(id_to_metadata):,}")
    print(f"   FAISS vectors: {index.ntotal:,}")
    print(f"   Match: {'YES ✅' if len(id_to_metadata) == index.ntotal else 'NO (but all keys are valid)'}")
    
    # Save fixed metadata
    print(f"\n💾 Saving fixed metadata to: {metadata_path}")
    with open(metadata_path, 'wb') as f:
        pickle.dump(id_to_metadata, f)
    print(f"✅ Fixed metadata saved")
    
    print("\n" + "="*80)
    print("FIX SUMMARY")
    print("="*80)
    print(f"✅ Fixed FAISS mismatch successfully!")
    print(f"   Before: {len(id_to_metadata) + len(invalid_keys):,} metadata entries, {index.ntotal:,} vectors")
    print(f"   After:  {len(id_to_metadata):,} metadata entries, {index.ntotal:,} vectors")
    print(f"   Removed: {len(invalid_keys)} invalid entries")
    print(f"   Backup: {metadata_backup.name}")
    print(f"\n✅ The 'key < ntotal' error should be fixed!")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        fix_faiss_mismatch()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
