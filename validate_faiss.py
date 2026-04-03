"""
Validate FAISS index and metadata consistency.
This script checks if all metadata indices are valid for the FAISS index.
"""
import pickle
import faiss
from pathlib import Path

def validate():
    print("\n" + "="*80)
    print("FAISS INDEX VALIDATION")
    print("="*80)
    
    # Load FAISS index
    index_path = Path("./embeddings/faiss_indexes/madverse_index.faiss")
    metadata_path = Path("./embeddings/faiss_indexes/id_to_metadata.pkl")
    
    print(f"\n📂 Loading FAISS index from: {index_path}")
    index = faiss.read_index(str(index_path))
    print(f"✓ FAISS index loaded: {index.ntotal:,} vectors")
    
    print(f"\n📂 Loading metadata from: {metadata_path}")
    with open(metadata_path, 'rb') as f:
        id_to_metadata = pickle.load(f)
    print(f"✓ Metadata loaded: {len(id_to_metadata):,} entries")
    
    # Check metadata keys
    print(f"\n🔍 Analyzing metadata keys...")
    keys = list(id_to_metadata.keys())
    min_key = min(keys)
    max_key = max(keys)
    expected_keys = set(range(len(id_to_metadata)))
    actual_keys = set(keys)
    
    print(f"   Min key: {min_key}")
    print(f"   Max key: {max_key}")
    print(f"   Expected range: 0 to {len(id_to_metadata)-1}")
    print(f"   Expected keys: {len(expected_keys)}")
    print(f"   Actual keys: {len(actual_keys)}")
    
    # Check for gaps
    missing_keys = expected_keys - actual_keys
    extra_keys = actual_keys - expected_keys
    
    if missing_keys:
        print(f"\n❌ PROBLEM: Missing keys in metadata: {len(missing_keys)}")
        print(f"   First 10 missing: {sorted(missing_keys)[:10]}")
    
    if extra_keys:
        print(f"\n❌ PROBLEM: Extra keys beyond expected range: {len(extra_keys)}")
        print(f"   First 10 extra: {sorted(extra_keys)[:10]}")
    
    # Check if any key exceeds FAISS index size
    invalid_keys = [k for k in keys if k >= index.ntotal]
    if invalid_keys:
        print(f"\n❌ CRITICAL: {len(invalid_keys)} metadata keys >= FAISS index size ({index.ntotal})")
        print(f"   This will cause 'key < ntotal' errors!")
        print(f"   First 10 invalid keys: {sorted(invalid_keys)[:10]}")
    else:
        print(f"\n✅ All metadata keys are valid (< {index.ntotal})")
    
    # Check brands
    print(f"\n🔍 Checking brand indices...")
    brand_issues = []
    for idx, meta in id_to_metadata.items():
        if idx >= index.ntotal:
            brand = meta.get('brand', 'Unknown')
            brand_issues.append((idx, brand))
    
    if brand_issues:
        print(f"❌ {len(brand_issues)} metadata entries point to invalid FAISS indices")
        print(f"   Sample issues (idx, brand):")
        for idx, brand in brand_issues[:10]:
            print(f"      Index {idx} (brand: {brand}) >= FAISS size {index.ntotal}")
    else:
        print(f"✅ All brand indices are valid")
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    if not missing_keys and not extra_keys and not invalid_keys:
        print("✅ FAISS index and metadata are CONSISTENT")
        print("✅ No issues found!")
    else:
        print("❌ INCONSISTENCIES DETECTED:")
        if missing_keys:
            print(f"   - {len(missing_keys)} missing keys in metadata")
        if extra_keys:
            print(f"   - {len(extra_keys)} extra keys in metadata")
        if invalid_keys:
            print(f"   - {len(invalid_keys)} invalid keys (>= FAISS size)")
        print("\n💡 RECOMMENDATION: Rebuild FAISS index and metadata using BuildFAISS.py")
    print("="*80)

if __name__ == "__main__":
    validate()
