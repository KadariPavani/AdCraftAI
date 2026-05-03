import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

class MAdVerseDataLoaderOptimized:
    def __init__(self, data_path="./data"):
        self.data_path = Path(data_path)
        self.annotations_path = self.data_path / "annotations"
        self.images_path = self.data_path / "images"
        self.file_index = {}  # Fast lookup: filename -> full_path

    def build_file_index(self):
        """Build complete file index once (FAST!)"""
        print("\n" + "="*80)
        print("Building file index (one-time scan)...")
        print("="*80)

        file_count = 0
        for source_folder in ['Advert_Gallery', 'OnlineAds', 'Epaper1', 'Epaper2']:
            search_dir = self.images_path / source_folder

            if not search_dir.exists():
                print(f"⚠️  Folder not found: {search_dir}")
                continue

            print(f"\nScanning {source_folder}...")

            for root, dirs, files in os.walk(search_dir):
                for filename in files:
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = str(Path(root) / filename)
                        self.file_index[filename.lower()] = full_path
                        file_count += 1

            print(f"  ✓ Found {file_count} images so far")

        print(f"\n✓ Total files indexed: {len(self.file_index)}")
        print("="*80)

    def load_annotations(self):
        """Load all annotation files"""
        annotations = {}

        print("\n" + "="*80)
        print("Loading annotations...")
        print("="*80 + "\n")

        for json_file in self.annotations_path.glob("*.json"):
            source_name = json_file.stem.replace("_annotation", "").replace("_annot_j", "")
            print(f"Loading {source_name}...", end=" ")

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                annotations[source_name] = data
                print(f"✓ ({len(data)} items)")

        return annotations

    def create_metadata_df(self, annotations):
        """Convert annotations to structured DataFrame"""
        rows = []

        print("\n" + "="*80)
        print("Processing annotations...")
        print("="*80 + "\n")

        for source, data in annotations.items():
            print(f"Processing {source} ({len(data)} items)...")

            for item in tqdm(data, desc=f"  {source}", ncols=80):
                row = self._extract_row_from_madverse_item(item, source)
                if row:
                    rows.append(row)

        df = pd.DataFrame(rows)
        print(f"\n✓ Created DataFrame with {len(df)} rows")

        # Fast image path lookup using pre-built index
        print("\nLinking image paths (using index)...")
        df['image_path'] = df['image_filename'].apply(self._fast_find_image)

        # Remove rows where images don't exist
        before_count = len(df)
        df = df[df['image_path'].notna()]
        found_count = len(df)
        missing_count = before_count - found_count

        print(f"✓ Found {found_count:,} images")
        print(f"✓ Missing {missing_count:,} images ({missing_count/before_count*100:.1f}%)")

        return df

    def _extract_row_from_madverse_item(self, item, source):
        """Extract row from MAdVerse annotation format"""

        # Extract filename from img_path
        img_path = item.get('img_path', '')

        # Handle both forward and backward slashes
        img_path = img_path.replace('\\\\', '/').replace('\\', '/')
        filename = Path(img_path).name

        # Extract hierarchical annotation
        hier_annot = item.get('hier_annot', [])
        category = hier_annot[0] if len(hier_annot) > 0 else 'unknown'
        subcategory = hier_annot[1] if len(hier_annot) > 1 else ''
        brand = hier_annot[2] if len(hier_annot) > 2 else ''

        # Map source to folder name
        source_folder = self._map_source_to_folder(source)

        # Create image_id from filename (without extension)
        image_id = Path(filename).stem

        row = {
            'source': source,
            'source_folder': source_folder,
            'image_id': image_id,
            'image_filename': filename,
            'category': category,
            'subcategory': subcategory,
            'brand': brand,
            'language': item.get('language', 'unknown'),
            'ad_type': item.get('ad_type', 'unknown'),
            'original_path': img_path
        }

        return row

    def _map_source_to_folder(self, annotation_source):
        """Map annotation source to actual image folder"""
        mapping = {
            'adgal': 'Advert_Gallery',
            'epaper1': 'Epaper1',
            'epaper2': 'Epaper2',
            'web': 'OnlineAds'
        }
        return mapping.get(annotation_source, annotation_source)

    def _fast_find_image(self, filename):
        """Fast O(1) lookup using pre-built index"""
        return self.file_index.get(filename.lower())

    def save_metadata(self, df, output_path="./processed/metadata/madverse_metadata.csv"):
        """Save metadata to CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"\n✓ Metadata saved to: {output_path}")
        print(f"✓ Total images processed: {len(df):,}")

        if len(df) > 0:
            print(f"\n📊 Breakdown by source:")
            print(df['source'].value_counts())

            print(f"\n📊 Breakdown by category (top 10):")
            print(df['category'].value_counts().head(10))

            print(f"\n📊 Breakdown by language:")
            print(df['language'].value_counts())

            print(f"\n📊 Breakdown by ad_type:")
            print(df['ad_type'].value_counts())

        return df

# Main execution
if __name__ == "__main__":
    import time
    start_time = time.time()

    print("\n" + "="*80)
    print("MAdVerse Dataset Loader (OPTIMIZED)")
    print("="*80)

    loader = MAdVerseDataLoaderOptimized()

    # Step 1: Build file index (fast one-time scan)
    loader.build_file_index()

    # Step 2: Load annotations
    annotations = loader.load_annotations()

    # Step 3: Create metadata DataFrame
    df = loader.create_metadata_df(annotations)

    # Step 4: Save metadata
    if len(df) > 0:
        loader.save_metadata(df)

        print(f"\n📋 Sample metadata:")
        print(df[['source', 'category', 'brand', 'language', 'image_filename']].head(15))

        print(f"\n✅ Columns: {list(df.columns)}")
        print(f"✅ Shape: {df.shape}")
    else:
        print("\n⚠️  No images found. Check your image folders.")

    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("\n" + "="*80)