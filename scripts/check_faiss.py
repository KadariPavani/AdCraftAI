#!/usr/bin/env python3
"""
FAISS Index Validation Script
Checks if FAISS index and embeddings are properly downloaded from Git LFS.
Run this after cloning to verify setup.
"""

import sys
from pathlib import Path

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'=' * 70}{RESET}")
    print(f"{BOLD}{BLUE}  {text}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def check_git_lfs_pointers(file_path):
    """Check if a file is a Git LFS pointer (not the actual file)"""
    if not file_path.exists():
        return True, "File does not exist"
    
    # Git LFS pointer files are small (< 200 bytes) and start with "version https://git-lfs.github.com"
    size = file_path.stat().st_size
    
    if size < 200:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                if first_line.startswith("version https://git-lfs.github.com"):
                    return True, "Git LFS pointer file (not downloaded)"
        except:
            pass
    
    return False, None

def check_faiss_index():
    """Main validation function"""
    print_header("MAdVerse FAISS Index Validator")
    
    base_dir = Path(__file__).parent.parent.resolve()
    print(f"Project root: {base_dir}\n")
    
    # Files to check
    files_to_check = {
        "FAISS Index": base_dir / "embeddings" / "faiss_indexes" / "madverse_index.faiss",
        "Metadata Pickle": base_dir / "embeddings" / "faiss_indexes" / "id_to_metadata.pkl",
        "Index Stats": base_dir / "embeddings" / "faiss_indexes" / "index_stats.json",
        "Embedding Metadata": base_dir / "embeddings" / "embedding_metadata.json",
        "Dataset Metadata CSV": base_dir / "processed" / "metadata" / "madverse_metadata.csv",
    }
    
    # Expected minimum sizes (in bytes)
    expected_sizes = {
        "FAISS Index": 100 * 1024 * 1024,  # 100 MB minimum
        "Metadata Pickle": 1 * 1024 * 1024,  # 1 MB minimum
        "Index Stats": 100,  # 100 bytes minimum
        "Embedding Metadata": 100,  # 100 bytes minimum
        "Dataset Metadata CSV": 1 * 1024 * 1024,  # 1 MB minimum
    }
    
    all_ok = True
    issues = []
    
    print(f"{BOLD}Checking critical files:{RESET}\n")
    
    for name, file_path in files_to_check.items():
        # Check if file exists
        if not file_path.exists():
            print_error(f"{name}: NOT FOUND")
            print(f"   Expected at: {file_path}")
            all_ok = False
            issues.append(f"{name} is missing")
            continue
        
        # Check if it's a Git LFS pointer
        is_pointer, msg = check_git_lfs_pointers(file_path)
        if is_pointer:
            print_error(f"{name}: {msg}")
            print(f"   Location: {file_path}")
            all_ok = False
            issues.append(f"{name} is a Git LFS pointer, not the actual file")
            continue
        
        # Check file size
        size = file_path.stat().st_size
        size_mb = size / (1024 * 1024)
        min_size = expected_sizes.get(name, 0)
        
        if size < min_size:
            print_warning(f"{name}: TOO SMALL ({size_mb:.2f} MB)")
            print(f"   Expected at least {min_size / (1024*1024):.0f} MB")
            all_ok = False
            issues.append(f"{name} is too small (corrupted or incomplete)")
        else:
            print_success(f"{name}: OK ({size_mb:.2f} MB)")
    
    # Additional validation: Try to load FAISS index
    print(f"\n{BOLD}Testing FAISS index integrity:{RESET}\n")
    
    try:
        import faiss
        import pickle
        
        index_path = files_to_check["FAISS Index"]
        metadata_path = files_to_check["Metadata Pickle"]
        
        if index_path.exists() and not check_git_lfs_pointers(index_path)[0]:
            try:
                index = faiss.read_index(str(index_path))
                print_success(f"FAISS index loaded: {index.ntotal:,} vectors, dimension {index.d}")
            except Exception as e:
                print_error(f"Failed to load FAISS index: {e}")
                all_ok = False
                issues.append("FAISS index is corrupted")
        
        if metadata_path.exists() and not check_git_lfs_pointers(metadata_path)[0]:
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print_success(f"Metadata loaded: {len(metadata):,} entries")
            except Exception as e:
                print_error(f"Failed to load metadata: {e}")
                all_ok = False
                issues.append("Metadata pickle is corrupted")
    
    except ImportError:
        print_warning("FAISS not installed, skipping integrity test")
        print("   (This is OK if running before Docker build)")
    
    # Final report
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    
    if all_ok:
        print(f"\n{GREEN}{BOLD}✓ ALL CHECKS PASSED!{RESET}")
        print(f"\n{GREEN}Your setup is ready. You can run:{RESET}")
        print(f"  • {BOLD}python run.py{RESET} (local)")
        print(f"  • {BOLD}docker-compose up -d{RESET} (Docker)\n")
        return 0
    else:
        print(f"\n{RED}{BOLD}✗ VALIDATION FAILED!{RESET}")
        print(f"\n{RED}Issues found:{RESET}")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print(f"\n{YELLOW}{BOLD}SOLUTION:{RESET}")
        print(f"\n{YELLOW}The FAISS index files are stored in Git LFS.{RESET}")
        print(f"{YELLOW}You need to pull them with:{RESET}\n")
        print(f"  {BOLD}git lfs install{RESET}")
        print(f"  {BOLD}git lfs pull{RESET}\n")
        print(f"{YELLOW}If you don't have Git LFS installed:{RESET}")
        print(f"  • Windows: {BOLD}https://git-lfs.github.com/{RESET}")
        print(f"  • Linux: {BOLD}sudo apt-get install git-lfs{RESET}")
        print(f"  • Mac: {BOLD}brew install git-lfs{RESET}\n")
        
        return 1

if __name__ == "__main__":
    sys.exit(check_faiss_index())
