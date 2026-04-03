#!/usr/bin/env python3
"""
Execute validation and capture full output.
"""
import sys
import os

# Change to the madverse directory
os.chdir('P:\\KHUB\\madverse')
sys.path.insert(0, 'P:\\KHUB\\madverse')

# Now run the validation
from validate_faiss import validate

if __name__ == "__main__":
    try:
        validate()
    except Exception as e:
        print(f"\n❌ ERROR during validation: {e}")
        import traceback
        traceback.print_exc()
