#!/usr/bin/env python3
"""Manually clean all batch files from OpenAI's file storage."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import openai

def main():
    files = openai.files.list()
    batch_files = [f for f in files.data if f.purpose in ("batch", "batch_output")]

    if not batch_files:
        print("No batch files found in OpenAI storage.")
        return

    print(f"Found {len(batch_files)} batch files. Deleting...")
    deleted = 0
    for f in batch_files:
        try:
            openai.files.delete(f.id)
            print(f"  Deleted {f.id} ({f.purpose}, {f.bytes or 0} bytes)")
            deleted += 1
        except Exception as e:
            print(f"  Failed to delete {f.id}: {e}")

    print(f"\nDone. Deleted {deleted}/{len(batch_files)} files.")


if __name__ == "__main__":
    main()
