#!/usr/bin/env python3

import json
import sys
import os

def update_files(instructions_file):
    # Load the instructions from the JSON file
    try:
        with open(instructions_file, 'r') as f:
            instructions = json.load(f)
    except Exception as e:
        print(f"Error reading instructions file: {e}")
        sys.exit(1)

    base_path = instructions.get("base_path")
    files = instructions.get("files", [])

    if not base_path or not files:
        print("Invalid instructions: 'base_path' and 'files' are required.")
        sys.exit(1)

    # Process each file
    for file_info in files:
        relative_path = file_info.get("relative_path")
        new_content = file_info.get("new_content")

        if not relative_path or new_content is None:
            print(f"Invalid file entry: {file_info}")
            continue

        # Resolve the full path
        full_path = os.path.join(base_path, relative_path)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Write the new content to the file
        try:
            with open(full_path, 'w') as f:
                f.write(new_content)
            print(f"Updated file: {full_path}")
        except Exception as e:
            print(f"Error writing to file {full_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_files.py <instructions_file.json>")
        sys.exit(1)

    instructions_file = sys.argv[1]
    update_files(instructions_file)
