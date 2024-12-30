#!/bin/bash

# Description: This script recursively deletes all node_modules folders in a given directory.
# Usage: ./clean_node_modules.sh /path/to/directory

# Check if a path argument is provided
if [ -z "$1" ]; then
  echo "❌ Usage: $0 /path/to/directory"
  exit 1
fi

# Validate if the path exists
TARGET_DIR="$1"
if [ ! -d "$TARGET_DIR" ]; then
  echo "❌ Error: Directory '$TARGET_DIR' does not exist."
  exit 1
fi

# Find and delete node_modules folders
echo "🚀 Searching for 'node_modules' folders in '$TARGET_DIR'..."
find "$TARGET_DIR" -type d -name "node_modules" -prune -print -exec rm -rf '{}' +

echo "✅ All 'node_modules' folders have been removed from '$TARGET_DIR'."
