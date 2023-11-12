#!/bin/bash

# Check if a directory is provided as an argument
if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <directory>"
	exit 1
fi

this_folder="$(dirname "$0")"
directory="$1"

# Check if the provided directory exists
if [ ! -d "$directory" ]; then
	echo "Error: Directory '$directory' not found."
	exit 1
fi

"$this_folder"/scripts/rename_images.sh "$directory"
python3 "$this_folder"/scripts/resize_images.py "$directory" "$directory (resized)"
python3 "$this_folder"/scripts/binary.py "$directory (resized)" "$directory (final)"

history="$directory (final)/history"
mkdir "$history"
mv "$directory" "$history/$(basename $directory) (original)"
mv "$directory (resized)" "$history"

