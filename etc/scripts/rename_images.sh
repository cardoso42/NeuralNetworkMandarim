#!/bin/bash

# Check if a directory is provided as an argument
if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <directory>"
	exit 1
fi

directory="$1"

# Check if the provided directory exists
if [ ! -d "$directory" ]; then
	echo "Error: Directory '$directory' not found."
	exit 1
fi

# Navigate to the specified directory
cd "$directory" || exit

# Initialize a counter
counter=1

# Iterate over each file in the directory
for file in *; do
	# Check if it is a regular file
	if [ -f "$file" ]; then
		# Extract file extension
		extension="${file##*.}"

		# Create the new file name with the counter and extension
		new_name="$(printf "%04d" "$counter").${extension}"

		# Rename the file
		mv "$file" "$new_name"

		if [ "$extension" != "png" ]; then			
			# Use mogrify to convert the file to PNG format
			mogrify -format png "$new_name"

			# Remove the original file
			rm -f "$new_name"
		fi

		# Increment the counter for the next file
		((counter++))
	fi
done

echo "Files renamed successfully."

