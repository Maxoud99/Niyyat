#!/bin/bash
# Wrapper script to run clustering comparison from any directory

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the scripts directory
cd "$SCRIPT_DIR/scripts"

# Run the Python script
python3 compare_clustering_algorithms.py "$@"
