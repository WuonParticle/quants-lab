#!/bin/bash

# Parse arguments
FORCE_REPACKAGE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-repackage)
            FORCE_REPACKAGE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get the absolute paths
QUANTS_LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUMMINGBOT_DIR="$(dirname "$QUANTS_LAB_DIR")/hummingbot"
WHEELS_DIR="$QUANTS_LAB_DIR/wheels"

# Check for existing wheel in Hummingbot dist directory
cd "$HUMMINGBOT_DIR" || exit 1
EXISTING_WHEEL=$(ls "dist/hummingbot-"*.whl 2>/dev/null)
if [ -n "$EXISTING_WHEEL" ] && [ "$FORCE_REPACKAGE" = false ]; then
    echo "Found existing wheel: $EXISTING_WHEEL"
    WHEEL_FILE="$EXISTING_WHEEL"
else
    if [ -n "$EXISTING_WHEEL" ]; then
        echo "Force repackage requested, rebuilding wheel..."
        rm -f "$EXISTING_WHEEL"
        echo "Removed existing wheel: $EXISTING_WHEEL"
    fi

    # Build Hummingbot wheel
    echo "Building Hummingbot wheel..."
    python setup.py sdist bdist_wheel

    # Find the wheel file
    WHEEL_FILE=$(ls "dist/hummingbot-"*.whl 2>/dev/null)
    if [ -z "$WHEEL_FILE" ]; then
        echo "Error: No wheel file found in dist directory"
        exit 1
    fi
fi

# Copy wheel to quants-lab/wheels
echo "Copying $(basename "$WHEEL_FILE") to wheels directory..."
cp "$WHEEL_FILE" "$WHEELS_DIR/"
WHEEL_FILE="$WHEELS_DIR/$(basename "$WHEEL_FILE")"

# Install new wheel
echo "Installing $(basename "$WHEEL_FILE")..."
conda run -n quants-lab pip install --force-reinstall "$WHEEL_FILE"

echo -e "\nSuccessfully installed $(basename "$WHEEL_FILE")" 