#!/bin/bash
set -e

echo "Building rby1-sdk..."

# Ensure conanfile.py is modified
if ! grep -q "protobuf/3.21.12" conanfile.py; then
    echo "ERROR: Please modify conanfile.py as described in BUILD_INSTRUCTIONS.md"
    exit 1
fi

# Clean
conan remove "*" -c
rm -rf build

# Install dependencies
conan install . -s build_type=Release -b missing -of build

# Build
cd build
cmake .. -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE=./conan_toolchain.cmake \
    -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
    -DCMAKE_BUILD_TYPE=Release

cmake --build .

echo "Build successful! Run 'sudo make install' to install."
