# RBY1-SDK C++ Build Instructions (Ubuntu/Linux)

This guide provides working build instructions for rby1-sdk v0.8.3 and v0.9.1 on Ubuntu Linux systems.

## Prerequisites
```bash
# Install Conan package manager
pip install conan

# Install build essentials
sudo apt-get update
sudo apt-get install build-essential cmake git
```

## Step 1: Clone the Repository
```bash
git clone --recurse-submodules https://github.com/RainbowRobotics/rby1-sdk.git
cd rby1-sdk
git checkout v0.8.3  # or v0.9.1
```

## Step 2: Configure Conan Profile

Create a default Conan profile with compiler settings:
```bash
conan profile detect
```

Edit the profile to ensure compiler is properly set:
```bash
nano ~/.conan2/profiles/default
```

The profile should look like this (adjust gcc version to match your system):
```ini
[settings]
arch=x86_64
build_type=Release
os=Linux
compiler=gcc
compiler.version=11
compiler.libcxx=libstdc++11
compiler.cppstd=17
```

## Step 3: Fix Protobuf Version Mismatch

The SDK has pre-generated protobuf files that are incompatible with the default protobuf 5.x pulled by grpc 1.72.0. We need to force both the protobuf library and protoc compiler to use version 3.21.12.

Edit `conanfile.py`:
```bash
nano conanfile.py
```

Modify the `requirements()` method and add a new `build_requirements()` method:
```python
def requirements(self):
    self.requires("protobuf/3.21.12", override=True)  # Force this version everywhere
    self.requires("grpc/1.54.3")
    self.requires("eigen/3.4.0")
    self.requires("tinyxml2/10.0.0", visible=False)
    self.requires("nlohmann_json/3.11.3")

def build_requirements(self):
    self.tool_requires("protobuf/3.21.12")  # Force protoc to match library version
```

## Step 4: Clean Previous Builds (if any)
```bash
# Remove any cached packages
conan remove "*" -c

# Remove build artifacts
rm -rf build
```

## Step 5: Install Dependencies
```bash
conan install . -s build_type=Release -b missing -of build
```

## Step 6: Configure CMake

Use the traditional CMake method (works with CMake 3.26+):
```bash
cd build
cmake .. -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE=./conan_toolchain.cmake \
    -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
    -DCMAKE_BUILD_TYPE=Release
```

## Step 7: Build
```bash
cmake --build .
```

## Step 8: Install
```bash
sudo make install
```

## Step 9: Install python binding
```bash
pip install -e .
```

## Troubleshooting

### Issue: `conan install` fails with "Pkg folder must exist"

**Solution:** Clean the Conan cache completely:
```bash
conan remove "*" -c
```

### Issue: Protobuf version errors during compilation

**Solution:** Ensure you've modified `conanfile.py` to force protobuf 3.21.12 for both library and build tools (Step 3).

### Issue: "No compiler was detected"

**Solution:** Make sure you've edited the Conan profile to include compiler settings (Step 2).

### Issue: CMake preset errors (CMake < 3.23)

**Solution:** Use the traditional CMake build method shown in Step 6 instead of `cmake --preset conan-release`.

## Alternative: Python SDK

If you only need Python bindings, the installation is much simpler:
```bash
pip install rby1-sdk
```

Or from source:
```bash
pip install -e .
```

## Version Compatibility

- Tested with CMake 3.26.0
- Tested with GCC 11
- Tested with Conan 2.x
- Working with SDK versions v0.8.3 and v0.9.1

## Notes

- The key issue is protobuf version compatibility. The SDK's pre-generated protobuf files require protobuf 3.x, while newer dependencies pull protobuf 5.x by default.
- Using `grpc/1.54.3` and `protobuf/3.21.12` provides a stable build environment.
- The traditional CMake build method (without presets) is more reliable across different CMake versions.
