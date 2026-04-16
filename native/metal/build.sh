#!/usr/bin/env bash
set -euo pipefail

rm -R bin
mkdir bin

clang++ \
  -std=c++17 \
  -shared \
  -fobjc-arc \
  -x objective-c++ \
  metal_add.mm \
  -framework Foundation \
  -framework Metal \
  -o bin/libdotllmmetal.dylib

# Dans build.sh
xcrun -sdk macosx metal -c add.metal -o bin/add.air
xcrun -sdk macosx metallib bin/add.air -o add.metallib
