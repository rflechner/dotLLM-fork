#!/usr/bin/env bash
set -euo pipefail

rm -R bin
mkdir bin

clang++ \
  -std=c++17 \
  -shared \
  -fobjc-arc \
  -x objective-c++ \
  bridge.mm \
  -framework Foundation \
  -framework Metal \
  -o bin/libdotllmmetal.dylib

for metal_file in *.metal; do
    name="${metal_file%.metal}"
    xcrun -sdk macosx metal -c "$metal_file" -o "bin/${name}.air"
    xcrun -sdk macosx metallib "bin/${name}.air" -o "bin/${name}.metallib"
done
