#!/usr/bin/env bash
set -euo pipefail

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

