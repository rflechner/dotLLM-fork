#!/usr/bin/env bash
set -euo pipefail

rm -R bin
mkdir bin

DYLIB="bin/libdotllmmetal.dylib"

clang++ \
  -std=c++17 \
  -shared \
  -fobjc-arc \
  -x objective-c++ \
  bridge.mm \
  -framework Foundation \
  -framework Metal \
  -framework MetalPerformanceShaders \
  -o "$DYLIB"

for metal_file in *.metal; do
    name="${metal_file%.metal}"
    xcrun -sdk macosx metal -c "$metal_file" -o "bin/${name}.air"
    xcrun -sdk macosx metallib "bin/${name}.air" -o "bin/${name}.metallib"
done

# ── Build summary ────────────────────────────────────────────────────────────
echo
echo "── Build summary ────────────────────────────────────────────"

# 1) Dylib size (KB) — quick sanity check that linking succeeded.
size_kb=$(($(stat -f%z "$DYLIB") / 1024))
echo "dylib              : $DYLIB (${size_kb} KB)"

# 2) Architecture (arm64 / x86_64 / universal).
arch=$(otool -hv "$DYLIB" | awk 'NR==4 {print $2}')
echo "architecture       : $arch"

# 3) Frameworks linked — filtered to what we care about. A missing framework
#    here means the build silently dropped a -framework flag.
echo "frameworks linked  :"
otool -L "$DYLIB" \
  | awk '/Metal|Foundation/ {print "    "$1}'

# 4) Exported entry points — every dotllm_metal_* symbol must be present,
#    otherwise P/Invoke fails at runtime with EntryPointNotFoundException.
exports=$(nm -gU "$DYLIB" 2>/dev/null | grep -c "_dotllm_metal_" || true)
echo "exported symbols   : $exports dotllm_metal_* entry points"

# 5) Metal libraries — one .metallib per .metal source.
metallib_count=$(ls bin/*.metallib 2>/dev/null | wc -l | tr -d ' ')
metal_count=$(ls *.metal 2>/dev/null | wc -l | tr -d ' ')
echo "metal libraries    : $metallib_count / $metal_count compiled"

if [ "$metallib_count" -ne "$metal_count" ]; then
    echo "  WARNING: mismatch between .metal sources and compiled .metallib files"
fi

echo "─────────────────────────────────────────────────────────────"
