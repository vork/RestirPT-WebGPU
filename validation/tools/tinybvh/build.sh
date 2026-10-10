#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
pin=4b8509fd26b29801c8f79386cf8a1ce5713070fb
upstream=validation/out/tinybvh-upstream
mkdir -p validation/out/tinybvh-probe
if [ ! -d "$upstream/.git" ]; then
  git clone https://github.com/jbikker/tinybvh.git "$upstream"
  git -C "$upstream" checkout --detach "$pin"
fi
if [ "$(git -C "$upstream" rev-parse HEAD)" != "$pin" ]; then
  echo "tinybvh checkout must be $pin; refusing to overwrite an existing checkout" >&2
  exit 1
fi
if ! git -C "$upstream" diff --quiet HEAD; then
  echo "tinybvh checkout has tracked modifications; refusing an unpinned build" >&2
  exit 1
fi
em++ validation/tools/tinybvh/builder.cpp -I "$upstream" -O3 -std=c++17 \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=33554432 -sMAXIMUM_MEMORY=2147483648 \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,node \
  -sEXPORTED_FUNCTIONS='["_malloc","_free"]' -sEXPORTED_RUNTIME_METHODS='["HEAPF32","HEAPU32"]' \
  -o validation/out/tinybvh-probe/builder-raw.mjs
