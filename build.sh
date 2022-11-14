#!/bin/sh

echo "Compiling..."

set -e
CALLING_FROM="$PWD"
cd "$(dirname "$0")"

mkdir -p build
cd build

/usr/local/cuda-11.4/bin/nvcc -std=c++17 -w -g -G -I ../include -I ../vendor/imgui -o visualisierung ../src/main.cu -lX11 -lGL

cd "$CALLING_FROM"
echo "Finished."
