#define TINYBVH_IMPLEMENTATION
#define TINYBVH_NO_SIMD
#define NO_THREADED_BUILDS
#define NO_DOUBLE_PRECISION_SUPPORT
#include "tiny_bvh.h"
#include <emscripten.h>
static tinybvh::BVH* tree = nullptr;
extern "C" {
EMSCRIPTEN_KEEPALIVE void build(float* vertices, unsigned vertexCount, unsigned* indices, unsigned triCount, unsigned mode) {
 delete tree; tree = new tinybvh::BVH();
 tree->settings.useSIMDifavailable = false;
 tree->settings.useSpatialSplits = mode == 2;
 tree->settings.postOptimize = mode == 1;
 tree->Build(tinybvh::bvhvec4slice((tinybvh::bvhvec4*)vertices, vertexCount), indices, triCount);

}
EMSCRIPTEN_KEEPALIVE void* nodes() { return tree->bvhNode; }
EMSCRIPTEN_KEEPALIVE unsigned nodeCount() { return tree->usedNodes; }
EMSCRIPTEN_KEEPALIVE void* prims() { return tree->primIdx; }
EMSCRIPTEN_KEEPALIVE unsigned indexCount() { return tree->idxCount; }
EMSCRIPTEN_KEEPALIVE void dispose() { delete tree; tree=nullptr; }
}
