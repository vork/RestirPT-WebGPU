// ReSTIR debug-view and probe hooks (restir-api.md §2.11): view ids 400–499, probe tags 64–95.
// OWNER WP-D. P0: empty hook bodies (no views). Each hook is a no-op unless its view is active or the pixel is the probe
// pixel; with DEBUG_NO_BINDINGS (test pipelines without G3) every hook compiles to an empty body.
fn rsdbg_reservoir(px: vec2u, ai: u32, tap: u32) { }
fn rsdbg_candidate(px: vec2u, d: u32, tech: u32, k: u32, w: f32, lumF: f32, counter: u32, selected: bool) { }
fn rsdbg_vertex(px: vec2u, path: u32, b: u32, pos: vec3f, lobeCode: u32) { }
fn rsdbg_slot(px: vec2u, s: u32, code: u32, J: f32, replayed: bool) { }
fn rsdbg_accept(px: vec2u, mask: u32) { }
fn rsdbg_mis(px: vec2u, k: u32, mc: f32, sumM: f32, lumRel: f32, s: u32, mj: f32, wj: f32, sel: u32) { }
