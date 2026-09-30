// Suffix refresh of a stored record under a frame's light / env state (restir-temporal-api.md §3.5, TD9; math.md
// #light-changes [M5 addition]). OWNER T-C. refresh_record(ai, fsFrom, fsTo) evaluates the prefix-independent end terms
// of record resIn[ai] (own frame fsFrom) under frame fsTo: translated entry, J_P, deep suffix radiance, (b)/(c) cache
// values and the N1 end visibility; the passes store the result as SfxRec (tframe.wgsl sfx_store).
// P0 STUB: every record is SXS_DONE | SXS_UNDEF (on TF_REFRESH frames every temporal sample and every canonical inverse
// is undefined: class zeroing of all classes, unbiased).
#include "restir/tframe.wgsl"
#include "restir/reservoir.wgsl"

fn refresh_record(ai: u32, fsFrom: u32, fsTo: u32) -> SfxRec {
  var r: SfxRec;
  r.status = SXS_DONE | SXS_UNDEF;
  r.entryTo = RC_NONE;
  r.jp = 1.0;
  r.gen = rsTemporal.frameGen;
  return r;
}
