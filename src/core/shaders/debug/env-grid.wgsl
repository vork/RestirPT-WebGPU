// Env orientation-grid debug overlay (plan §6 M1 "Env": lat-long lines, glTF axis markers, horizon).
// math.md#env-mapping. Composite as mix(background, rgb, alpha).
//
// - Lat-long lines every 30° drawn in the ENV frame (they rotate with the map by −γ about +Y, like the texels);
//   the equator of the map (v = 0.5) is the bright horizon line and the u = 0.5 meridian (image centre) is thicker.
// - World-fixed filled discs on the horizon at the glTF axes: +X_g red, −Z_g green, +Z_g blue, −X_g yellow;
//   a white disc at the zenith (+Y_g) and a grey one at the nadir.
// - Env-fixed rings in the same colours at the map points that face those axes when γ = 0
//   (u = 0.5, 0.25, 0.75, 0). At γ = 0 each ring encloses its disc; a rotated env moves the rings away.
#include "lights/env.wgsl"

const ENV_GRID_STEP: f32 = 0.52359877559829887308;   // 30° in radians
const ENV_GRID_HALF_WIDTH: f32 = 0.0035;             // line half-width (rad, ~0.2°)
const ENV_GRID_DISC: f32 = 0.9986295347545738;       // cos(3°): world-axis disc radius
const ENV_GRID_RING_IN: f32 = 0.9965732497555728;    // cos(4.75°)
const ENV_GRID_RING_OUT: f32 = 0.9958049276520112;   // cos(5.25°)

fn env_grid_line(dist: f32, halfWidth: f32) -> f32 {
  return 1.0 - smoothstep(halfWidth * 0.5, halfWidth, dist);
}

fn env_grid_overlay(d: vec3f, cg: f32, sg: f32) -> vec4f {
  var rgb = vec3f(0.0);
  var a = 0.0;

  // World-fixed axis discs (drawn last = on top), env-fixed rings, then grid lines underneath.
  var axes = array<vec3f, 4>(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, -1.0), vec3f(0.0, 0.0, 1.0), vec3f(-1.0, 0.0, 0.0));
  var cols = array<vec3f, 4>(vec3f(1.0, 0.1, 0.1), vec3f(0.1, 1.0, 0.1), vec3f(0.2, 0.4, 1.0), vec3f(1.0, 0.9, 0.1));
  var us = array<f32, 4>(0.5, 0.25, 0.75, 0.0);

  // Grid in the env frame: θ from the map zenith, φ = atan2(b.y, b.x) (φ = 0 ↔ u = 0.5).
  let b = envToBlender(d, cg, sg);
  let bz = clamp(b.z, -1.0, 1.0);
  let elev = asin(bz);                                   // latitude of the map, 0 on its horizon (v = 0.5)
  let sinT = sqrt(max(0.0, 1.0 - bz * bz));
  let phi = atan2(b.y, b.x);
  let dLat = abs(elev - round(elev / ENV_GRID_STEP) * ENV_GRID_STEP);
  let kLon = round(phi / ENV_GRID_STEP);
  let dLon = abs(phi - kLon * ENV_GRID_STEP) * sinT;     // great-circle distance to the nearest meridian
  let lonW = select(ENV_GRID_HALF_WIDTH, 2.0 * ENV_GRID_HALF_WIDTH, kLon == 0.0);
  let lat = env_grid_line(dLat, ENV_GRID_HALF_WIDTH);
  let lon = env_grid_line(dLon, lonW) * smoothstep(0.02, 0.08, sinT); // fade where meridians converge
  let horizon = env_grid_line(abs(elev), 2.0 * ENV_GRID_HALF_WIDTH);
  let g = max(lat, lon);
  if (g > 0.0) { rgb = vec3f(0.85); a = 0.6 * g; }
  if (horizon > 0.0) { rgb = mix(rgb, vec3f(1.0), horizon); a = max(a, 0.9 * horizon); }

  for (var i = 0u; i < 4u; i++) {
    let m = envDir(vec2f(us[i], 0.5), cg, sg);
    let c = dot(d, m);
    if (c > ENV_GRID_RING_OUT && c < ENV_GRID_RING_IN) { rgb = cols[i]; a = 1.0; }
  }
  for (var i = 0u; i < 4u; i++) {
    if (dot(d, axes[i]) > ENV_GRID_DISC) { rgb = cols[i]; a = 1.0; }
  }
  if (d.y > ENV_GRID_DISC) { rgb = vec3f(1.0); a = 1.0; }        // zenith +Y_g
  if (-d.y > ENV_GRID_DISC) { rgb = vec3f(0.4); a = 1.0; }       // nadir −Y_g
  return vec4f(rgb, a);
}
