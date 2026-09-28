// Shared constants and small helpers. Never write IEEE Inf/NaN literals (const-eval error) and never compute
// with Inf under Chrome's relaxed Metal math (plan §1.7): use FLT_MAX or explicit flags.
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWO_PI: f32 = 0.15915494309189533577;
const FLT_MAX: f32 = 3.40282346638528859812e38;

fn sqr(x: f32) -> f32 { return x * x; }
fn luminance(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
