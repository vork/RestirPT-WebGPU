// Compiles dumped Dawn MSL files and prints the Metal pipeline limits (a register-pressure proxy):
//   swift pso-stats.swift [--strip] [--fast] file.metal ...
// --strip removes [[max_total_threads_per_threadgroup(N)]] so the hardware/register limit shows.
import Foundation
import Metal
let dev = MTLCreateSystemDefaultDevice()!
var args = Array(CommandLine.arguments.dropFirst())
let strip = args.contains("--strip"); let fast = args.contains("--fast")
args.removeAll { $0.hasPrefix("--") }
print("device \(dev.name) maxThreadsPerThreadgroup \(dev.maxThreadsPerThreadgroup) strip=\(strip) fast=\(fast)")
for path in args {
  var src = try! String(contentsOfFile: path, encoding: .utf8)
  if strip { src = src.replacingOccurrences(of: #"\[\[max_total_threads_per_threadgroup\(\d+\)\]\]"#, with: "", options: .regularExpression) }
  let opts = MTLCompileOptions()
  if #available(macOS 15.0, *) { opts.mathMode = fast ? .fast : .safe } else { opts.fastMathEnabled = fast }
  let t0 = Date()
  do {
    let lib = try dev.makeLibrary(source: src, options: opts)
    guard let name = lib.functionNames.first(where: { $0.hasPrefix("dawn_entry_point") }) ?? lib.functionNames.first,
          let fn = lib.makeFunction(name: name) else { print("\(path): no function"); continue }
    let t1 = Date()
    let pso = try dev.makeComputePipelineState(function: fn)
    let t2 = Date()
    print(String(format: "%-28@ maxTPT %4d  simd %2d  tgmem %6d  msl->air %.2fs  pso %.2fs", (path as NSString).lastPathComponent, pso.maxTotalThreadsPerThreadgroup, pso.threadExecutionWidth, pso.staticThreadgroupMemoryLength, t1.timeIntervalSince(t0), t2.timeIntervalSince(t1)))
  } catch { print("\(path): \(error)") }
}
