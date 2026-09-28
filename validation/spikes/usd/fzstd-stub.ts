// lightusd@1.0.0-rc4 LightUSDLoader.js does `import('fzstd')` (only for useZstdCompressedWasm) but does not
// declare the dependency, so Vite's import analysis fails. The stock-loader spike aliases it here.
export function decompress(): never {
  throw new Error('fzstd is not installed (lightusd rc4 undeclared optional dependency)');
}
