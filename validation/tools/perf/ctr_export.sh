#!/bin/zsh
# ctr_export.sh <trace> <outprefix> : TOC + every table needed for per-kernel GPU counters (MST + Metal GPU Counters)
tr=$1; pre=$2
xcrun xctrace export --input $tr --toc > ${pre}toc.xml 2>&1
for t in metal-gpu-intervals metal-application-encoders-list graphics-compiler-spill-events metal-gpu-counter-intervals gpu-counter-value \
         metal-shader-profiler-shader-list metal-shader-profiler-intervals gpu-shader-profiler-interval gpu-shader-profiler-sample gpu-aps-stream \
         metal-gpu-info device-gpu-info gpu-performance-state-intervals metal-gpu-state-intervals; do
  xcrun xctrace export --input $tr --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"$t\"]" > ${pre}$t.xml 2>${pre}$t.err
  echo "$t rows=$(grep -c '<row' ${pre}$t.xml) $(head -c 200 ${pre}$t.err)"
done
# counter-profile tables carry attributes; export each one present in the TOC
for p in $(grep -o 'schema="gpu-counter-info" counter-profile="[0-9]*"' ${pre}toc.xml | grep -o '[0-9]*"$' | tr -d '"' | sort -u); do
  xcrun xctrace export --input $tr --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"gpu-counter-info\" and @counter-profile=\"$p\"]" > ${pre}gpu-counter-info-$p.xml 2>/dev/null
  echo "gpu-counter-info profile $p rows=$(grep -c '<row' ${pre}gpu-counter-info-$p.xml)"
done
true
