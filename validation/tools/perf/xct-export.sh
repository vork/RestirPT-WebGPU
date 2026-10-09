#!/bin/zsh
# xct-export.sh <trace> <outprefix> [schema ...] : exports the xctrace tables used by xct_groups.py / xct_split.py /
# xct_spill.py (default: all of them, plus the per-profile gpu-counter-info tables); with schemas, only those.
# (perf2 tools, docs/decisions/perf2-api.md §4; promoted from validation/out/perf2-profile/tools)
tr=$1; pre=$2; shift 2
tables=(${@:-metal-gpu-intervals metal-application-encoders-list graphics-compiler-spill-events gpu-performance-device-state-intervals gpu-performance-state-intervals device-thermal-state-intervals metal-shader-profiler-shader-list gpu-shader-profiler-interval gpu-shader-profiler-sample metal-gpu-counter-intervals gpu-counter-value})
for t in $tables; do
  xcrun xctrace export --input $tr --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"$t\"]" > ${pre}$t.xml 2>/dev/null
  echo "$t rows=$(grep -c '<row' ${pre}$t.xml)"
done
if [ $# -eq 0 ]; then
  for p in 1 2 3 4 5 6 7 8 9 10 11 12; do
    xcrun xctrace export --input $tr --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"gpu-counter-info\" and @counter-profile=\"$p\"]" > ${pre}gpu-counter-info-$p.xml 2>/dev/null
    n=$(grep -c '<row' ${pre}gpu-counter-info-$p.xml); [ "$n" -gt 0 ] && echo "gpu-counter-info profile $p rows=$n"
  done
fi
true
