#!/bin/bash
# Run MINS on one NCLT session streamed straight from the NCLT bucket (nothing stored), once per sensor set,
# all sets fed from the one stream, then score each trajectory against the ground truth.
#
#   bench.sh <mins_bag> <session> <out dir> <set>...      set = letters of I C G L W, e.g. ICGLW ILW IW
#
# Cameras 1-5 are streamed only if some set has C; that needs <session>_lb3.idx.gz (nclt-index.yml) here.
set -eo pipefail
mins=$(realpath "$1") seq=$2 out=$(realpath -m "$3"); shift 3
here=$(dirname "$(realpath "$0")")
mkdir -p "$out/work"

cams=()
case "$*" in *C*) cams=(--index "$seq"_lb3.idx.gz --cams 1,2,3,4,5) ;; esac
sinks=()
for set in "$@"; do
  # a copy of the config with this set's sensors switched on and outputs in $out
  cfg=$out/cfg_$set
  rm -rf "$cfg" && cp -r "$here/configs/nclt" "$cfg"
  sed -i "s|path_trajectory:.*|path_trajectory: \"$out/$set.txt\"|; s|path_timing:.*|path_timing: \"$out/$set.time\"|; s|verbosity:.*|verbosity: 3|" "$cfg/config_system.yaml"
  for s in C:camera G:gps L:lidar W:wheel; do
    on=false; [[ $set == *${s%:*}* ]] && on=true
    sed -i "0,/enabled:/s|enabled:.*|enabled: $on|" "$cfg/config_${s#*:}.yaml"
  done
  # set +e: the subshell inherits -e and would skip writing the exit code when mins_bag fails
  sinks+=(">(set +e; \"$mins\" \"$cfg/config.yaml\" /dev/stdin > \"$out/$set.log\" 2>&1; echo \$? > \"$out/$set.exit\")")
done

# one stream, one mins_bag per set; tee waits for the slowest and keeps feeding the rest if one dies
rm -f "$out"/*.exit
eval "python3 '$here/nclt.py' '$seq' '$out/work' --stream ${cams[*]} | tee -p ${sinks[*]} > /dev/null"
for set in "$@"; do until [ -f "$out/$set.exit" ]; do sleep 5; done; done
for set in "$@"; do
  python3 "$here/nclt_ate.py" "$out/work/$seq.txt" "$out/$set.txt" "$seq" "$set" "$(cat "$out/$set.exit" 2>/dev/null)"
done | tee "$out/score.jsonl"
