# ----------------------------------------------------------------------------------------------------------------------
set -euo pipefail

FILE=${1:-big_corpus.txt}

if [ ! -f "$FILE" ]; then
  echo "Could not find $FILE, downloading big_corpus.zip..."
  wget -O 'big_corpus.zip' 'https://gwu.box.com/shared/static/54k67360xyp7f3s8zuq5j1v6nktl8awp.zip'
  echo "Unzipping big_corpus.zip..."
  unzip big_corpus.zip
  echo "$FILE downloaded and unzipped."
else
  echo "$FILE already exists, skipping download."
fi

PATTERN="transformer"
N=5

# ----------------------------------------------------------------------------------------------------------------------
SLOW_BASELINE_ONLY_ONCE=true
SKIP_BASELINE=false
# ----------------------------------------------------------------------------------------------------------------------
echo "Benchmarking on: $FILE"
echo "Searching for:   $PATTERN"
echo "Repetitions (fast tests): $N"
$SKIP_BASELINE || $SLOW_BASELINE_ONLY_ONCE && echo "(baseline will run once)"
echo "───────────────────────────────────────────────"
# ----------------------------------------------------------------------------------------------------------------------
median_time () {
  local cmd="$1"; local reps="${2:-$N}"
  { for _ in $(seq "$reps"); do
        /usr/bin/time -f "%e" bash -c "$cmd" 2>&1
    done; } |
    sort -n | awk 'NR==(NR+1)/2 {printf "%.4f s\n", $1}'
}
# ----------------------------------------------------------------------------------------------------------------------
if ! $SKIP_BASELINE; then
  baseline_cmd="
    count=0
    while IFS= read -r line; do
        [[ \$line == *$PATTERN* ]] && ((count++))
    done < $FILE
    echo \$count >/dev/null
  "
  baseline_reps=1; $SLOW_BASELINE_ONLY_ONCE || baseline_reps=$N
  printf '%-32s %s\n' "Bash while-read loop:" \
      "$(median_time "$baseline_cmd" "$baseline_reps")"
fi
# ----------------------------------------------------------------------------------------------------------------------
printf '%-32s %s\n' "grep:" \
    "$(median_time "grep -c \"$PATTERN\" \"$FILE\" >/dev/null")"
# ----------------------------------------------------------------------------------------------------------------------
printf '%-32s %s\n' "grep via pipe to wc:" \
    "$(median_time "grep \"$PATTERN\" \"$FILE\" | wc -l >/dev/null")"
# ----------------------------------------------------------------------------------------------------------------------
split -n l/8 -d "$FILE" "${FILE}.chunk."   # makes *.chunk.00 …
loop_chunks_cmd='
  for f in '"${FILE}.chunk."*'; do
      grep -c "'"$PATTERN"'" "$f" >/dev/null || true
  done
'
printf '%-32s %s\n' "Looping grep per chunk:" \
    "$(median_time "$loop_chunks_cmd")"
printf '%-32s %s\n' "One-shot grep on chunks:" \
    "$(median_time "grep -c \"$PATTERN\" ${FILE}.chunk.* >/dev/null")"
# ----------------------------------------------------------------------------------------------------------------------
fanout_cmd='
  printf "%s\n" '"${FILE}.chunk."*' |
  xargs -P8 -I{} bash -c "grep -c \"'"$PATTERN"'\" {} >/dev/null || true"
'
printf '%-32s %s\n' "grep + xargs -P (8 cores):" \
    "$(median_time "$fanout_cmd")"
# ----------------------------------------------------------------------------------------------------------------------
printf '%-32s %s\n' "Built-in arithmetic demo:" \
    "$(median_time 's=0; for i in {1..1000000}; do : $((s+=i)); done')"

rm -f "${FILE}.chunk."*
