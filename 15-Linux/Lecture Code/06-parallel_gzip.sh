# ----------------------------------------------------------------------------------------------------------------------
FILE="big_corpus.txt"

if [ $# -lt 1 ]; then
  echo "Usage: $0 [FILES...]"
  exit 1
fi

echo "Benchmarking sequential gzip vs. parallel gzip..."
echo "Files to compress: $@"
echo
# ----------------------------------------------------------------------------------------------------------------------
echo "==> Sequential gzip..."
start_seq=$(date +%s)
for file in "$@"; do
  gzip -k "$file"
  mv "$file.gz" "$file.sequential.gz"
done
end_seq=$(date +%s)
seq_time=$((end_seq - start_seq))
# ----------------------------------------------------------------------------------------------------------------------
for file in "$@"; do
  rm -f "$file.sequential.gz"
done
# ----------------------------------------------------------------------------------------------------------------------
echo "==> Parallel gzip (using xargs -P)..."
start_par=$(date +%s)
echo "$@" | xargs -n1 -P"$(nproc)" gzip -k
end_par=$(date +%s)
par_time=$((end_par - start_par))
# ----------------------------------------------------------------------------------------------------------------------
for file in "$@"; do
  mv "$file.gz" "$file.parallel.gz"
done

echo
echo "==> Results:"
echo "Sequential gzip time: ${seq_time}s"
echo "Parallel gzip time:   ${par_time}s"
echo
echo "Compressed files:"
ls -lh *.gz
