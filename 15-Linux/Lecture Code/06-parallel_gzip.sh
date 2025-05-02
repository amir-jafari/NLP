#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------------------------------------------------
FILE="corpus.txt"

# 1) Check if corpus.txt exists; if not, download and unzip
if [ ! -f "$FILE" ]; then
  echo "Could not find $FILE, downloading corpus.zip..."
  wget -O 'corpus.zip' 'https://gwu.box.com/shared/static/keps7nlt0dyc01ldp1vrsykh8wpn6v20.zip'
  echo "Unzipping corpus.zip..."
  unzip corpus.zip
  echo "$FILE downloaded and unzipped."
else
  echo "$FILE already exists, skipping download."
fi

# 2) If no arguments are passed to the script, default to corpus.txt
if [ $# -eq 0 ]; then
  set -- "corpus.txt"
fi

# 3) Benchmarking sequential vs. parallel gzip
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
