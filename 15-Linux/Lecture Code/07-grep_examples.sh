set -euo pipefail
FILE=${1:-small_corpus.txt}
PATTERN="transformer"
# ----------------------------------------------------------------------------------------------------------------------
echo "Demo file : $FILE"
echo "Search term: $PATTERN"
echo "--------------------------------------------------"

# ----------------------------------------------------------------------------------------------------------------------
#  1. Basic search
echo -e "\n1) Basic search:"
grep "$PATTERN" "$FILE"
# ----------------------------------------------------------------------------------------------------------------------
#  2. Case-insensitive
echo -e "\n2) Case-insensitive (-i):"
grep -i "$PATTERN" "$FILE"
# ----------------------------------------------------------------------------------------------------------------------
#  3. Multiple files (showing we can reuse the same pattern)
echo -e "\n3) Multiple files:"
cp "$FILE" copy1.txt; cp "$FILE" copy2.txt
grep "$PATTERN" "$FILE" copy1.txt copy2.txt
rm copy1.txt copy2.txt
# ----------------------------------------------------------------------------------------------------------------------
#  4. Recursive through a directory
echo -e "\n4) Recursive (-r) through ./demo_dir/:"
mkdir -p demo_dir && cp "$FILE" demo_dir/a.txt && cp "$FILE" demo_dir/b.txt
grep -r "$PATTERN" demo_dir/
rm -r demo_dir
# ----------------------------------------------------------------------------------------------------------------------
#  5. Show line numbers
echo -e "\n5) With line numbers (-n):"
grep -n "$PATTERN" "$FILE"
# ----------------------------------------------------------------------------------------------------------------------
#  6. Count matches
echo -e "\n6) Count lines that match (-c):"
grep -c "$PATTERN" "$FILE"
# ----------------------------------------------------------------------------------------------------------------------
#  7. Invert the match
echo -e "\n7) Invert match (-v):"
grep -v "$PATTERN" "$FILE" | head -n 5   # first five lines that *don’t* match
# ----------------------------------------------------------------------------------------------------------------------
#  8. Filter another command’s output
echo -e "\n8) Filtering ‘cat’ output:"
cat "$FILE" | grep "$PATTERN"
# ----------------------------------------------------------------------------------------------------------------------
#  9. Regular-expression power
echo -e "\n9) Regex example (words ending in 'ing'):"
grep -E '\b[[:alpha:]]+ing\b' "$FILE" | head -n 5
# ----------------------------------------------------------------------------------------------------------------------
# 10. Whole-word match
echo -e "\n10) Whole word only (-w):"
grep -w "$PATTERN" "$FILE"
# ----------------------------------------------------------------------------------------------------------------------