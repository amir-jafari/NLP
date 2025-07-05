#!/bin/bash

# ----------------------------------------------------------------------------------------------------------------------
pos_words=(good great excellent amazing awesome fantastic love happy)
neg_words=(bad terrible awful horrible worst hate sad angry)

pos_emoji="😀 😃 😊 🤩 ❤️ 👍"
neg_emoji="😠 😡 😞 😢 👎"
# ----------------------------------------------------------------------------------------------------------------------
score_keywords () {
  local line=$1
  local p=0 n=0
  for w in "${pos_words[@]}"; do
    p=$(( p + $(grep -oiw "$w" <<<"$line" | wc -l) ))
  done
  for w in "${neg_words[@]}"; do
    n=$(( n + $(grep -oiw "$w" <<<"$line" | wc -l) ))
  done
  [[ $p -gt $n ]] && echo "POS" || [[ $n -gt $p ]] && echo "NEG" || echo "NEU"
}

score_emoji () {
  local line=$1
  p=$(grep -o -F -f <(tr ' ' '\n' <<<"$pos_emoji") <<<"$line" | wc -l)
  n=$(grep -o -F -f <(tr ' ' '\n' <<<"$neg_emoji") <<<"$line" | wc -l)
  [[ $p -gt $n ]] && echo "POS" || [[ $n -gt $p ]] && echo "NEG" || echo "NEU"
}

score_exclaim () {
  local ex=$(grep -o "!" <<<"$1" | wc -l)
  # Example rules:
  (( ex >= 3 )) && echo "POS" || (( ex == 1 )) && echo "NEG" || echo "NEU"
}

vote () {
  # Majority vote among the three scores
  if [[ $1 == "$2" || $1 == "$3" ]]; then
    echo "$1"
  elif [[ $2 == "$3" ]]; then
    echo "$2"
  else
    echo "NEU"
  fi
}
# ----------------------------------------------------------------------------------------------------------------------
# 3) Perform sentiment analysis and show results on terminal
echo "Sentiment Analysis Results:"
echo "Label | Line"
echo "----- | ----"

pos=0; neg=0; neu=0

while IFS= read -r line; do
  kw=$(score_keywords "$line")
  em=$(score_emoji    "$line")
  ex=$(score_exclaim  "$line")
  label=$(vote "$kw" "$em" "$ex")

  case $label in
    POS) ((pos++));;
    NEG) ((neg++));;
    *)   ((neu++));;
  esac

  echo "$label | $line"
done < demo_corpus.txt

echo -e "\n── summary ──────────────────────────"
echo "Positive : $pos"
echo "Negative : $neg"
echo "Neutral  : $neu"