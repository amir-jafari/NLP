1- Run the download shell script file. 

./download_wiki_dump.sh en
# -----------------------------------------------------------
2- install wiki extractor

sudo pip3 install pip install wikiextractor
# -----------------------------------------------------------
3- Run the wikiextractor in the terminal

./extract_and_clean_wiki_dump.sh enwikilatest-pages-articles.xml.bz2
# -----------------------------------------------------------
4- Install Blingfire

sudo pip3 install -U blingfire
# -----------------------------------------------------------
5- Run the python file.

python3 preprocess_wiki_dump.py enwiki-latest-pages-articles.txt
# -----------------------------------------------------------
-- Linux commands
Link to split :https://www.geeksforgeeks.org/split-command-in-linux-with-examples/
split --verbose -b500M file.txt file.
# -----------------------------------------------------------
-- Linux commands
cat enwiki-latest-pages-articles_preprocessed.txt | awk 'BEGIN {srand()} !/^$/ { if (rand() <= .2) print $0}' > train_r.txt