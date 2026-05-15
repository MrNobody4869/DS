# Practical No. 7: Text Analytics
# 1. Apply document preprocessing: Tokenization, POS Tagging, stop words removal, Stemming, and Lemmatization.
# 2. Create representation of documents by calculating TF-IDF.

import nltk
import re
import math
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# -------------------------------------------------
# Download required NLTK packages (run once)
# -------------------------------------------------

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# -------------------------------------------------
# Initialize Text
# -------------------------------------------------

text = "Tokenization is the first step in text analytics. The process of breaking down a text paragraph into smaller chunks such as words or sentences is called Tokenization."

# -------------------------------------------------
# Step 1: Sentence Tokenization
# -------------------------------------------------

# sent_tokenize(): Splits a paragraph into individual sentences using punctuation rules.
sentences = sent_tokenize(text)
print("\nSentences:", sentences)

# -------------------------------------------------
# Step 2: Word Tokenization
# -------------------------------------------------

# word_tokenize(): Splits text into individual word tokens (including punctuation).
words = word_tokenize(text)
print("\nWords:", words)

# -------------------------------------------------
# Step 3: POS Tagging
# -------------------------------------------------

# pos_tag(): Labels each token with its grammatical role (NN=Noun, VB=Verb, JJ=Adj, etc.)
print("\nPOS Tags:", pos_tag(words))

# -------------------------------------------------
# Step 4: Text Cleaning
# -------------------------------------------------

# re.sub('[^a-zA-Z]', ' ', text): Removes everything that is NOT a letter (removes numbers, punctuation).
# .lower(): Converts all characters to lowercase for consistency.
text_clean = re.sub('[^a-zA-Z]', ' ', text)
tokens = word_tokenize(text_clean.lower())

# -------------------------------------------------
# Step 5: Stopwords Removal
# -------------------------------------------------

# stopwords.words("english"): A built-in list of ~180 common filler words (the, is, a, of...)
# that carry no analytical value and are removed to reduce noise.
stop_words = set(stopwords.words("english"))

filtered_words = []
for w in tokens:
    if w not in stop_words:
        filtered_words.append(w)

print("\nFiltered Words:", filtered_words)

# -------------------------------------------------
# Step 6: Stemming
# -------------------------------------------------

# PorterStemmer(): A rule-based algorithm that chops off word endings to get the raw root.
# Fast but crude — may produce non-dictionary words (e.g., "studies" → "studi").
ps = PorterStemmer()
stemmed_words = []

for w in filtered_words:
    stemmed_words.append(ps.stem(w))  # stem(): Applies suffix-stripping rules to reduce word to its stem.

print("\nStemmed Words:", stemmed_words)

# -------------------------------------------------
# Step 7: Lemmatization
# -------------------------------------------------

# WordNetLemmatizer(): Uses the WordNet dictionary to map a word to its true base form.
# Smarter than stemming — always produces a valid dictionary word (e.g., "studies" → "study").
lemmatizer = WordNetLemmatizer()
lemmatized_words = []

for w in filtered_words:
    lemmatized_words.append(lemmatizer.lemmatize(w))  # lemmatize(): Looks up the word in WordNet and returns its canonical form.

print("\nLemmatized Words:", lemmatized_words)

# -------------------------------------------------
# Step 8: Stemming vs Lemmatization Comparison
# -------------------------------------------------

# A side-by-side comparison to show the difference in output between the two methods.
sample = "studies studying cries cry"
sample_words = word_tokenize(sample)

print("\nComparison (Stemming vs Lemmatization):")
for w in sample_words:
    print(f"  {w}  →  Stem: {ps.stem(w)}   |   Lemma: {lemmatizer.lemmatize(w)}")

# -------------------------------------------------
# Step 9: TF-IDF (Manual Calculation)
# -------------------------------------------------

# TF-IDF = Term Frequency × Inverse Document Frequency
# TF: How often a word appears in one document. IDF: How rare a word is across all documents.
# Result: High TF-IDF score = word is both frequent in THIS document AND rare in others.

# Two sample documents (lowercase for consistent matching)
documentA = 'Jupiter is the largest planet'.lower()
documentB = 'Mars is the fourth planet from the sun'.lower()

bagA = documentA.split()
bagB = documentB.split()

# Combine all unique words from both documents
uniqueWords = set(bagA).union(set(bagB))

# Count occurrences of each unique word in each document
dictA = dict.fromkeys(uniqueWords, 0)
dictB = dict.fromkeys(uniqueWords, 0)

for w in bagA:
    dictA[w] += 1

for w in bagB:
    dictB[w] += 1

# --- TF (Term Frequency) ---
# TF = count of word in doc / total words in doc
# Normalizes counts so longer documents don't get unfair advantage.
def computeTF(wordDict, doc):
    tfDict = {}
    total = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count / total
    return tfDict

tfA = computeTF(dictA, bagA)
tfB = computeTF(dictB, bagB)

# --- IDF (Inverse Document Frequency) ---
# IDF = log(Total documents / Documents containing the word)
# math.log(): Compresses the scale so very rare words are rewarded without being excessively boosted.
def computeIDF(docs):
    N = len(docs)
    idfDict = dict.fromkeys(docs[0].keys(), 0)

    for doc in docs:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word in idfDict:
        idfDict[word] = math.log(N / idfDict[word])

    return idfDict

idf = computeIDF([dictA, dictB])

# --- TF-IDF (Combined Score) ---
def computeTFIDF(tf, idf):
    tfidf = {}
    for word, val in tf.items():
        tfidf[word] = val * idf[word]
    return tfidf

tfidfA = computeTFIDF(tfA, idf)
tfidfB = computeTFIDF(tfB, idf)

# Display results as a table
df_tfidf = pd.DataFrame([tfidfA, tfidfB])
print("\nTF-IDF Table:\n", df_tfidf)
























"""
--- Code Explanation ---
1. sent_tokenize(): Splits a paragraph into individual sentences using punctuation rules.
2. word_tokenize(): Splits text into individual word tokens (including punctuation marks).
3. pos_tag(): Labels each token with its grammatical role — NN (Noun), VB (Verb), JJ (Adjective), RB (Adverb), etc.
4. re.sub('[^a-zA-Z]', ' ', text): Removes all non-letter characters (numbers, punctuation) from the text for clean processing.
5. stopwords.words("english"): A built-in NLTK list of ~180 common filler words (the, is, a, of) that are removed to reduce noise.
6. PorterStemmer / ps.stem(): Rule-based algorithm that chops word endings to get the raw root — fast but crude (may produce non-dictionary words).
7. WordNetLemmatizer / lemmatizer.lemmatize(): Dictionary-based algorithm that maps words to their true base form — smarter, always produces a valid word.
8. computeTF(): Calculates how often each word appears in a single document, normalized by total word count.
9. computeIDF(): Calculates how rare each word is across all documents using log(N / doc_frequency).
10. computeTFIDF(): Multiplies TF × IDF to get the final importance score. High score = frequent in THIS doc AND rare in others.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS NLP (Natural Language Processing)?
   - Definition: The branch of Artificial Intelligence that enables computers to read, understand,
     interpret, and generate human language in a meaningful way.
   - Why is it hard? Human language is ambiguous, context-dependent, and full of exceptions.
     "I saw the man with the telescope" — who has the telescope? Computer doesn't know.
   - Applications: Google Search, ChatGPT, Siri, Gmail spam filter, autocorrect, sentiment analysis.
   - NLP Pipeline Order: Raw Text → Tokenization → Lowercasing → Stop Word Removal
     → Stemming/Lemmatization → POS Tagging → Feature Extraction (TF-IDF) → ML Model.

2. TOKENIZATION:
   - Definition: The process of splitting a large body of text into smaller, individual units called "tokens."
   - Types:
     a) Word Tokenization: Split by words. "I love NLP" → ["I", "love", "NLP"].
     b) Sentence Tokenization: Split by sentences. A paragraph → list of sentences.
   - Why is it the first step? You cannot analyze a whole paragraph as one unit. You must break
     it into individual "atoms" (words) before any further processing can happen.
   - Punctuation handling: We filter punctuation using `word.isalpha()` — keep only real words.

3. POS TAGGING (Part-of-Speech Tagging):
   - Definition: Labeling each token with its grammatical role (Noun, Verb, Adjective, etc.).
   - Common POS Tags:
     - NN = Noun (singular): "dog", "car".
     - VB = Verb: "run", "process".
     - JJ = Adjective: "fast", "beautiful".
     - RB = Adverb: "quickly", "very".
     - DT = Determiner: "the", "a".
   - Why useful: The word "bank" means a financial institution (NN) or to tilt/bank an airplane (VB).
     POS tags resolve this ambiguity. Lemmatization also needs POS to find the correct root.

4. STOP WORDS:
   - Definition: Extremely common words that carry no real semantic meaning in a sentence.
   - Examples: "the", "is", "at", "which", "on", "and", "a", "an", "in", "it".
   - Why remove? If we count "the" in TF-IDF, it appears in every document with a very high
     TF score, but adds zero information to distinguish one document from another.
   - NLTK's English stop word list: Contains ~179 words.

5. STEMMING VS LEMMATIZATION:
   Both try to reduce a word to its base/root form to normalize vocabulary.
   
   a) STEMMING (Porter Stemmer):
      - Method: Uses a set of hard-coded rules to chop off suffixes.
      - Speed: Very fast (no dictionary lookup).
      - Accuracy: POOR. Often creates non-real words.
      - Examples:
        "running" → "run"     ✓ (correct)
        "caring"  → "car"     ✗ (wrong — should be "care")
        "studies" → "studi"   ✗ (wrong — "studi" is not a word)
   
   b) LEMMATIZATION (WordNet Lemmatizer):
      - Method: Uses a real dictionary (WordNet) and grammar rules to find the true dictionary root.
      - Speed: Slower than stemming (requires dictionary lookup).
      - Accuracy: EXCELLENT. Always produces a valid English word.
      - Examples:
        "running" → "run"     ✓
        "caring"  → "care"    ✓
        "studies" → "study"   ✓
        "better"  → "good"    ✓ (knows it's the comparative of "good")
      - Advantage over stemming: Never produces a nonsense word.

6. TF-IDF (Term Frequency — Inverse Document Frequency):
   - Definition: A numerical statistic that reflects how important a word is to a specific
     document within a collection (corpus) of documents.
   - Goal: Give high scores to words that are UNIQUE to a document, and low scores to words
     that appear everywhere (like "the" or "data").
   
   a) Term Frequency (TF):
      - Formula: TF(word, doc) = (Count of word in doc) / (Total words in doc)
      - Meaning: How often does this word appear in THIS specific document?
      - Example: Doc1 = "cat sat cat" (3 words). TF("cat") = 2/3 = 0.667.
   
   b) Inverse Document Frequency (IDF):
      - Formula: IDF(word) = log(Total docs / Docs containing the word)
      - Meaning: How RARE is this word across ALL documents?
      - If a word appears in all 3 docs: IDF = log(3/3) = log(1) = 0 (penalized to zero).
      - If a word appears in only 1 doc: IDF = log(3/1) = log(3) ≈ 1.1 (rewarded).
      - We use log() to compress the IDF scale — without it, rare words would be rewarded too aggressively.
   
   c) TF-IDF Score:
      - Formula: TF-IDF = TF × IDF
      - High score: Word appears frequently in one doc BUT rarely in others (unique, important word).
      - Low score: Word appears in all docs (common, uninformative) OR barely appears in any doc.
      - Use case: Search engines use TF-IDF to rank which documents best match a search query.

--- Detailed Viva Q&A ---

Q1: Why do we convert all text to lowercase before processing?
A1: Computers are case-sensitive. "Apple" and "apple" are treated as two completely different words. Without lowercasing, "Apple" appearing at the start of a sentence and "apple" mid-sentence would have separate counts, ruining frequency analysis.

Q2: Give an example of why Stemming can be bad.
A2: Stemming "Universe" and "University" produces "Univers" for both. The computer then thinks these two completely different concepts are the same root word. Lemmatization would correctly keep them separate since neither is the comparative form of the other.

Q3: What is the main difference between Stemming and Lemmatization?
A3: Stemming is a crude, rule-based algorithm that forcefully chops off word endings — fast but often creates non-words ("studi" from "studies"). Lemmatization uses an actual language dictionary (WordNet) to convert words to their true grammatical root — slower but always produces a valid English word.

Q4: What exactly does the TF-IDF formula measure?
A4: It measures a word's "uniqueness" and "relevance" to a specific document. TF rewards words that appear frequently in this document. IDF penalizes words that appear in every document. The product TF×IDF gives high scores only to words that are both common in THIS doc and rare across ALL docs.

Q5: In TF-IDF, what would happen to "the" if we didn't remove stop words?
A5: The TF for "the" would be very high (it appears constantly). But the IDF would be log(N/N) = log(1) = 0 because it appears in EVERY document. So TF-IDF = high × 0 = 0. Stop word removal saves computation time since TF-IDF already mathematically neutralizes them.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: What does `word.isalpha()` do in your tokenization code?
   A1: It returns True only if ALL characters in the string are alphabetical letters. It filters out: numbers (e.g., "2024"), punctuation (".", "!", ","), and mixed tokens ("NLP2024"). Only pure words like "data" and "science" pass through.

   Q2: What is the output of `vectorizer.fit_transform()`?
   A2: A "Sparse Matrix" — a massive 2D grid where rows are documents, columns are every unique word in the entire corpus, and each cell contains the TF-IDF score for that word in that document. It is called "sparse" because most cells are 0 (most words don't appear in most documents).

   Q3: Why do we print TF, IDF, and TF-IDF separately in this code?
   A3: To demonstrate the step-by-step mathematical reasoning: TF shows raw word frequency, IDF shows word rarity across documents, and TF-IDF shows the final importance score. Showing all three separately proves we understand the formula, not just the final output.

------------------------- End of Viva Notes -------------------------
"""
