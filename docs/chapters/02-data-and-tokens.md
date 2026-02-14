# Chapter 2: Data and Tokens

## 2.1 The Dataset

Neural networks learn from examples. Before a model can learn the structure of human names, it needs to see thousands of them. microgpt uses a dataset of approximately 32,000 names, sourced from Andrej Karpathy's makemore project:

```python
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
```

The code downloads the file only if it doesn't already exist — a polite pattern that avoids redundant network calls. It then reads the file, splits it into lines, strips whitespace, filters out empty lines, and shuffles the result. Each line is one name: "emma", "olivia", "ava", "isabella", and so on.

The variable name `docs` is worth noting. The code calls each name a "document." This is deliberate generality — the algorithm doesn't know or care that these are names. It treats each entry as a document: a sequence of characters to learn from. You could replace `names.txt` with a file of city names, chemical compounds, or Pokémon names, and the algorithm would work the same way. The model learns whatever patterns exist in the data it sees.

Names are an excellent first dataset for several reasons. They are short — typically 3 to 10 characters — so the model can process an entire name in one training step without truncation. They have clear structure — English names follow patterns of consonants and vowels that the model can learn. And the quality of generation is easy to evaluate by inspection: you can look at "mede" and immediately judge whether it looks like a plausible name.

## 2.2 Character-Level Tokenization

Neural networks operate on numbers, not characters. The model needs a way to translate between the world of strings ("emma") and the world of integers that it can multiply and add. This translation is called *tokenization*.

```python
uchars = sorted(set(''.join(docs)))
```

This single line builds the vocabulary. It concatenates every name into one long string, collects the unique characters into a set, and sorts them alphabetically. For English names, the result is the 26 lowercase letters plus possibly a few special characters (hyphens, apostrophes) — around 27 characters total.

The list `uchars` serves as a bidirectional mapping. To convert a character to an integer (encoding), you find its index: `uchars.index('e')` might return `4`. To convert an integer back to a character (decoding), you index into the list: `uchars[4]` returns `'e'`. This is character-level tokenization — each token is a single character.

Why character-level? Modern language models like GPT-2 and GPT-4 use subword tokenization (Byte Pair Encoding, or BPE), where tokens can be entire words ("hello"), common subwords ("ing", "tion"), or individual characters for rare sequences. BPE is more efficient — it represents common words in fewer tokens — but it requires a separate algorithm to learn the token vocabulary from data.

Character-level tokenization is simpler. The vocabulary is just the set of characters that appear in the data. There is no learning step, no merging algorithm, no special handling of unknown words. Every possible string over the vocabulary can be tokenized. For short sequences like names, the efficiency loss compared to BPE is minimal — a 5-character name is 5 tokens either way.

## 2.3 The BOS Token

Beyond the characters from the data, the model needs one more token: a special symbol that marks the boundary of a sequence.

```python
BOS = len(uchars)
vocab_size = len(uchars) + 1
```

`BOS` stands for Beginning of Sequence. Its token ID is the number right after the last character ID — if there are 27 characters (IDs 0 through 26), then BOS is 27. The total `vocab_size` is 28: the 27 characters plus the BOS token.

Why does the model need a special boundary token? Consider the task: the model sees a sequence of characters and must predict what comes next. But what is the "input" before the very first character of a name? The model needs something to condition on to generate the first character. BOS fills this role: it says "a new name is starting — predict the first character."

The model also needs to know when a name *ends*. microgpt uses the same BOS token for this purpose:

```python
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
```

A name like "emma" becomes the token sequence `[BOS, e, m, m, a, BOS]`. During training, the model learns that:
- Given BOS, the next token is likely a common first letter (consonants like 'j', 'm', 's' are common)
- Given the sequence [BOS, e, m, m, a], the next token is likely BOS (the name is over)

Using one token for both start and stop is an elegant design choice. Many implementations use separate BOS and EOS (End of Sequence) tokens, which requires two special token IDs. microgpt achieves the same functionality with one, keeping the vocabulary as small as possible. The model learns from context whether BOS means "start" or "stop" — if BOS appears at the beginning of the input, the model generates the first character; if BOS appears as the model's prediction, generation halts.

## 2.4 From Documents to Token Sequences

To see the full tokenization pipeline, take the name "ada":

1. The raw string is `"ada"`.
2. Character-to-integer encoding: `a` → 0, `d` → 3, `a` → 0 (assuming standard alphabetical ordering).
3. BOS wrapping: `[BOS, 0, 3, 0, BOS]` → using BOS = 27: `[27, 0, 3, 0, 27]`.

This integer sequence is what the model processes. During training, the model sees it as a series of input-target pairs:

```
Position 0:  input = 27 (BOS),  target = 0  (a)    → "After BOS, predict 'a'"
Position 1:  input = 0  (a),    target = 3  (d)    → "After 'a', predict 'd'"
Position 2:  input = 3  (d),    target = 0  (a)    → "After 'd', predict 'a'"
Position 3:  input = 0  (a),    target = 27 (BOS)  → "After 'a', predict end"
```

Each position teaches the model one prediction: given this token at this position, what should come next? The model processes these positions sequentially — each token is fed through the model one at a time, building up context about the characters seen so far. (Chapter 5 covers the details of this sequential processing.)

The `vocab_size` — 28 in this example — determines the shape of several key components in the model. The token embedding table has one row per token ID (28 rows). The output layer produces one score per token ID (28 logits). When the model makes a prediction, it produces 28 numbers — one for each possible next token — and the loss measures how much probability it assigned to the correct one.

This is the entire data pipeline: a text file of names becomes a list of integer sequences, and each sequence becomes a set of input-target pairs for next-token prediction. Everything downstream — the autograd engine, the model architecture, the training loop, the inference procedure — operates on these integer sequences and this vocabulary.
