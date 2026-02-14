# Chapter 2: Data and Tokens

## Intro

This chapter covers how raw text becomes the discrete symbols a neural network can process. The gap between human language and mathematical computation starts here: a file of names becomes a list of integer sequences, and a vocabulary maps characters to numbers and back. This is the bridge between the world of strings and the world of vectors, and every operation in the rest of the book depends on it.

## Sections

### 2.1 The Dataset
Loading `names.txt`: a list of 32,000+ human names, one per line. Each name is a "document" in the model's world. Why names are a good first dataset — short sequences, clear structure, easy to evaluate output quality by inspection.

### 2.2 Character-Level Tokenization
Building the vocabulary from the unique characters in the dataset. Why character-level tokenization (as opposed to word-level or subword BPE): simplicity, no external tokenizer needed, and the model learns character patterns directly. The `uchars` list as a bidirectional mapping between characters and integer IDs.

### 2.3 The BOS Token
The Beginning of Sequence token: a special symbol that doesn't correspond to any character. Why it exists — the model needs a signal for "start generating" and "stop generating." The elegant trick of using the same token for both boundaries: `[BOS] + characters + [BOS]`. How this shapes the training signal.

### 2.4 From Documents to Token Sequences
The complete tokenization pipeline: take a name like "emma", wrap it as `[BOS, e, m, m, a, BOS]`, convert each symbol to its integer ID. This is the input format the model will consume during training. The `vocab_size` as the total number of distinct symbols the model must handle.

## Conclusion

The reader now understands the data pipeline: text file → list of strings → list of integer sequences. They know what a token is, why a special BOS token exists, and what `vocab_size` means. This vocabulary and tokenization scheme is used unchanged through training (Chapter 5) and inference (Chapter 6).

## Cross-Chapter Coordination

- **Introduces**: Tokens, vocabulary (`uchars`), `BOS`, `vocab_size`, the concept of a document as a sequence of integer IDs
- **Referenced by**: Chapter 4 (embedding tables are sized by `vocab_size`), Chapter 5 (training loop tokenizes documents), Chapter 6 (inference starts with `BOS` and stops at `BOS`)
- **Depends on**: Chapter 1 (the "what the code does" overview)
