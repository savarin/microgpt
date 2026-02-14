# Parts Assessment

## Decision: No Parts

This book has 7 chapters with a strict linear progression — each chapter builds directly on the previous one. The scope is tight (a single 201-line file), the narrative arc is clear (from raw text to generated text), and the chapter titles alone communicate the structure.

Three natural clusters exist (Foundation / The Model / Using It), but formalizing them as "Part I, Part II, Part III" adds navigational overhead without helping the reader. The book is short enough to hold in your head as a single arc. Parts would fragment what should feel like one continuous build.

## Why Not Parts

- **Linear dependency**: Every chapter depends on the one before it. Parts suggest independent sections you can read in any order — that's not this book.
- **Single source file**: The entire codebase is one file. Parts would impose artificial boundaries on what is fundamentally a continuous walkthrough.
- **Seven chapters**: Short enough that a flat table of contents is immediately scannable. Parts help when you need to find Chapter 14 in a 25-chapter book — not here.
- **The clusters are implicit**: A reader will naturally feel the shift from "building tools" (Chapters 1-3) to "building the model" (Chapters 4-5) to "using the model" (Chapters 6-7). Naming these transitions would over-explain what the reader already perceives.
