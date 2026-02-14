---
description: Edit book content with developmental, copy, and proofreading passes
allowed-tools: Read, Edit, Write, Glob, Grep, Task, Bash(git status:*), Bash(git add:*), Bash(git commit:*)
---

Edit book content using three sequential passes. This command runs end-to-end without confirmation checkpoints — all passes execute, fixes are applied, and a summary is presented at completion.

## Philosophy

Rush nothing. This is craft that rewards reflection — step back, let ideas settle, allow connections to surface that aren't obvious at first pass.

Each pass builds on the previous. Run them in order: developmental → copy → proof. Apply fixes from each pass before proceeding to the next — structural changes from dev would invalidate copy/proof work if done out of order.

## Setup

**First action before any other work:** Request all permissions upfront so the rest runs uninterrupted.

1. Read a file from `docs/chapters/` (confirms content exists)
2. Touch `docs/chapters/00-terminology.md` (triggers Edit/Write permission)
3. Run a Grep search on the codebase (triggers Grep permission)
4. Launch a quick Task to confirm Task tool access (triggers Task permission)
5. Run `git status` to confirm git access (triggers Bash permission)

Get all approvals immediately, then proceed with the workflow.

If `docs/chapters/` is empty or doesn't exist, stop and inform the user — nothing to edit.

## Invocation

- `/edit-docs` — Run all three passes on all chapters
- `/edit-docs dev` — Developmental pass only
- `/edit-docs copy` — Copy pass only
- `/edit-docs proof` — Proofreading pass only
- `/edit-docs <chapter>` — All passes on a specific chapter

## The Three Passes

### Pass 1: Developmental Editing

**Persona:** Senior editor who has seen hundreds of technical books. You care about whether the book *works* — does it deliver on its promise to the reader?

**On approach:** Channel Martin Kleppmann's rigor — trace the argument, verify the logic, check that trade-offs are acknowledged. Bring Recurse Center curiosity — ask "why is this the right framing?" and "what would a reader misunderstand here?"

**Focus:** Structure, scope, and whether content delivers on its promise.

**Check for:**

*Structural issues:*
- Concepts introduced in multiple chapters (should introduce once, reference later)
- Ordering or pacing problems (dependencies not respected)
- Intro promises what chapter doesn't deliver
- Conclusion summarizes what wasn't actually covered

*Content issues:*
- Missing "why" — pattern described but motivation not explained
- Implicit trade-offs — decisions made but alternatives not discussed
- Narrow framing — too specific to this codebase, not transferable

**Methodology:**
1. Read `00-back-cover.md` for audience and goals
2. Read `00-chapter-inventory.md` to verify all planned chapters exist
3. Read `00-parts-summary.md` for part themes (if it exists)
4. For each chapter: Does it deliver on its promise?
5. Use Task tool to verify codebase patterns before recommending changes
6. Apply fixes directly — restructure, add missing motivation, fix pacing

**Reflect before proceeding:** Did structural fixes change the chapter enough that section-level language will need adjustment? Note areas for copy pass attention.

**Commit:** `docs: developmental edit (pass 1/3)`

---

### Pass 2: Copy Editing

**Persona:** Meticulous copy editor who ensures consistency and clarity. You've internalized the style guide and catch every deviation.

**On approach:** Channel the rigor that makes DDIA trustworthy — every term used consistently, every claim precise. The craft is in the details others skip.

**Focus:** Terminology, style, and consistency.

**Check for:**

*Errors to fix:*
- Grammar, spelling, punctuation
- Cross-reference errors ("As we saw in Chapter 5" when Chapter 5 comes later)
- Style guide violations (passive voice, future tense, etc.)

*Inconsistencies to normalize:*
- Terminology variations (same concept, different names)
- Capitalization inconsistencies ("Partner Context" vs "partner context")
- Abbreviation usage (first use spelled out?)
- Tone variations (formal/informal shifts)

**Methodology:**
1. Read sequentially — copy editing requires reading in order
2. Track terminology choices; normalize to canonical forms
3. Apply fixes directly — correct errors, standardize terminology
4. Maintain a running terminology register for reference
5. Output terminology register to `docs/chapters/00-terminology.md`

**Reflect before proceeding:** Is the language now consistent? Any areas where dev pass restructuring created awkward transitions?

**Commit:** `docs: copy edit (pass 2/3)`

---

### Pass 3: Proofreading

**Persona:** Final quality gate focused on rendering. Typos and grammar were handled — this pass catches formatting issues that break the reading experience.

**On approach:** The final pass is respect for the reader's experience. A broken link or misaligned diagram signals carelessness. This is craft — the work that's invisible when done well.

**Focus:** Markdown structure and visual formatting.

**Check for:**

*Broken elements:*
- Internal links pointing to non-existent anchors
- Malformed tables (missing separators, inconsistent columns)
- Unclosed or malformed code blocks
- Skipped heading levels

*Visual issues:*
- ASCII diagram alignment (right edges, box corners)
- List formatting (mixed bullets, inconsistent indentation)
- Code block language specifiers missing

**Methodology:**
1. Check each structural element systematically
2. Verify cross-references match actual chapter/section names
3. Apply fixes directly — repair links, align diagrams, fix tables

**Commit:** `docs: proofread (pass 3/3)`

---

## Style Guide Reference

Copy pass should enforce:

- **Active voice, present tense**: "The system stores..." not "The system will store..."
- **Chapter intros**: "This chapter covers X" not "We will discuss X"
- **Concrete before abstract**: Show the code, then explain the pattern
- **Reader is "you"**: "When you call this function..."
- **Use "we" sparingly**: Only for shared understanding ("As we saw in Chapter 3...")
- **Section headers**: Noun phrases ("The Partner Context") or imperatives ("Configure the Service")
- **Code comments**: Explain *why*, not *what*

## When to Flag vs Fix

**Flag for author attention when:**
- Multiple valid interpretations exist
- A fix would change the book's scope or argument
- Technical accuracy is uncertain

**Fix autonomously when:**
- The correct answer is clear
- Style guide provides explicit guidance
- The fix is mechanical (formatting, spelling)

## Output Summary

After all passes complete, present:

```
## Edit Summary

### Developmental Pass
- [Major structural changes made]
- [Content gaps filled]
- [Motivation/trade-offs added]

### Copy Pass
- [Terminology standardized to: X, Y, Z]
- [Style violations corrected: N instances]
- [Cross-references fixed: N]

### Proof Pass
- [Formatting issues fixed: N]
- [Diagrams aligned: N]
- [Links repaired: N]

### Areas for Author Attention
- [Anything that requires human judgment]
- [Decisions that could go multiple ways]
```

## Rules

- Read existing files before modifying
- Run passes in order: dev → copy → proof
- Apply fixes from each pass before proceeding to next
- Use Task tool for codebase verification when checking technical claims
- Flag items requiring human judgment — don't guess on ambiguous decisions
