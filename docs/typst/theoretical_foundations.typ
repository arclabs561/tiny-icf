#set page(margin: 2cm)
#set text(size: 11pt)
#set heading(numbering: "1.")

// Code highlighting available via codly package if installed
// #import "@preview/codly:0.1.0": *
// #show: codly.with(theme: "github-dark")

#title[
  Theoretical Foundations: Information Theory and Complexity Constraints
]

#text(fill: gray)[
  ICF Estimation Project · #datetime.today().display()
]

#par(justify: true)[
  This document establishes the theoretical foundations for character-level ICF prediction, including information-theoretic bounds, Kolmogorov complexity constraints, and fundamental questions about feasibility. We explore whether the goal is even possible, what constraints we must satisfy, and what the fundamental limits are.
]

== The Fundamental Question

Is it even possible to build a model that compresses word→ICF mappings better than a dictionary while generalizing to unseen words? This question leads us to explore Kolmogorov complexity, information theory, and the structure of language itself.

== Notation

- #strong[K(x)]: Kolmogorov complexity of x (length of shortest program producing x)
- #strong[K(M)]: Kolmogorov complexity of model M
- #strong[K(D)]: Kolmogorov complexity of dictionary D (word → ICF mapping)
- #strong[H(ICF)]: Shannon entropy of ICF distribution
- #strong[I(X; Y)]: Mutual information between X and Y
- #strong[MDL]: Minimum Description Length = K(model) + K(data | model)
- #strong[V]: Vocabulary size
- #strong[N]: Number of training samples

== Kolmogorov Complexity Constraint

=== The Theorem

#strong[Kolmogorov Complexity is invariant up to an additive constant:]

#align(center)[
  $ K_U(x) = K_V(x) + O(1) $
]

for any two universal Turing machines U, V. This means K is well-defined (up to a constant).

=== Implication for Our Problem

- #strong[K(ICF_function)] is well-defined (up to a constant)
- #strong[K(dictionary)] = K(ICF_function) + O(1) (if dictionary is optimal encoding)
- #strong[K(model)] = K(ICF_function) + K(architecture) + O(1)

=== Should K(model) = K(dictionary) by Definition?

#strong[If model perfectly learns ICF:]

#align(center)[
  $ K("model") approx K("ICF_function") + K("architecture") $
]

#align(center)[
  $ K("dict") approx K("ICF_function") $ (if optimal encoding)
]

#strong[Therefore:]

#align(center)[
  $ K("model") approx K("dict") + K("architecture") $ (up to constants)
]

#strong[But there's a crucial distinction:]

- #strong[Dictionary]: Stores explicit mapping (word → ICF for seen words)
- #strong[Model]: Stores implicit function (f: words → ICF for all words)
- #strong[If ICF has structure]: K(model) < K(dict) possible (structure compresses)
- #strong[If ICF is random]: K(model) ≈ K(dict) (cannot compress, must memorize)

#strong[Answer]: No, they don't have to be the same. If ICF has structure, the model can be smaller because it encodes the structure, not just the mapping.

== Compression Constraint

=== Dictionary Complexity

#strong[Uncompressed Dictionary:]

#align(center)[
  $ K(D) = V times ("avg_word_bytes" + 4 "bytes") $
]

For V=100k words: ~900 KB

#strong[Compressed Dictionary:]

- gzip: ~3-4× compression → ~225-300 KB for V=100k
- LZMA/xz: ~4-6× compression → ~150-200 KB for V=100k
- zstd/brotli: ~4-5× compression → ~180-225 KB for V=100k
- #strong[Typical compressed size]: ~180-250 KB for V=100k

#strong[Additional optimizations:]

- #strong[Sparse dictionary] (only rare words, ICF > 0.5): ~50k words → ~90 KB compressed
- #strong[Trie/prefix tree] (share common prefixes): ~60-70% of flat → ~130 KB compressed

=== Model Complexity

Our model M: f(word) → ICF has:

#align(center)[
  $ K(M) = |theta| times 4 "bytes" + K("architecture") $
]

- |θ| = 40k parameters: ~160 KB
- K(architecture) ≈ few KB (fixed, reusable code)
- #strong[Total: ~160 KB + architecture]

=== The Constraint (Revised)

#strong[K(M) < K(D_compressed)] for the model to be useful compression.

#strong[Our case (compressed dictionary):]

- Model: 160 KB vs Compressed dict: 180-250 KB ✓ (satisfied, but marginal)
- Model: 160 KB vs Sparse dict: 90 KB ✗ (VIOLATED - sparse dict is smaller)
- Model: 160 KB vs Trie dict: 130 KB ✗ (VIOLATED - trie dict is smaller)

#strong[Critical insight]:

- If dictionary is compressed, model advantage is #strong[marginal (1.1-1.6×)], not 5.6×
- If dictionary is sparse/trie-optimized, model may be #strong[larger] than dictionary
- #strong[Generalization is the key advantage]: Dictionary cannot handle unseen words, model can
- Model must compress AND generalize to justify its existence

== Minimum Description Length (MDL) Principle

=== Definition

#align(center)[
  $ "MDL" = K("model") + K("data" | "model") $
]

Where:
- #strong[K(model)]: Model complexity (~160 KB)
- #strong[K(data | model)]: Compressed training data given model
  - = -log P(data | model) (bits)
  - ≈ training_loss × N (information content)

=== Optimal Model

Minimizes MDL = model_size + training_loss.

- #strong[Too simple]: High K(data|model) (underfitting, high training loss)
- #strong[Too complex]: High K(model) (overfitting, memorization)
- #strong[Optimal]: Balance where MDL is minimized

=== For Our Case

#align(center)[
  $ "MDL"("model") = 160 "KB" + "training_loss" times N $
]

#align(center)[
  $ "MDL"("dictionary") = 900 "KB" + 0 $ (perfect fit, no compression of data)
]

#strong[Model wins if]: 160 KB + loss×N < 900 KB

#strong[This requires]: loss×N < 740 KB

#strong[For N=50k]: loss < 0.015 per example (very strict requirement)

#strong[Current status]: Our training loss is ~0.1-0.2, so loss×N ≈ 5-10 MB >> 740 KB. This suggests either:
1. Model is not compressing well (high K(data|model))
2. Need better regularization (reduce effective capacity)
3. Need more data (increase N to reduce per-example loss)

== Information-Theoretic Lower Bound

=== Shannon Entropy

The Shannon entropy H(ICF) of the ICF distribution:

#align(center)[
  $ H("ICF") = -sum_i P("ICF"_i) log_2 P("ICF"_i) $
]

- #strong[If ICF values are random]: H(ICF) ≈ log₂(V) bits per word
- #strong[If ICF has structure]: H(ICF) < log₂(V)
- #strong[For ICF in [0,1]]: H(ICF) ≤ log₂(V) (equality if uniform)

=== Model Capacity Requirement

Model must capture at least H(ICF) bits:

#align(center)[
  $ "Model capacity" >= H("ICF") times N $ (for N words)
]

#strong[But can compress] if structure exists.

#strong[Compression ratio]: H(ICF) / actual_model_capacity

=== For Our Model

- #strong[Capacity]: 40k params × 32 bits = 1.28 Mbits
- #strong[If H(ICF) ≈ 10 bits/word] (structured, not uniform):
  - For N=50k: need 500k bits = 62.5 KB
  - Our model: 160 KB > 62.5 KB ✓ (sufficient capacity)

== Maximum Spearman Correlation Bound

=== Information-Theoretic Bound

Maximum Spearman correlation is bounded by:

#align(center)[
  $ rho_max <= sqrt(frac(I(X; Y), H(Y))) $
]

Where:
- #strong[X] = Character features
- #strong[Y] = ICF values
- #strong[I(X; Y)] = Mutual information between characters and ICF
- #strong[H(Y)] = Shannon entropy of ICF distribution

=== Formal Derivation

#align(center)[
  $ rho_max^2 <= frac(I(X; Y), H(Y)) <= frac(H(X) - H(X|Y), H(Y)) $
]

This follows from the data processing inequality and the relationship between correlation and mutual information.

=== Key Insight

If `I(Characters; ICF)` is low relative to `H(ICF)`, the maximum achievable correlation is inherently limited.

#strong[Hypothesis]: Character-level features capture approximately 18-19% of ICF variance because:
- Character patterns → semantic frequency is an #strong[indirect mapping]
- Many words with similar character patterns have different ICF values
- ICF depends on corpus/domain characteristics (not just characters)

== Generalization Constraint

=== Requirement

Model must learn function f: words → ICF that:
- #strong[Fits training]: f(word_i) ≈ ICF_i for i ∈ training set
- #strong[Generalizes]: f(word_new) ≈ ICF_new for unseen words

=== Implies Regularity

This requires regularity in the ICF function:
- Words with similar patterns → similar ICF
- Morphological structure → frequency patterns
- Character sequences → rarity indicators

=== Model Capacity

Must be:
- #strong[Sufficient]: Capture regularity (not too simple)
- #strong[Limited]: Prevent memorization (not too complex)
- #strong[Optimal]: Match complexity of true function

== Summary: All Constraints

=== Explicit Constraints

1. #strong[K(model) < K(dictionary_compressed)] [Kolmogorov complexity]
   - ✓ vs compressed dict (180-250 KB): 160 KB < 180-250 KB (marginal)
   - ✗ vs sparse dict (90 KB): 160 KB > 90 KB (violated)
   - ✗ vs trie dict (130 KB): 160 KB > 130 KB (violated)

2. #strong[MDL = K(model) + K(data|model) minimized] [MDL principle]

3. #strong[Generalization: f(word_new) ≈ ICF_new] [generalization] ✓

4. #strong[Data efficiency: N samples sufficient] [sample complexity] ⚠️

5. #strong[Computational: fast inference] [speed constraint] ⚠️ (dict is 100-1000× faster)

6. #strong[Storage: K(model) < K(dict_compressed)] [compression] ⚠️ (marginal)

=== Implicit Constraints

7. #strong[Regularity: ICF has structure] [regularity assumption] ⚠️

8. #strong[Capacity: VC_dim matches data] [capacity constraint] ⚠️

9. #strong[Architecture: expressive but compact] [architecture design] ✓

10. #strong[Information: H(ICF) < log₂(V)] [entropy constraint] ⚠️

== Why We Don't Use Nearest-Word Mapping

A natural question: why not just map unseen words to their nearest neighbors in our frequency dictionary? This approach is problematic because #strong[frequency ≠ similarity].

Consider these examples:
- "cat" (edit distance 1 from "bat") - very different frequencies
- "the" (most common) vs "thy" (rare, archaic) - similar form, opposite frequency
- "computer" vs "computers" - similar, but frequencies differ
- "run" vs "fun" - edit distance 1, but "run" is much more common

If we mapped unseen words to their nearest neighbors, we'd train the model with wrong frequency labels, teaching it incorrect patterns. Instead, we learn the structure directly from character patterns.

== Key Insights

1. #strong[The Kolmogorov complexity constraint is nuanced]:
   - #strong[Uncompressed]: Model (160 KB) << Dictionary (900 KB) ✓ (5.6× advantage)
   - #strong[Compressed]: Model (160 KB) vs Dictionary (180-250 KB) ⚠️ (1.1-1.6×, marginal)
   - #strong[Sparse/Trie]: Model (160 KB) > Dictionary (90-130 KB) ✗ (violated)

2. #strong[Compression matters critically]: Dictionary compression (3-6×) reduces K(dict) from 900 KB to 180-250 KB, making model advantage marginal.

3. #strong[Generalization is the key differentiator]:
   - Dictionary: Only seen words (sparse coverage)
   - Model: Any UTF-8 string (dense coverage)
   - #strong[This is why model is useful despite size constraints]

4. #strong[MDL suggests issues]: High training loss means K(data|model) is large, so total MDL may not be optimal.

5. #strong[Sample complexity is tight]: We have 50k samples but model has 40k params, suggesting we need either:
   - More data (increase N)
   - More regularization (reduce effective capacity)
   - Smaller model (reduce |θ|)

6. #strong[Regularity assumption is critical]: If ICF has no structure, model cannot compress better than dictionary. We assume morphology/phonotactics predict frequency.

7. #strong[Use case determines winner]:
   - #strong[If only need seen words]: Dictionary wins (smaller, faster, exact)
   - #strong[If need OOV/generalization]: Model wins (handles unseen words)
   - #strong[If need both]: Hybrid approach (dict for seen, model for OOV)

== Is Our Goal Even Interesting?

This is a question worth asking explicitly. What makes a goal "interesting"?

1. #strong[Novelty]: Character-level frequency prediction is a novel approach, though frequency dictionaries exist.

2. #strong[Practical utility]: OOV handling and RAG cost reduction are useful, though dictionaries work for seen words.

3. #strong[Theoretical insight]: Understanding the structure of language/frequency relationships is interesting, though the structure may be weak.

4. #strong[Compression ratio]: Uncompressed gives 5.6× advantage, but compressed gives only 1.1-1.6× (marginal).

#strong[Verdict]: The goal is interesting IF structure exists and generalization is needed. The compression advantage is marginal, but generalization is the key differentiator - dictionaries cannot handle unseen words, models can.

== The Regularity Assumption

For compression to work, the ICF function must have structure:
- #strong[Regularity]: Similar words → similar ICF
- #strong[Patterns]: Morphology/phonotactics → frequency
- #strong[Redundancy]: Not all word→ICF pairs are independent

If ICF is random (no structure), then K(ICF) = V × 32 bits (cannot compress), and the model must memorize everything. If ICF has structure, then K(ICF) << V × 32 bits (can compress), and generalization is possible.

#strong[Current evidence]: Spearman correlation of 0.18-0.19 suggests we're approaching the theoretical bound for character-level models. This could indicate either:
1. The structure is weak (character patterns provide limited information about frequency)
2. The model is not learning the structure (training issues)
3. We've reached the fundamental limit (information-theoretic bound)

This is an open question that motivates our research. The fact that we're consistently hitting ~0.18-0.19 across multiple experiments suggests we may be hitting a fundamental limit rather than an optimization problem.

