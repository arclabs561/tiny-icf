# Repository Name Options

## Current Name: `tiny-icf`

**Pros:**
- Clear and technical
- Describes what it does (IDF estimation)

**Cons:**
- Not very memorable
- Doesn't convey the "fun" aspect
- Doesn't hint at text reduction/embedding regret use case
- "est" suffix is a bit generic

## Name Options

### Option 1: `tiny-icf`
**Description:** Tiny ICF model

**Pros:**
- Emphasizes the "tiny model" goal
- Clear and descriptive
- Short and memorable

**Cons:**
- Still technical
- Doesn't hint at applications

### Option 2: `word-rare`
**Description:** Predicts word rarity/informativeness

**Pros:**
- Playful and memorable
- Describes the core function
- Easy to understand

**Cons:**
- Might be confused with "rare word detection"
- Doesn't emphasize model size

### Option 3: `text-shrink`
**Description:** Text reduction using ICF

**Pros:**
- Fun and descriptive
- Emphasizes the text reduction use case
- Memorable

**Cons:**
- Doesn't mention ICF/frequency
- Might imply general text compression

### Option 4: `byte-rare`
**Description:** Byte-level rarity estimation

**Pros:**
- Emphasizes byte-level processing
- Clear about rarity prediction
- Short and catchy

**Cons:**
- "byte" might be too technical
- Doesn't hint at text reduction

### Option 5: `icf-tiny`
**Description:** Tiny ICF estimator

**Pros:**
- Clear about model size and function
- Professional yet approachable
- Good for technical audience

**Cons:**
- Similar to current name
- Not very playful

### Option 6: `word-weight`
**Description:** Weight words by informativeness

**Pros:**
- Describes the use case (weighting)
- Clear and practical
- Easy to understand

**Cons:**
- Doesn't emphasize model size
- Might imply general weighting (not frequency-based)

### Option 7: `embed-regret`
**Description:** Minimize embedding regret with ICF

**Pros:**
- Unique and memorable
- Describes the text reduction application
- Technical but interesting

**Cons:**
- "regret" might be confusing
- Doesn't mention ICF/frequency

### Option 8: `rare-words`
**Description:** Predicts rare words

**Pros:**
- Simple and clear
- Easy to understand
- Memorable

**Cons:**
- Might imply word list, not model
- Doesn't emphasize applications

### Option 9: `text-info`
**Description:** Text informativeness estimator

**Pros:**
- Describes the core concept (informativeness)
- Clear and professional
- Good for technical audience

**Cons:**
- A bit generic
- Doesn't hint at model size

### Option 10: `icf-model`
**Description:** ICF prediction model

**Pros:**
- Clear and direct
- Professional
- Easy to find

**Cons:**
- Very generic
- Not memorable
- Doesn't convey "fun" aspect

### Option 11: `wordfreq-tiny` (inspired by wordfreq library)
**Description:** Tiny word frequency estimator

**Pros:**
- References established library (wordfreq)
- Clear about size and function
- Professional

**Cons:**
- Might be confused with wordfreq itself
- Less unique
- Doesn't emphasize ICF specifically

### Option 12: `freq-est-tiny`
**Description:** Tiny frequency estimator

**Pros:**
- Clear and descriptive
- Emphasizes size
- Professional

**Cons:**
- Generic
- Doesn't mention ICF
- Not very memorable

### Option 13: `icf-nano`
**Description:** Nano-sized ICF model

**Pros:**
- Emphasizes extreme smallness (nano < tiny)
- Clear about ICF
- Memorable

**Cons:**
- "nano" might be too extreme
- Less common term

### Option 14: `word-info`
**Description:** Word informativeness predictor

**Pros:**
- Describes the core concept
- Clear and professional
- Easy to understand

**Cons:**
- Doesn't emphasize model size
- A bit generic

### Option 15: `byte-icf`
**Description:** Byte-level ICF estimator

**Pros:**
- Emphasizes byte-level processing (key differentiator)
- Clear about ICF
- Technical but clear

**Cons:**
- "byte" might be too technical for some
- Doesn't hint at applications

### Option 16: `micro-icf`
**Description:** Micro-sized ICF model

**Pros:**
- Emphasizes small size (micro < tiny)
- Clear about ICF
- Professional

**Cons:**
- Similar to tiny-icf
- Less common term

### Option 17: `icf-compact`
**Description:** Compact ICF estimator

**Pros:**
- Emphasizes compactness
- Professional
- Clear

**Cons:**
- Doesn't emphasize extreme smallness
- Less memorable

### Option 18: `word-weight-icf`
**Description:** ICF-based word weighting

**Pros:**
- Describes use case (weighting)
- Mentions ICF
- Clear

**Cons:**
- Longer name
- Less catchy

### Option 19: `rare-word-est`
**Description:** Rare word estimator

**Pros:**
- Clear about predicting rarity
- Memorable
- Easy to understand

**Cons:**
- Doesn't emphasize model size
- "est" suffix is generic

### Option 20: `icf-light`
**Description:** Lightweight ICF model

**Pros:**
- Emphasizes lightweight nature
- Professional
- Clear

**Cons:**
- Doesn't emphasize extreme smallness
- Less memorable

## Research-Based Insights

From research on similar projects:
- **wordfreq**: Established library for word frequency (40+ languages)
- Common patterns: `{function}-{size}` or `{size}-{function}`
- Tiny models often use: `tiny-`, `nano-`, `micro-`, `mini-`, `light-`
- Frequency estimation: `freq`, `frequency`, `icf`, `idf`
- Word processing: `word`, `text`, `token`

## Recommendation

**Top 5 Options (Updated):**

1. **`tiny-icf`** ⭐⭐⭐
   - Best balance of clarity and memorability
   - Emphasizes the "tiny model" goal
   - Professional yet approachable
   - Good for technical audience
   - Follows common naming pattern (`{size}-{function}`)

2. **`word-rare`** ⭐⭐⭐
   - Most playful and memorable
   - Clear about what it does
   - Good for "fun" project vibe
   - Easy to understand
   - Unique and catchy

3. **`byte-icf`** ⭐⭐
   - Emphasizes byte-level processing (key differentiator)
   - Clear about ICF
   - Technical but clear
   - Highlights unique architecture

4. **`icf-nano`** ⭐⭐
   - Emphasizes extreme smallness
   - Clear about ICF
   - Memorable
   - More extreme than "tiny"

5. **`text-shrink`** ⭐⭐
   - Emphasizes the text reduction use case
   - Fun and memorable
   - Describes a key application
   - Good for showcasing the embedding regret work

## Final Recommendation: `tiny-icf`

**Why:**
- Clear and professional (good for technical audience)
- Emphasizes the "tiny model" goal (core differentiator)
- Short and memorable
- Works well for both the ICF estimation and text reduction applications
- Can be explained: "Tiny model for ICF estimation"

**Alternative:** If you want something more playful, go with `word-rare`.

## Migration Steps (if renaming)

1. Update `pyproject.toml`:
   ```toml
   name = "tiny-icf"  # or chosen name
   ```

2. Update `README.md`:
   - Change all references from `tiny-icf` to new name
   - Update installation instructions

3. Update package name:
   - Rename `src/tiny_icf/` to `src/tiny_icf/` (or equivalent)
   - Update all imports

4. Update scripts:
   - Change module references
   - Update paths

5. Git:
   ```bash
   git mv tiny-icf tiny-icf  # if renaming repo
   ```

## Decision Criteria

Consider:
- **Clarity**: Does it clearly describe what the project does?
- **Memorability**: Is it easy to remember?
- **Fun Factor**: Does it match the "fun project" vibe?
- **Professionalism**: Is it appropriate for technical audience?
- **Uniqueness**: Is it easy to find/search for?

