# Repository Name Decision

## Analysis Summary

After analyzing 10+ name options, here are the top recommendations:

### 🏆 Top Choice: `tiny-icf`

**Why:**
- ✅ Emphasizes the "tiny model" goal (core differentiator)
- ✅ Clear and professional (good for technical audience)
- ✅ Short and memorable
- ✅ Works for both ICF estimation and text reduction applications
- ✅ Can be explained simply: "Tiny model for ICF estimation"

**Best for:** Technical audience, clear communication, professional projects

### 🎯 Alternative: `word-rare`

**Why:**
- ✅ Most playful and memorable
- ✅ Clear about what it does (predicts word rarity)
- ✅ Good for "fun" project vibe
- ✅ Easy to understand

**Best for:** Fun projects, playful exploration, memorable branding

### 📦 Alternative: `text-shrink`

**Why:**
- ✅ Emphasizes the text reduction use case
- ✅ Fun and memorable
- ✅ Describes a key application (embedding regret minimization)
- ✅ Good for showcasing the text reduction work

**Best for:** Use-case focused, application-driven projects

## Recommendation

**Go with `tiny-icf`** for the best balance of clarity, professionalism, and memorability.

**If you want something more playful:** `word-rare` is a great alternative.

## Migration Plan (if renaming)

### Step 1: Update Package Name
```bash
# Rename package directory
mv src/tiny_icf src/tiny_icf

# Update all imports
find . -name "*.py" -exec sed -i '' 's/tiny_icf/tiny_icf/g' {} +
```

### Step 2: Update Configuration
- `pyproject.toml`: Change `name = "tiny-icf"`
- `README.md`: Update all references
- Scripts: Update module paths

### Step 3: Update Documentation
- All `.md` files with `tiny-icf` references
- Code comments
- Docstrings

### Step 4: Git (if renaming repo)
```bash
# If renaming the repository itself
git remote set-url origin <new-repo-url>
# Or just update locally
```

## Current Status

**Current name:** `tiny-icf`  
**Recommended name:** `tiny-icf`  
**Decision:** Pending user approval

## Quick Comparison

| Name | Clarity | Memorability | Fun Factor | Professional |
|------|---------|--------------|------------|--------------|
| `tiny-icf` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| `word-rare` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `text-shrink` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

