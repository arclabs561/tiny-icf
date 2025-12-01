# Repository Rename: idf-est → tiny-icf

## ✅ Completed Changes

### 1. Package Configuration
- ✅ `pyproject.toml`: Updated `name = "tiny-icf"`
- ✅ Description updated to "Tiny ICF Model"

### 2. Package Directory
- ✅ Renamed `src/idf_est/` → `src/tiny_icf/`
- ✅ All Python files moved

### 3. Code Updates
- ✅ All `import idf_est` → `import tiny_icf`
- ✅ All `from idf_est` → `from tiny_icf`
- ✅ All `idf_est.` → `tiny_icf.`
- ✅ All `-m idf_est` → `-m tiny_icf`

### 4. Documentation
- ✅ `README.md`: Updated title and all references
- ✅ All `.md` files: Updated references
- ✅ All shell scripts: Updated references

### 5. Configuration Files
- ✅ All `.toml` files updated
- ✅ All `.json` files updated (if any)

## 🔄 Next Steps

### GitHub Repository Rename

If you have a GitHub repository, rename it:

**Option 1: Via GitHub Web UI**
1. Go to repository settings
2. Scroll to "Repository name"
3. Change from `idf-est` to `tiny-icf`
4. Click "Rename"

**Option 2: Via GitHub API** (if you have a repo)
```bash
# Get current repo info
gh repo view

# Rename (if using GitHub CLI)
gh repo rename tiny-icf
```

**Option 3: Update Remote URL** (after renaming on GitHub)
```bash
git remote set-url origin https://github.com/USERNAME/tiny-icf.git
```

### Verification

Test the rename:
```bash
# Reinstall package
uv pip install -e .

# Test import
python -c "import tiny_icf; print('✓ Import works')"

# Test CLI
python -m tiny_icf.predict --words "the apple"
```

## 📝 Migration Checklist

- [x] Update pyproject.toml
- [x] Rename package directory
- [x] Update all imports
- [x] Update README
- [x] Update all docs
- [x] Update scripts
- [ ] Rename GitHub repo (if exists)
- [ ] Update git remote URL (if exists)
- [ ] Test installation
- [ ] Test imports
- [ ] Test CLI commands

## 🎯 New Usage

```bash
# Install
uv pip install -e .

# Train
python -m tiny_icf.train --data data/word_frequency.csv --epochs 50

# Predict
python -m tiny_icf.predict --words "the apple xylophone"

# Test
pytest tests/
```

## ✨ Benefits of New Name

1. **Clearer**: "tiny-icf" clearly describes a tiny ICF model
2. **Memorable**: Short and easy to remember
3. **Professional**: Follows common naming patterns
4. **Accurate**: Emphasizes the "tiny model" differentiator

