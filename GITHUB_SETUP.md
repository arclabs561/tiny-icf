# GitHub Repository Setup for tiny-icf

## Status

GitHub API creation failed (permissions issue). Use one of these methods:

## Option 1: GitHub CLI (Recommended)

If you have `gh` CLI installed:

```bash
# Authenticate (if not already)
gh auth login

# Create repository
gh repo create tiny-icf \
  --public \
  --description "Tiny ICF Model: Compressed character-level model for word commonality estimation" \
  --source=. \
  --remote=origin \
  --push
```

## Option 2: GitHub Web UI

1. Go to https://github.com/new
2. Repository name: `tiny-icf`
3. Description: `Tiny ICF Model: Compressed character-level model for word commonality estimation`
4. Public/Private: Choose (public recommended for fun project)
5. **Don't** initialize with README (we already have one)
6. Click "Create repository"

Then connect:

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: tiny-icf - Tiny ICF model for word frequency estimation"

# Add remote
git remote add origin https://github.com/arclabs561/tiny-icf.git

# Push
git branch -M main
git push -u origin main
```

## Option 3: Manual Setup

```bash
# Initialize git
git init

# Create .gitignore if needed
cat > .gitignore << EOF
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.venv/
venv/
env/
*.pt
*.pth
models/
*.log
training_*.log
.DS_Store
*.swp
*.swo
*~
EOF

# Add and commit
git add .
git commit -m "Initial commit: tiny-icf - Complete rename from idf-est"

# Create repo on GitHub (via web), then:
git remote add origin https://github.com/arclabs561/tiny-icf.git
git branch -M main
git push -u origin main
```

## Quick Start (All-in-One)

```bash
cd /Users/arc/Documents/dev/idf-est

# Initialize git
git init

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.venv/
venv/
env/
*.pt
*.pth
models/
*.log
training_*.log
.DS_Store
*.swp
*.swo
*~
EOF

# Add all files
git add .

# Commit
git commit -m "Initial commit: tiny-icf - Tiny ICF model for word frequency estimation"

# Create repo (choose one):
# Option A: GitHub CLI
gh repo create tiny-icf --public --source=. --remote=origin --push

# Option B: Manual (after creating on GitHub web)
# git remote add origin https://github.com/arclabs561/tiny-icf.git
# git branch -M main
# git push -u origin main
```

## Repository Details

- **Name**: `tiny-icf`
- **Description**: Tiny ICF Model: Compressed character-level model for word commonality estimation
- **Visibility**: Public (recommended for fun project)
- **Owner**: arclabs561

## What's Included

- ✅ Complete package: `src/tiny_icf/`
- ✅ Training scripts
- ✅ Text reduction with embeddings
- ✅ Documentation
- ✅ Tests
- ✅ All renamed from `idf-est`

