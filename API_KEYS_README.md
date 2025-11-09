# API Keys Configuration Guide

## How to Set Your API Keys

You have **two options** for setting API keys in the ConsistencyAI notebook:

### Option 1: Set Keys Directly in the Notebook (Recommended)

In [main2.ipynb](main2.ipynb), find the "API Key Configuration" cell (cell #5) and uncomment the lines:

```python
# Set your API keys here or via environment variables
# Uncomment and add your keys:
config.set_openrouter_key("your-key-here")
config.set_openai_key("your-key-here")  # Optional

# Example:
config.set_openrouter_key("sk-or-v1-abc123...")
config.set_openai_key("sk-proj-xyz789...")  # Only needed if using OpenAI direct
```

**Important:** If you import duplicity and THEN set keys, you need to restart the kernel:
- Kernel → Restart & Clear Output
- Then re-run all cells

### Option 2: Set Environment Variables (More Secure)

Set environment variables **before** starting Jupyter:

```bash
# In your terminal, before running jupyter notebook
export OPENROUTER_API_KEY="sk-or-v1-abc123..."
export OPENAI_API_KEY="sk-proj-xyz789..."  # Optional

# Then start Jupyter
jupyter notebook main2.ipynb
```

Or create a `.env` file (don't commit it!):

```bash
# .env file
OPENROUTER_API_KEY=sk-or-v1-abc123...
OPENAI_API_KEY=sk-proj-xyz789...
```

Then load it before importing duplicity:

```python
# In notebook cell BEFORE importing duplicity
from dotenv import load_dotenv
load_dotenv()

# Now import duplicity
from duplicity import ...
```

---

## Which API Keys Do You Need?

### Required:
- **OpenRouter API Key** - Routes most models (Anthropic, Google, Mistral, etc.)
  - Get it at: https://openrouter.ai/
  - Used when `ALL_OPEN_ROUTER = True`

### Optional:
- **OpenAI API Key** - For direct OpenAI API access
  - Get it at: https://platform.openai.com/
  - OpenAI models (`openai/gpt-*`) ALWAYS use direct API (even with `ALL_OPEN_ROUTER = True`)
  - Not strictly required if you're only using non-OpenAI models

### Not Used in This Experiment:
- **Google API Key** - Only needed if calling Google Gemini directly
  - Not used when `ALL_OPEN_ROUTER = True` (which is the default)

---

## Verifying Your Keys

After setting keys, check the configuration cell output:

```
✅ OPENROUTER_API_KEY is set
✅ OPENAI_API_KEY is set (optional)
```

If you see warnings, your keys aren't set correctly.

---

## Security Best Practices

1. **Never commit API keys to git**
   - Use `.gitignore` to exclude `.env` files
   - Don't hardcode keys in notebooks you plan to share

2. **Use environment variables for production**
   - More secure than hardcoding
   - Easier to rotate keys

3. **Restart kernel after setting keys**
   - Config is loaded at import time
   - Changes won't take effect until restart
