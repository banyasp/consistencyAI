# ConsistencyAI Quick Start Guide

## Virtual Environment Setup ✅

Your virtual environment has been successfully set up with all dependencies!

---

## How to Activate the Virtual Environment

### Option 1: Using the activation script (Recommended)
```bash
cd /Users/peterbanyas/Desktop/Cyber/openai/whitehouse/consistencyAI
source activate_venv.sh
```

### Option 2: Manual activation
```bash
cd /Users/peterbanyas/Desktop/Cyber/openai/whitehouse/consistencyAI
source venv/bin/activate
```

You'll know it's activated when you see `(venv)` at the beginning of your terminal prompt.

---

## Set Your API Keys

Before running experiments, you need to set your API keys. Choose one method:

### Method 1: In the Notebook (Easiest)
1. Open `main2.ipynb` in Jupyter
2. Find cell #5 (API Key Configuration)
3. Uncomment and fill in your keys:
   ```python
   config.set_openrouter_key("sk-or-v1-your-key-here")
   config.set_openai_key("sk-proj-your-key-here")  # Optional
   ```
4. **Important:** Restart the kernel (Kernel → Restart & Clear Output)
5. Re-run all cells

### Method 2: Environment Variables (More Secure)
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
export OPENAI_API_KEY="sk-proj-your-key-here"  # Optional
```

Then start Jupyter:
```bash
jupyter notebook main2.ipynb
```

See [API_KEYS_README.md](API_KEYS_README.md) for detailed instructions.

---

## Running the Experiments

### Option A: Jupyter Notebook (Recommended)
```bash
# Activate venv first
source activate_venv.sh

# Start Jupyter
jupyter notebook main2.ipynb
```

Then run cells top-to-bottom:
- **PART 1:** Control Experiment (~30 min, 300 queries)
- **PART 2:** Main Experiment (6-12 hours, 45,000 queries)
- **PART 3:** Variance Analysis (comparison + visualizations)
- **PART 4:** Standard Analyses (heatmaps, clustering, etc.)

### Option B: Python Script (Unattended)
```bash
# Activate venv first
source activate_venv.sh

# Run complete experiment
python run_both_experiments.py
```

This runs everything automatically but takes 6-12 hours.

---

## Resuming Interrupted Experiments

If the main experiment gets interrupted:

1. Open `main2.ipynb`
2. Go to **cell #25** (the resume cell)
3. **Uncomment all lines** in that cell
4. Run cell #25 instead of cell #23
5. It will automatically:
   - Find your latest save
   - Show progress (e.g., "1,234/45,000 complete")
   - Resume from where it left off

---

## Understanding the Output

After running experiments, you'll find:

\`\`\`
logs/
├── control/              # Within-model variance data
│   ├── results_*.json
│   └── similarities_*.pkl
└── main/                 # Across-persona variance data
    ├── results_*.json
    └── similarities_*.pkl

output/
├── variance_comparison/  # NEW! Variance analysis
│   ├── control_variance.csv
│   ├── persona_variance.csv
│   ├── comparison_bar_chart.png
│   ├── comparison_scatter.png
│   ├── comparison_heatmap.png
│   ├── variance_distributions.png
│   └── report.txt       # Read this for key insights!
├── analysis/             # Central analysis CSVs
├── clustering/           # Clustering visualizations
└── heatmap_*.png        # Similarity heatmaps
\`\`\`

**Start by reading:** `output/variance_comparison/report.txt`

---

## Key Concepts

### Control Experiment (Within-Model Variance)
- **What:** Same prompt, 10 times per model
- **Measures:** How consistent is each model with itself?
- **High similarity** = Model is stable/consistent
- **Low similarity** = Model is noisy/random

### Main Experiment (Across-Persona Variance)
- **What:** Different personas, same topic, per model
- **Measures:** Does the model adapt to different demographics?
- **High similarity** = Model ignores persona (gives same answer to everyone)
- **Low similarity** = Model adapts to persona (personalizes responses)

### Ideal Model Profile
✅ **High control similarity** (consistent with itself)
✅ **Lower persona similarity** (persona-aware)
= Stable, reliable behavior that adapts appropriately

---

## Troubleshooting

### "Module not found" error
\`\`\`bash
# Make sure venv is activated (you should see "(venv)" in prompt)
source activate_venv.sh

# Verify installation
python -c "import duplicity; print(duplicity.__version__)"
\`\`\`

### "API key not set" warning
- See [API_KEYS_README.md](API_KEYS_README.md)
- Make sure to restart Jupyter kernel after setting keys

### Experiment interrupted
- Use the resume cell (#25) in main2.ipynb
- Progress is saved in `logs/main/incremental/`

### Out of memory
- Reduce `MAX_CONCURRENCY` in configuration cells
- Close other applications
- Run experiments one at a time

---

## What's Installed

Your virtual environment includes:

**Core Dependencies:**
- `torch` 2.9.0 - Deep learning framework
- `transformers` 4.57.1 - Hugging Face models
- `sentence-transformers` 5.1.2 - Semantic embeddings (SBERT)
- `scikit-learn` 1.7.2 - Clustering and analysis
- `pandas` 2.3.3 - Data manipulation
- `numpy` 2.3.4 - Numerical computing

**Visualization:**
- `matplotlib` 3.10.7
- `seaborn` 0.13.2
- `plotly` 6.4.0

**Async & API:**
- `aiohttp` 3.13.2 - Async HTTP
- `nest-asyncio` 1.6.0 - Jupyter compatibility

**Jupyter:**
- `jupyter` 1.1.1
- `ipykernel` 7.1.0
- `notebook` 7.4.7
- `jupyterlab` 4.4.10

**Total:** 80+ packages, all tested and compatible!

---

## Need Help?

- **Setup issues:** [README.md](README.md)
- **API keys:** [API_KEYS_README.md](API_KEYS_README.md)
- **Configuration:** [config.md](config.md)
- **Paper:** https://arxiv.org/pdf/2510.13852
- **Web app:** https://v0-llm-comparison-webapp.vercel.app/
- **Contact:** peter dot banyas at duke dot edu

---

## Deactivate Virtual Environment

When you're done:
\`\`\`bash
deactivate
\`\`\`

This returns your terminal to normal Python environment.

---

**Built by the Duke Phishermen 🎣**
*Peter Banyas, Shristi Sharma, Alistair Simmons, Atharva Vispute*
