# Odor Prediction App - Streamlit Cloud Deployment

Machine learning app for predicting odor descriptors and perceptual ratings from molecule SMILES or names.

## Deployment

### Streamlit Cloud

1. **Connect your GitHub repository** to [Streamlit Cloud](https://streamlit.io/cloud)

2. **Configure your app:**
   - Main file path: `odor-app/app.py`
   - Python version: 3.11+
   - Requirements file: `odor-app/requirements.txt`

3. **Model files location:**
   - Models are in `models/` at repository root
   - All models use pickle-free formats for security

### Local Development

```bash
# Activate virtual environment
cd /path/to/project
source venv/bin/activate

# Run the app
cd odor-app
streamlit run app.py
```

## Model Files

The app uses pickle-free model formats:

| Component | Format | File | Size |
|-----------|--------|------|------|
| Classifier | JSON (117 files) | `models/xgboost_models/estimator_*.json` | ~50MB |
| Regressor | Safetensors | `models/regressor_model.safetensors` | 11MB |
| Scaler | NumPy NPZ | `models/scaler.npz` | 33KB |
| Metadata | JSON | `models/*.json` | ~70KB |

**Total model size:** ~72MB

## Features

- **Molecule Input:** Accept SMILES or molecule names (via PubChem)
- **Odor Descriptors:** 117 possible descriptors with tuned thresholds
- **Perceptual Ratings:** Pleasantness, Intensity, Familiarity (0-1 scale)
- **Radar Chart:** Visual odor fingerprint

## Performance

**Classifier (XGBoost):**
- AUC-ROC: 0.84
- Top-20 F1: 0.45 (27% improvement with threshold tuning)
- Full F1: 0.33

**Regressor (MLP):**
- Pleasantness: r = 0.55
- Intensity: r = 0.34
- Familiarity: r = 0.44

## Dependencies

- `streamlit` - Web app framework
- `torch` - PyTorch for regressor
- `xgboost` - Classifier
- `safetensors` - Safe model loading
- `rdkit` - Molecular fingerprints
- `pubchempy` - Molecule name resolution
- `plotly` - Interactive charts

## Model Architecture

### Classifier
- 117 XGBoost binary classifiers (one per odor descriptor)
- Per-label decision thresholds (optimized for F1)
- Morgan fingerprints (2048 bits, radius=2)

### Regressor
- 4-layer MLP (1024 → 512 → 256 → 64 → 3)
- BatchNorm + ReLU + Dropout
- Sigmoid output (0-1 scale)

## License

MIT
