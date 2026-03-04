---
license: mit
tags:
- chemistry
- odor-prediction
- molecular-properties
- xgboost
- pytorch
library_name: sklearn
pipeline_tag: tabular-classification
---

# Odor Prediction Model

Predict odor descriptors and perceptual ratings from a molecule's SMILES string.

**What it does:**
- Predicts 112 odor descriptors (fruity, floral, woody, sweet, etc.)
- Predicts 3 perceptual ratings (Pleasantness, Intensity, Familiarity)

## Results

| Task | Metric | Score |
|------|--------|-------|
| Odor Classification | Top-20 Macro F1 | 0.45 |
| Odor Classification | Mean AUC-ROC | 0.84 |
| Pleasantness | Pearson r | 0.55 |
| Intensity | Pearson r | 0.34 |
| Familiarity | Pearson r | 0.44 |

## Quick Start

```bash
pip install torch xgboost rdkit numpy
```

```python
from src.predict import predict

predict("c1ccc(cc1)C=O")  # Benzaldehyde - smells like almonds
```

Output:
```
Odor descriptors:
  sweet      : 0.98
  spicy      : 0.98
  almond     : 0.98
  floral     : 0.96
  green      : 0.93

Perceptual ratings (0-1 scale):
  Pleasantness: 0.99
  Intensity   : 0.89
  Familiarity : 0.92
```

## Data Sources

All training data comes from the [Pyrfume](https://github.com/pyrfume/pyrfume-data) project:

| Dataset | What it contains |
|---------|------------------|
| GoodScents | ~4,500 molecules with expert odor annotations |
| Leffingwell | ~3,200 molecules with odor descriptors |
| Keller et al. 2016/2012 | Perceptual ratings (pleasantness, intensity, familiarity) |
| MA et al. 2021 | Additional perceptual ratings |
| Arshamian et al. 2022 | Pleasantness rankings |

## Model Details

**Classifier (odor descriptors):**
- XGBoost with 300 trees
- Morgan fingerprints (2048-bit) from RDKit
- Per-label threshold tuning

**Regressor (perceptual ratings):**
- PyTorch MLP (1024-512-256 hidden layers)
- Separate models for each rating

## Limitations

- Only works on single molecules, not mixtures
- Intensity predictions are weaker (r=0.34)
- Perceptual ratings trained on ~500 molecules only

## Citation

```bibtex
@misc{molecular-odor-prediction,
  title={Molecular Odor Prediction},
  author={batterdaysahead},
  year={2026},
  url={https://huggingface.co/batterdaysahead/molecular-odor-prediction},
}
```

## License

MIT
