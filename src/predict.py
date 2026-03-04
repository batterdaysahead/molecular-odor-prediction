#!/usr/bin/env python3
"""
Inference script for odor prediction.
Accepts a SMILES string and returns predicted odor descriptors and perceptual ratings.

Pickle-free version - uses JSON for XGBoost, NPZ for scaler, safetensors for regressor.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from xgboost import XGBClassifier
from safetensors import safe_open

warnings.filterwarnings('ignore')

OUTPUT_PATH = Path(__file__).parent.parent / 'models'


def generate_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Generate Morgan fingerprint from SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp).reshape(1, -1)
    except Exception as e:
        print(f"Error generating fingerprint: {e}")
        return None


class PickleFreeScaler:
    """Simple scaler that loads from numpy .npz file."""
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.mean_ = data['mean']
        self.scale_ = data['scale']
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_


class PerceptualMLP(nn.Module):
    """MLP for perceptual ratings."""
    def __init__(self, input_dim=2048, hidden_dims=[1024, 512, 256], output_dim=3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, output_dim))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_safetensors_model(path, model_class, **model_kwargs):
    """Load model from safetensors file."""
    model = model_class(**model_kwargs)
    state_dict = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_models():
    """Load trained models from pickle-free formats."""
    with open(OUTPUT_PATH / 'label_columns.json', 'r') as f:
        label_cols = json.load(f)
    
    with open(OUTPUT_PATH / 'label_thresholds.json', 'r') as f:
        thresholds = json.load(f)
    
    rating_path = OUTPUT_PATH / 'rating_columns.json'
    if rating_path.exists():
        with open(rating_path, 'r') as f:
            rating_cols = json.load(f)
    else:
        rating_cols = []
    
    xgb_path = OUTPUT_PATH / 'xgboost_models'
    estimators = []
    for i in range(len(label_cols)):
        est = XGBClassifier()
        est.load_model(str(xgb_path / f'estimator_{i:03d}.json'))
        estimators.append(est)
    
    scaler = PickleFreeScaler(OUTPUT_PATH / 'scaler.npz')
    
    regressor = None
    regressor_path = OUTPUT_PATH / 'regressor_model.safetensors'
    if regressor_path.exists() and rating_cols:
        regressor = load_safetensors_model(
            regressor_path, 
            PerceptualMLP, 
            input_dim=2048, 
            output_dim=3
        )
    
    return estimators, regressor, scaler, label_cols, rating_cols, thresholds


def predict(smiles: str):
    """Make predictions for a given SMILES string."""
    print(f"\nAnalyzing molecule: {smiles}")
    print("=" * 60)
    
    fp = generate_morgan_fingerprint(smiles)
    if fp is None:
        print("ERROR: Could not generate fingerprint from SMILES")
        return None
    
    try:
        estimators, regressor, scaler, label_cols, rating_cols, thresholds = load_models()
    except Exception as e:
        print(f"ERROR loading models: {e}")
        return None
    
    fp_scaled = scaler.transform(fp)
    
    print("\nOdor descriptors:")
    if estimators and label_cols:
        try:
            probs = []
            for i, label in enumerate(label_cols):
                prob = estimators[i].predict_proba(fp_scaled)[0, 1]
                probs.append((label, prob))
            
            probs_sorted = sorted(probs, key=lambda x: x[1], reverse=True)
            
            top_preds = []
            for label, prob in probs_sorted:
                thresh = thresholds.get(label, 0.5)
                if prob >= thresh:
                    top_preds.append((label, prob))
            
            if top_preds:
                print("  Predicted odor descriptors:")
                for label, prob in top_preds[:10]:
                    thresh = thresholds.get(label, 0.5)
                    print(f"    {label:20s}: {prob:.2f} (threshold: {thresh:.2f})")
            else:
                print("  No descriptors predicted above threshold")
                print("  Top 5 by probability:")
                for label, prob in probs_sorted[:5]:
                    print(f"    {label:20s}: {prob:.2f}")
        except Exception as e:
            print(f"  Error predicting descriptors: {e}")
    else:
        print("  Classifier not available")
    
    print("\nPerceptual ratings (0-1 scale):")
    if regressor is not None and rating_cols:
        try:
            with torch.no_grad():
                fp_tensor = torch.FloatTensor(fp_scaled)
                ratings = regressor(fp_tensor).numpy()[0]
            
            for i, col in enumerate(rating_cols):
                print(f"  {col:15s}: {ratings[i]:.2f}")
        except Exception as e:
            print(f"  Error predicting ratings: {e}")
    else:
        print("  Regressor not available")
    
    print("=" * 60)
    return fp


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <SMILES>")
        print("\nExample:")
        print('  python predict.py "CC(=O)OCC"   # Ethyl acetate')
        print('  python predict.py "CC(C)=O"     # Acetone')
        print('  python predict.py "c1ccc(cc1)C=O"  # Benzaldehyde')
        sys.exit(1)
    
    args = sys.argv[1:]
    smiles = ' '.join(args)
    
    if smiles.startswith('-'):
        print(f"Warning: argument starts with '-', SMILES string expected.")
        print("Treating as SMILES anyway...")
    
    fp = generate_morgan_fingerprint(smiles)
    if fp is None:
        print(f"Error: Could not parse SMILES: {smiles}")
        sys.exit(1)
    
    predict(smiles)


if __name__ == '__main__':
    main()