#!/usr/bin/env python3
"""
Streamlit Odor Prediction Web App
Predicts odor descriptors and perceptual ratings from molecule SMILES or names.

Pickle-free model loading for Streamlit Cloud deployment.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from PIL import Image
import pubchempy as pcp
import plotly.graph_objects as go
from safetensors.torch import load_file
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Try to import RDKit Draw (may not be available on Streamlit Cloud)
try:
    from rdkit.Chem import Draw
    HAS_RDKIT_DRAW = True
except Exception:
    HAS_RDKIT_DRAW = False

# ============================================================================
# Path Configuration
# ============================================================================
# Models are at repo root in 'models/' folder
MODELS_DIR = Path(__file__).parent.parent / "models"

REGRESSOR_PATH = MODELS_DIR / "regressor_model.safetensors"
SCALER_PATH = MODELS_DIR / "scaler.npz"
XGBOOST_MODELS_DIR = MODELS_DIR / "xgboost_models"
CONFIG_PATH = MODELS_DIR / "config.json"
FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.json"
LABEL_COLS_PATH = MODELS_DIR / "label_columns.json"
LABEL_THRESHOLDS_PATH = MODELS_DIR / "label_thresholds.json"
RATING_COLS_PATH = MODELS_DIR / "rating_columns.json"


# ============================================================================
# Model Classes
# ============================================================================
class PerceptualMLP(torch.nn.Module):
    """Multi-layer perceptron for perceptual ratings prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [1024, 512, 256], output_dim: int = 3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(torch.nn.Linear(prev_dim, 64))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(64, output_dim))
        layers.append(torch.nn.Sigmoid())
        
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# Cached Resource Loaders
# ============================================================================
@st.cache_resource
def load_config():
    """Load model configuration."""
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


@st.cache_resource
def load_classifier():
    """Load XGBoost multi-label classifier from JSON files."""
    from xgboost import XGBClassifier
    
    # Load the first estimator to get the model structure
    first_model_path = XGBOOST_MODELS_DIR / "estimator_000.json"
    
    # Create a MultiOutputClassifier wrapper
    # We need to load all 117 estimators
    classifiers = []
    model_files = sorted(XGBOOST_MODELS_DIR.glob("estimator_*.json"))
    
    for model_path in model_files:
        clf = XGBClassifier()
        clf.load_model(str(model_path))
        classifiers.append(clf)
    
    # Create a simple wrapper class
    class MultiOutputWrapper:
        def __init__(self, estimators):
            self.estimators_ = estimators
        
        def predict_proba(self, X):
            """Return probability of positive class for each estimator."""
            probs = []
            for clf in self.estimators_:
                prob = clf.predict_proba(X)[0, 1]
                probs.append(prob)
            return np.array(probs).reshape(1, -1)
    
    return MultiOutputWrapper(classifiers)


@st.cache_resource
def load_regressor():
    """Load PyTorch MLP regressor from safetensors format."""
    # Load scaler from npz
    scaler_data = np.load(SCALER_PATH)
    scaler_mean = scaler_data['mean']
    scaler_scale = scaler_data['scale']
    
    class SimpleScaler:
        def transform(self, X):
            return (X - scaler_mean) / scaler_scale
    
    scaler = SimpleScaler()
    
    # Load regressor from safetensors
    state_dict = load_file(str(REGRESSOR_PATH))
    
    # Convert to regular state dict format (remove 'network.' prefix if present)
    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_state_dict[key] = torch.tensor(value)
    
    input_dim = 2048
    output_dim = 3
    regressor = PerceptualMLP(input_dim=input_dim, output_dim=output_dim)
    
    # Try to load with CPU first (for Streamlit Cloud)
    device = torch.device("cpu")
    regressor.load_state_dict(clean_state_dict)
    regressor.to(device)
    regressor.eval()
    
    return regressor, scaler, device


@st.cache_resource
def load_metadata():
    """Load feature and label column names and tuned thresholds."""
    with open(FEATURE_COLS_PATH, 'r') as f:
        feature_cols = json.load(f)
    
    with open(LABEL_COLS_PATH, 'r') as f:
        label_cols = json.load(f)
    
    rating_cols = []
    if RATING_COLS_PATH.exists():
        with open(RATING_COLS_PATH, 'r') as f:
            rating_cols = json.load(f)
    
    # Load tuned thresholds (if available)
    thresholds = {}
    if LABEL_THRESHOLDS_PATH.exists():
        with open(LABEL_THRESHOLDS_PATH, 'r') as f:
            thresholds = json.load(f)
    
    return feature_cols, label_cols, rating_cols, thresholds


# ============================================================================
# Helper Functions
# ============================================================================
def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Generate Morgan fingerprint from SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        fp_array = np.array(fp).reshape(1, -1)
        return fp_array, mol
    except Exception as e:
        st.error(f"Error generating fingerprint: {e}")
        return None, None


def resolve_molecule(user_input: str):
    """
    Resolve molecule input (SMILES or name) to SMILES, name, CID, and structure.
    Returns dict with keys: smiles, name, cid, mol, image, error
    """
    # Try parsing as SMILES first
    mol = Chem.MolFromSmiles(user_input)
    if mol is not None:
        # Valid SMILES - try to find compound info
        try:
            # Search PubChem by SMILES
            compound = pcp.get_compounds(user_input, 'smiles')[0]
            image = None
            if HAS_RDKIT_DRAW:
                try:
                    image = Draw.MolToImage(mol, size=(300, 300))
                except Exception:
                    pass
            return {
                'smiles': compound.smiles,
                'name': compound.iupac_name or compound.synonyms[0] if compound.synonyms else "Unknown",
                'cid': compound.cid,
                'mol': mol,
                'image': image,
                'error': None
            }
        except Exception:
            # SMILES valid but not in PubChem
            image = None
            if HAS_RDKIT_DRAW:
                try:
                    image = Draw.MolToImage(mol, size=(300, 300))
                except Exception:
                    pass
            return {
                'smiles': user_input,
                'name': "Custom Molecule",
                'cid': None,
                'mol': mol,
                'image': image,
                'error': None
            }
    
    # Not a valid SMILES - try as molecule name
    try:
        compound = pcp.get_compounds(user_input, 'name')[0]
        mol = Chem.MolFromSmiles(compound.smiles)
        if mol is None:
            return {
                'smiles': None,
                'name': None,
                'cid': None,
                'mol': None,
                'image': None,
                'error': "Could not generate structure from PubChem data"
            }
        image = None
        if HAS_RDKIT_DRAW:
            try:
                image = Draw.MolToImage(mol, size=(300, 300))
            except Exception:
                pass
        return {
            'smiles': compound.smiles,
            'name': compound.iupac_name or user_input,
            'cid': compound.cid,
            'mol': mol,
            'image': image,
            'error': None
        }
    except Exception as e:
        return {
            'smiles': None,
            'name': None,
            'cid': None,
            'mol': None,
            'image': None,
            'error': f"Molecule '{user_input}' not found in PubChem. Please check the spelling or try a SMILES string."
        }


def predict_descriptors(fp_scaled, classifier, label_cols, thresholds):
    """Predict odor descriptors using XGBoost classifier with tuned thresholds.
    
    Args:
        fp_scaled: Scaled fingerprint features
        classifier: XGBoost multi-label classifier
        label_cols: List of label names
        thresholds: Dict of per-label decision thresholds
    
    Returns:
        List of (label, probability, threshold, above_threshold) tuples sorted by probability
    """
    # Get all probabilities at once
    probs_array = classifier.predict_proba(fp_scaled)[0]
    
    probs = []
    for i, prob in enumerate(probs_array):
        label = label_cols[i] if i < len(label_cols) else f"Label_{i}"
        threshold = thresholds.get(label, 0.5)  # Default 0.5 if no threshold
        probs.append((label, float(prob), float(threshold), prob >= threshold))
    
    # Sort by probability
    probs_sorted = sorted(probs, key=lambda x: x[1], reverse=True)
    return probs_sorted


def predict_ratings(fp_scaled, regressor, scaler, device, rating_cols):
    """Predict perceptual ratings using PyTorch regressor."""
    with torch.no_grad():
        fp_tensor = torch.FloatTensor(fp_scaled).to(device)
        ratings = regressor(fp_tensor).cpu().numpy()[0]
    
    return {rating_cols[i]: float(ratings[i]) for i in range(len(rating_cols))}


def get_confidence_color(prob):
    """Return color based on confidence level."""
    if prob >= 0.7:
        return "#22c55e", "high"  # Green
    elif prob >= 0.5:
        return "#f97316", "medium"  # Orange
    else:
        return "#6b7280", "low"  # Grey


def create_radar_chart(top_descriptors, n=8):
    """Create Plotly radar chart for top N descriptors.
    
    Args:
        top_descriptors: List of (label, prob, threshold, above_threshold) tuples
        n: Number of top descriptors to include in chart
    """
    top_n = top_descriptors[:n]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=[prob for _, prob, _, _ in top_n],
        theta=[label for label, _, _, _ in top_n],
        fill='toself',
        line=dict(color='#6366f1', width=2),
        fillcolor='rgba(99, 102, 241, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%',
                gridcolor='#e5e7eb',
                linecolor='#9ca3af'
            )
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=400
    )
    
    return fig


# ============================================================================
# Streamlit UI
# ============================================================================
def main():
    st.set_page_config(
        page_title="Odor Prediction",
        page_icon="👃",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #e5e7eb;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("👃 Odor Prediction")
        st.markdown("""
        Predict odor descriptors and perceptual ratings 
        from molecule SMILES or names using machine learning.
        """)
        
        st.divider()
        
        # Load config for display
        config = load_config()
        st.subheader("Model Info")
        st.markdown(f"""
        **{config.get('model_name', 'Odor Prediction')}**
        - Labels: {config.get('num_labels', 'N/A')} descriptors
        - Ratings: {', '.join(config.get('ratings', []))}
        - Features: Morgan FP ({config.get('feature_dim', 'N/A')} bits)
        """)
        
        # Show metrics if available
        if 'metrics' in config:
            st.divider()
            st.subheader("Performance")
            metrics = config['metrics']
            if 'classifier' in metrics:
                clf_metrics = metrics['classifier']
                st.markdown(f"""
                **Classifier**
                - AUC-ROC: {clf_metrics.get('mean_auc_roc', 'N/A')}
                - Top-20 F1: {clf_metrics.get('top20_macro_f1', 'N/A')}
                """)
            if 'regressor' in metrics:
                reg_metrics = metrics['regressor']
                st.markdown(f"""
                **Regressor**
                - Pleasantness: r = {reg_metrics.get('pleasantness_pearson_r', 'N/A')}
                - Intensity: r = {reg_metrics.get('intensity_pearson_r', 'N/A')}
                - Familiarity: r = {reg_metrics.get('familiarity_pearson_r', 'N/A')}
                """)
        
        st.divider()
        
        st.info("Perceptual ratings based on ~500 human ratings from Keller et al.")
        
        st.caption("✨ Top-20 F1 improved by 27% with threshold tuning")
        
        st.markdown("---")
        st.markdown("Built with Streamlit, RDKit, PyTorch & XGBoost")
    
    # Main content
    st.title("👃 Odor Prediction")
    st.markdown("Enter a molecule name or SMILES string to predict its odor profile")
    
    # Example molecules
    examples = [
        "vanillin",
        "ethyl acetate",
        "CC(=O)OCC",
        "benzaldehyde",
        "limonene",
        "cinnamaldehyde"
    ]
    
    # Input section
    user_input = st.text_input(
        "Molecule",
        placeholder="e.g., vanillin or CC(=O)OCC (ethyl acetate)",
        help="Enter a molecule name (e.g., 'vanillin') or SMILES string (e.g., 'CC(=O)OCC')"
    )
    
    # Show examples as chips
    st.markdown("Try: " + " ".join([f"`{ex}`" for ex in examples]))
    
    if not user_input:
        st.stop()
    
    # Load models
    with st.spinner("Loading models..."):
        try:
            classifier = load_classifier()
            regressor, scaler, device = load_regressor()
            feature_cols, label_cols, rating_cols, thresholds = load_metadata()
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.stop()
    
    # Resolve molecule
    with st.spinner("Looking up molecule..."):
        mol_data = resolve_molecule(user_input)
    
    if mol_data['error']:
        st.error(mol_data['error'])
        st.stop()
    
    # Display molecule info
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if mol_data['image']:
            st.image(mol_data['image'], caption=mol_data['name'], width="stretch")
        else:
            st.info("🧪 Structure visualization not available")
    
    with col2:
        st.subheader(mol_data['name'])
        st.code(mol_data['smiles'], language="text")
        if mol_data['cid']:
            st.markdown(f"**PubChem CID:** [{mol_data['cid']}](https://pubchem.ncbi.nlm.nih.gov/compound/{mol_data['cid']})")
    
    # Generate fingerprint
    fp_array, mol = smiles_to_fingerprint(mol_data['smiles'])
    if fp_array is None:
        st.error("Failed to generate molecular fingerprint")
        st.stop()
    
    # Scale features
    fp_scaled = scaler.transform(fp_array)
    
    # Predict odor descriptors
    st.divider()
    st.subheader("🎯 Predicted Odor Descriptors")
    
    top_predictions = predict_descriptors(fp_scaled, classifier, label_cols, thresholds)
    
    # Filter predictions above threshold
    above_threshold = [(label, prob, thresh) for label, prob, thresh, above in top_predictions if above]
    
    # If fewer than threshold, show top 10 by probability
    if len(above_threshold) < 3:
        above_threshold = [(label, prob, thresh) for label, prob, thresh, _ in top_predictions[:10]]
    
    if above_threshold:
        st.markdown("**Top predicted descriptors** (above tuned threshold)")
        
        for label, prob, thresh in above_threshold[:10]:
            color, level = get_confidence_color(prob)
            threshold_indicator = "✅" if prob >= thresh else "⚠️"
            
            # Create custom progress bar with color and threshold marker
            bar_html = f"""
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="font-weight: 500;">{threshold_indicator} {label}</span>
                    <span style="color: {color}; font-weight: bold;">{prob:.0%}</span>
                </div>
                <div style="background-color: #e5e7eb; border-radius: 0.25rem; height: 0.5rem; overflow: hidden; position: relative;">
                    <div style="background-color: {color}; width: {prob*100}%; height: 100%; transition: width 0.3s;"></div>
                    <div style="position: absolute; left: {thresh*100}%; top: 0; bottom: 0; width: 2px; background-color: #ef4444;"></div>
                </div>
                <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.25rem;">Threshold: {thresh:.0%}</div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
    else:
        st.info("No odor descriptors predicted above threshold")
        st.markdown("**Top 5 predictions:**")
        for label, prob, thresh, _ in top_predictions[:5]:
            st.write(f"- {label}: {prob:.0%} (threshold: {thresh:.0%})")
    
    # Predict perceptual ratings
    st.divider()
    st.subheader("📊 Perceptual Ratings")
    
    ratings = predict_ratings(fp_scaled, regressor, scaler, device, rating_cols)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Pleasantness",
            value=f"{ratings['Pleasantness']:.2f}",
            help="0 = Very unpleasant, 1 = Very pleasant"
        )
    
    with col2:
        st.metric(
            label="Intensity",
            value=f"{ratings['Intensity']:.2f}",
            help="0 = Very weak, 1 = Very strong"
        )
    
    with col3:
        st.metric(
            label="Familiarity",
            value=f"{ratings['Familiarity']:.2f}",
            help="0 = Unfamiliar, 1 = Very familiar"
        )
    
    st.caption("Based on ~500 human ratings from Keller et al.")
    
    # Radar chart
    st.divider()
    st.subheader("🕸️ Odor Fingerprint")
    
    fig = create_radar_chart(top_predictions, n=8)
    st.plotly_chart(fig, width="stretch")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
            Odor Prediction App | Powered by XGBoost & PyTorch | Morgan Fingerprints (2048 bits)
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
