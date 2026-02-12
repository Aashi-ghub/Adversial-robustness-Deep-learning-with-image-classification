"""
Streamlit Demo Application for Adversarial Robustness

This interactive web application demonstrates:
- Image classification on CIFAR-10
- Adversarial attack generation (FGSM and PGD)
- Comparison of standard vs robust models
- Real-time visualization of attacks
"""

import os
import sys
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import get_model
from utils.data_loader import get_cifar10_dataloaders, denormalize, CIFAR10_CLASSES
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


# Page config
st.set_page_config(
    page_title="Adversarial Robustness Demo",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_data():
    """Load CIFAR-10 dataset (cached)"""
    _, test_loader, _, testset = get_cifar10_dataloaders(batch_size=1, num_workers=0)
    return test_loader, testset


@st.cache_resource
def load_models(device='cpu'):
    """Load trained models (cached)"""
    models = {}
    
    # Try to load standard model
    standard_path = 'checkpoints/best_model_standard.pth'
    if os.path.exists(standard_path):
        model = get_model(device=device)
        checkpoint = torch.load(standard_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models['standard'] = model
    
    # Try to load adversarially trained model
    robust_path = 'checkpoints/best_model_adversarial.pth'
    if os.path.exists(robust_path):
        model = get_model(device=device)
        checkpoint = torch.load(robust_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models['robust'] = model
    
    # If no models available, create a fresh one (for demo purposes)
    if not models:
        st.warning("⚠️ No trained models found. Using a randomly initialized model for demonstration.")
        models['demo'] = get_model(device=device)
        models['demo'].eval()
    
    return models


def tensor_to_image(tensor):
    """Convert tensor to displayable image"""
    img = denormalize(tensor.cpu()).numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return img


def create_comparison_plot(clean_img, adv_img, clean_pred, adv_pred, true_label, attack_name, epsilon):
    """Create side-by-side comparison of clean and adversarial images"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Clean image
    ax1.imshow(clean_img)
    ax1.set_title(f'Clean Image\nTrue: {CIFAR10_CLASSES[true_label]}\nPredicted: {CIFAR10_CLASSES[clean_pred]}', 
                  fontsize=11)
    ax1.axis('off')
    
    # Adversarial image
    ax2.imshow(adv_img)
    color = 'green' if adv_pred == true_label else 'red'
    ax2.set_title(f'{attack_name} (ε={epsilon})\nTrue: {CIFAR10_CLASSES[true_label]}\nPredicted: {CIFAR10_CLASSES[adv_pred]}', 
                  fontsize=11, color=color)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig


def main():
    # Title and description
    st.title("🛡️ Adversarial Robustness Demo")
    st.markdown("""
    This interactive demo showcases adversarial attacks on image classification models 
    and the effectiveness of adversarial training as a defense mechanism.
    
    **What are Adversarial Attacks?** Small, imperceptible perturbations added to images 
    that can fool neural networks into making incorrect predictions.
    """)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        test_loader, testset = load_data()
        models = load_models(device)
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_names = list(models.keys())
    model_name = st.sidebar.selectbox(
        "Select Model",
        model_names,
        format_func=lambda x: {
            'standard': 'Standard Training',
            'robust': 'Adversarial Training',
            'demo': 'Demo Model (Untrained)'
        }.get(x, x)
    )
    model = models[model_name]
    
    # Attack selection
    attack_type = st.sidebar.selectbox(
        "Attack Type",
        ["FGSM", "PGD"]
    )
    
    # Epsilon slider
    epsilon = st.sidebar.slider(
        "Attack Strength (ε)",
        min_value=0.0,
        max_value=0.2,
        value=0.03,
        step=0.01,
        help="Higher values create stronger attacks but more visible perturbations"
    )
    
    # PGD specific parameters
    if attack_type == "PGD":
        num_iter = st.sidebar.slider(
            "PGD Iterations",
            min_value=1,
            max_value=20,
            value=10,
            help="More iterations create stronger attacks"
        )
    
    # Image selection
    image_idx = st.sidebar.number_input(
        "Image Index",
        min_value=0,
        max_value=len(testset)-1,
        value=0,
        help="Select an image from the CIFAR-10 test set"
    )
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Image Classification")
        
        # Get image and label
        image, label = testset[image_idx]
        image_tensor = image.unsqueeze(0).to(device)
        
        # Clean prediction
        with torch.no_grad():
            clean_output = model(image_tensor)
            clean_prob = torch.softmax(clean_output, dim=1)
            clean_pred = clean_output.argmax(1).item()
            clean_confidence = clean_prob[0, clean_pred].item()
        
        # Display clean image
        clean_img = tensor_to_image(image)
        st.image(clean_img, caption=f"Original Image", use_container_width=True)
        
        # Prediction info
        st.write(f"**True Label:** {CIFAR10_CLASSES[label]}")
        st.write(f"**Predicted:** {CIFAR10_CLASSES[clean_pred]}")
        st.write(f"**Confidence:** {clean_confidence*100:.2f}%")
        
        # Show top-3 predictions
        top_probs, top_indices = torch.topk(clean_prob, 3)
        st.write("**Top 3 Predictions:**")
        for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
            st.write(f"{i+1}. {CIFAR10_CLASSES[idx]}: {prob.item()*100:.2f}%")
    
    with col2:
        st.subheader("⚔️ Adversarial Attack")
        
        if epsilon == 0.0:
            st.info("Set epsilon > 0 to generate adversarial examples")
            st.image(clean_img, caption="No Attack (ε=0)", use_container_width=True)
        else:
            # Generate adversarial example
            with st.spinner(f"Generating {attack_type} attack..."):
                label_tensor = torch.tensor([label]).to(device)
                
                if attack_type == "FGSM":
                    adv_image = fgsm_attack(model, image_tensor, label_tensor, epsilon=epsilon)
                else:  # PGD
                    adv_image = pgd_attack(model, image_tensor, label_tensor, 
                                          epsilon=epsilon, num_iter=num_iter)
            
            # Adversarial prediction
            with torch.no_grad():
                adv_output = model(adv_image)
                adv_prob = torch.softmax(adv_output, dim=1)
                adv_pred = adv_output.argmax(1).item()
                adv_confidence = adv_prob[0, adv_pred].item()
            
            # Display adversarial image
            adv_img = tensor_to_image(adv_image[0])
            st.image(adv_img, caption=f"{attack_type} Attack (ε={epsilon})", use_container_width=True)
            
            # Prediction info
            st.write(f"**True Label:** {CIFAR10_CLASSES[label]}")
            st.write(f"**Predicted:** {CIFAR10_CLASSES[adv_pred]}")
            st.write(f"**Confidence:** {adv_confidence*100:.2f}%")
            
            # Attack success
            if adv_pred != label:
                st.error("🎯 **Attack Successful!** Model was fooled.")
            else:
                st.success("🛡️ **Attack Failed!** Model is robust.")
            
            # Show top-3 predictions
            top_probs, top_indices = torch.topk(adv_prob, 3)
            st.write("**Top 3 Predictions:**")
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                st.write(f"{i+1}. {CIFAR10_CLASSES[idx]}: {prob.item()*100:.2f}%")
            
            # Show perturbation
            perturbation = adv_image[0] - image_tensor[0]
            perturbation_magnitude = torch.norm(perturbation).item()
            st.write(f"**Perturbation L2 Norm:** {perturbation_magnitude:.4f}")
    
    # Additional Information
    st.markdown("---")
    st.subheader("ℹ️ About")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **FGSM Attack**
        - Fast Gradient Sign Method
        - Single-step attack
        - Adds ε × sign(gradient)
        - Fast but less powerful
        """)
    
    with col2:
        st.markdown("""
        **PGD Attack**
        - Projected Gradient Descent
        - Iterative attack
        - Multiple small steps
        - Stronger than FGSM
        """)
    
    with col3:
        st.markdown("""
        **Adversarial Training**
        - Train on adversarial examples
        - Improves robustness
        - May reduce clean accuracy
        - Effective defense method
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Dataset:** CIFAR-10 (10 classes, 32×32 RGB images)
    
    **Model:** Custom CNN with 3 convolutional blocks
    
    **More Info:** [GitHub Repository](https://github.com/Aashi-ghub/Adversial-robustness-Deep-learning-with-image-classification)
    """)


if __name__ == "__main__":
    main()
