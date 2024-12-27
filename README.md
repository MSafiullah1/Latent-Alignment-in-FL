Here’s a concise and professional README for your project: 

---

# Federated Learning with Latent Space Alignment Techniques

## Overview
This project investigates the impact of latent space alignment on mitigating data heterogeneity in Federated Learning (FL). Data heterogeneity, a significant challenge in FL, arises due to variations in client data distributions, leading to client drift and degraded model performance. We explore existing alignment methods and propose two novel techniques to improve model robustness and convergence in heterogeneous settings.

---

## Key Features
- **Evaluation of Existing Alignment Methods**:  
  - Kullback-Leibler (KL) Divergence  
  - Maximum Mean Discrepancy (MMD)  
  - Wasserstein Distance  
  - Ridge Regularization  

- **Proposed Novel Techniques**:  
  - **Gradient Harmonization Between Losses (FedGHBL)**: Harmonizes gradients of client-level alignment and classification losses.  
  - **Adversarial Feature Alignment**: Aligns client latent distributions with global latent distributions using adversarial training.  

- **Performance Metrics**:  
  - Classification Accuracy  
  - KL Divergence for latent space alignment evaluation  

---

## Dataset and Experimental Setup
- **Dataset**: CIFAR-10 (split into 80% training and 20% testing)  
- **Clients**: 5 clients with heterogeneous data distributions simulated using a Dirichlet distribution ($\alpha = 0.5$ for label skew, $\alpha = 2$ for quantity skew).  
- **Model**: Tiny VGG trained with a learning rate of 0.001, batch size of 128, over 15 communication rounds with 5 local epochs per round.  

---

## Results
Our experimental results demonstrate:  
1. **Improved Convergence**: Proposed methods significantly reduced client drift compared to baseline techniques.  
2. **Enhanced Robustness**: Models exhibited better alignment and generalization across heterogeneous client data.  

---

## Future Work
- Extend proposed techniques to larger, real-world datasets.  
- Integrate alignment strategies with advanced FL optimization methods.  
- Explore scalability for large-scale federated systems.  

---

## How to Use
1. Clone the repository:  
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run experiments:  
   ```bash
   python main.py
   ```
4. View results: Check the logs and output files for metrics and visualizations.

---

## Contributors
- Muhammad Abubakr Butt  

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

--- 
