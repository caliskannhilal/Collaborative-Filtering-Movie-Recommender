# 🎬 Collaborative Filtering Movie Recommender

An end-to-end recommender system project analyzing user behavior and algorithm performance on a sparse movie ratings dataset. Using a 100k subset of the MovieLens dataset, this project compares neighbourhood-based and matrix factorization collaborative filtering methods implemented with the `surprise` library.

---

## 🚀 Project Motivation
Recommendation systems are at the heart of platforms like Netflix and Spotify, yet implementing and benchmarking them can be challenging due to data sparsity and scalability issues.

In this project, I set out to:
- Explore how user–movie interactions behave in a sparse dataset.
- Compare memory-based (KNN) vs. latent-factor (SVD, NMF, etc.) collaborative filtering methods.
- Evaluate performance trade-offs between different algorithms.
- Build a reproducible end-to-end workflow for recommender systems.

---

## 📊 Workflow
1. **Data Preparation**
   - Random sampling of 100k ratings from MovieLens (27M+ original).
   - Ensured reproducibility with fixed seed (`random_state=42`).
   - Data cleaning and basic exploration (missing values, distributions).

2. **Exploratory Data Analysis (EDA)**
   - Distribution of ratings, sparsity patterns, popularity bias.
   - Visualizations using Pandas, Matplotlib, and Seaborn.

3. **Model Training & Evaluation**
   - Models tested:
     - `NormalPredictor` (baseline)
     - `KNNBasic`, `KNNWithMeans`, `KNNBaseline`, `KNNWithZScore`
     - `SVD`, `SVD++`, `NMF`
   - Hyperparameter tuning with `GridSearchCV`.
   - Cross-validation with RMSE scoring.

---

## 🔍 Results & Benchmarks
| Model                  | Best RMSE (sample 100k) | Notes |
|-------------------------|-------------------------|-------|
| Normal Predictor        | ~1.43                  | Baseline |
| KNNWithZScore           | ~0.84                  | Best neighbourhood model |
| SVD                     | ~0.77                  | Strong latent-factor model |
| NMF                     | ~0.82                  | Competitive alternative |
| SVD++                   | In progress            | Exploring performance gains |

(Full results, hyperparameter grids, and plots are available in the notebook)

## ✨ Key Features
- End-to-end pipeline with Pandas + Surprise + Matplotlib
- Configurable train/test split & sampling size
- Automated grid search for KNN & SVD hyperparameters
- Multiple collaborative filtering methods benchmarked
- Rich inline documentation explaining workflow and findings

## 📂 Project Structure
assignmentFINAL.ipynb   # Full workflow: EDA, training, evaluation
MovieLens-Ratings.csv   # Sampled 100k ratings (source: MovieLens)
README.md               # Project documentation

🚧 Limitations & Future Work
	•	Limited to 100k ratings for efficiency → full dataset would improve robustness.
	•	Additional tuning for SVD++ and hybrid methods planned.
	•	Potential extension: implement deep learning recommenders (AutoEncoders, Neural CF).

⸻

👤 Author

This project was developed independently by Hilal Caliskan Egilli as part of my academic and personal exploration into recommender systems.
