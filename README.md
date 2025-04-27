# Diabetes Detection using Machine Learning

This project investigates the impact of feature transformation techniques on the performance of various machine learning (ML) models for early detection of diabetes.

## Project Description
- Dataset: Pima Indians Diabetes Dataset (768 samples, 8 features)
- Techniques: No Transformation, Normalization, Min-Max Scaling
- Models Tested: Decision Tree, Random Forest, SVM, LGBM, XGBoost, etc.
- Best Result: LGBM with Min-Max Scaling achieving 82.91% accuracy.

## How to Run
1. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the main script:
    ```bash
    python main.py
    ```

## Folder Structure
- `/data/` - sample dataset
- `/images/` - figures like feature distribution and correlation matrix
- `/models/` - trained models (optional)

## Future Work
- Validate models on real-world data
- Add explainability (SHAP, LIME)
- Explore deployment in clinical settings

---

© Stuti Kataria, SPSU
