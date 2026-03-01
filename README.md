# StyleSense: Fashion Forward Forecasting

### Data Science Product Recommendation Pipeline

## Project Overview
StyleSense is a rapidly growing online women's clothing retailer. With a surge in new customers, a backlog of product reviews has emerged where the "Recommendation" indicator is missing. This project leverages historical review data—including text, numerical demographics, and product categories—to build a machine learning pipeline that predicts whether a customer would recommend a product.

## Dataset Summary
The dataset contains 18,442 anonymized reviews with the following features:

    - Textual: Title, Review Text (The primary source of sentiment).
    - Numerical: Age, Positive Feedback Count.
    - Categorical: Clothing ID, Division Name, Department Name, Class Name.
    - Target: Recommended IND (1 = Recommended, 0 = Not Recommended).

## Key Findings & Challenges
    - Class Imbalance: Approximately 81.5% of the data consists of positive recommendations. To address this, the final model was tuned using class_weight='balanced' and evaluated using F1-weighted scoring.

    - Feature Variance: Positive Feedback Count showed high skewness (ranging from 0 to 122), requiring robust scaling within the pipeline.

    - Text Insights: "Not Recommended" reviews tended to be slightly longer, suggesting dissatisfied customers provide more detailed critiques.

## Model Pipeline Architecture
The project utilizes a Scikit-Learn Pipeline combined with a ColumnTransformer to ensure clean, modular code and prevent data leakage:

    1. Numerical Transformer: Applies StandardScaler to Age and Positive Feedback Count.

    2. Categorical Transformer: Applies OneHotEncoder to department and division names.

    3. Text Transformer: Utilizes TfidfVectorizer (NLP) to tokenize and vectorize the review text, removing English stop words.

    4. Classifier: A RandomForestClassifier was fine-tuned to handle high-dimensional text data.

## Performance & Evaluation
Through GridSearchCV, the model was optimized for the minority class.

    - Initial Baseline: High accuracy (86%) but poor recall for negative reviews (0.36).

    - Fine-Tuned Model: Improved Recall for "Not Recommended" items to 0.68, providing StyleSense with a much more reliable tool for identifying customer dissatisfaction.

## Files in Repository
starter.ipynb: The primary Jupyter Notebook containing data exploration, pipeline construction, and model training.

data/reviews.csv: The anonymized dataset used for the project.

requirements.txt: List of necessary Python packages (scikit-learn, pandas, spacy, etc.).