# Vinum Analytica
## Wine Review Insight

This project is part of the **Data Mining and Machine Learning (DMML)** course at the University of Pisa. The primary objective is to classify grape varieties based on wine descriptions using machine learning techniques.

## Project Overview
The dataset used in this project contains wine reviews, each offering details about the wine, such as its description, country of origin, winery, and more. The core aim is to develop a machine learning model capable of accurately predicting the variety of grapes from these descriptions.

## Dataset
The dataset is sourced from [Wine Enthusiast via Kaggle](https://www.kaggle.com/datasets/zynicide/wine-reviews). It includes a large collection of wine reviews with textual descriptions. Before analysis, the dataset undergoes preprocessing and cleaning to ensure quality. For example, stop words are removed, and features such as term frequency-inverse document frequency (TF-IDF) are extracted from the descriptions.

### Key Attributes:
- **Description**: A detailed review of the wine.
- **Variety**: The target variable representing the grape variety.
- **Additional fields**: Country, winery, region, and other metadata.

## Methodology
The project follows a systematic approach:

1. **Data Preprocessing**: Cleaning and preprocessing the text and metadata to prepare for analysis.
2. **Feature Extraction**: Using techniques such as TF-IDF to convert textual descriptions into numeric features.
3. **Model Training**: Training models including Decision Tree, Random Forest, and Neural Networks to predict the grape variety.
4. **Model Evaluation**: Evaluating model performance using metrics like accuracy and confusion matrices.

### Preprocessing:
- Removal of duplicates.
- Normalization of text (e.g., converting Unicode to ASCII).
- Handling class imbalance with techniques like SMOTE and undersampling.

## Results
The results will be presented through validation and test accuracy scores. Additionally, confusion matrices will be provided for each model, helping visualize performance in classifying the grape varieties.

### Key Results:
- **Neural Network**: Validation accuracy of 0.5894, test accuracy of 0.5996.
- **Random Forest**: Validation accuracy of 0.5889, test accuracy of 0.5925.
- **Decision Tree**: Validation accuracy of 0.3977, test accuracy of 0.3987.

## Usage
### Installation
1. Clone the repository to your local machine.
2. Install dependencies from `requirements.txt`.
3. Install the project with `pip install -e .`

### Workflow
#### Cleaning
1. Run the script in `src/scripts/` for data cleaning.
2. Split the dataset into training and test sets using the 6th notebook.

#### Hyperparameter Tuning
1. Tune the models by executing the tuning scripts:  
   `python tuning/tuning_xx.py`  
   Replace `xx` with the model identifier: `dt` (Decision Tree), `nn` (Neural Network), `rf` (Random Forest).
2. Analyze tuning results in the 7th notebook.

#### Training
1. Train models by running the classification scripts:  
   `python trains/classification_xx.py`  
   Replace `xx` with the model identifier.

#### Testing
1. Use the 8th notebook to compare the best-performing models.

## Conclusion
Successfully classifying grape varieties based on wine descriptions can enhance wine analysis, offering insights for sommeliers, winemakers, and wine enthusiasts. Future work will explore regression analysis on wine prices and fine-tuning the models for better accuracy.