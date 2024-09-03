# Brainwave_Matrix_Intern
## FAKE NEWS DETECTION

### Overview
  This project focuses on detecting fake news using machine learning techniques. The model was trained on a dataset containing news articles with labels indicating whether the news is real or fake. The model achieved a 97% accuracy score, making it a highly effective tool for identifying misinformation.

### Project Structure
  * data/: Contains the dataset used for training and testing the model.
  * notebooks/: Jupyter notebooks used for data analysis, feature engineering, and model training.
  * src/: Python scripts for preprocessing, model training, and evaluation.
  * models/: Serialized models and related files.
  * README.md: Project documentation.

### Dataset
  The dataset used for this project includes the following columns:
    URLs: The URL of the news article.
    Headline: The headline of the news article.
    Body: The full text of the news article.
    Label: Indicates whether the news is real or fake.

### Methodology
  1. Data Preprocessing: The data was cleaned and preprocessed to remove noise and irrelevant information.
  2. Feature Engineering: Key features such as word embeddings, TF-IDF scores, and sentiment analysis were extracted.
  3. Model Training: Various machine learning models were trained, including logistic regression, decision trees, and neural networks. A Convolutional Neural Network (CNN) was also implemented to capture complex patterns in the text data.
  4. Evaluation: The models were evaluated using accuracy, precision, recall, and F1-score. The best model achieved a 97% accuracy score.

### Installation
  To run this project locally, follow these steps:

  Clone the repository:
  ```
    git clone https://github.com/Harish2404lll/Brainwave_Matrix_Intern.git
```

  Install the required dependencies:

  ```
     !pip install tensorflow
```

### Output
  ![image](https://github.com/user-attachments/assets/74b4a187-45a4-4185-bfa7-2e66ca222b9f)

### Results
  The model achieved a 97% accuracy score on the test dataset. The confusion matrix and other evaluation metrics can be found in the notebooks/evaluation.ipynb.

### Contributing
  Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.
