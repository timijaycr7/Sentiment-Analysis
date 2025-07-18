# Sentiment Analysis with BERT

This project implements a text classification model using BERT (Bidirectional Encoder Representations from Transformers) to classify movie reviews from the IMDB dataset. The project is structured into several modules for better organization and maintainability.

## Project Structure

```
sentiment_analysis-bert
├── src
│   ├── data
│   │   ├── __init__.py
│   │   └── load_data.py
│   ├── models
│   │   ├── __init__.py
│   │   └── bert_classifier.py
│   ├── training
│   │   ├── __init__.py
│   │   └── train.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   └── evaluate.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── tokenizer.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sentiment_analysis-bert
   ```

2. **Install the required libraries**:
   It is recommended to create a virtual environment before installing the dependencies. You can use `venv` or `conda` for this purpose.

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Load the data**:
   The `load_data` function in `src/data/load_data.py` loads the IMDB dataset from a specified path.

2. **Train the model**:
   The `train_model` function in `src/training/train.py` sets up the training arguments and trains the BERT model using the loaded datasets.

3. **Evaluate the model**:
   The `evaluate_model` function in `src/evaluation/evaluate.py` evaluates the trained model on the test dataset and returns a classification report.

4. **Run the application**:
   The entry point of the application is `src/main.py`, which orchestrates the loading of data, model training, and evaluation.

## Example

To run the entire pipeline, execute the following command:

```bash
python src/main.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.