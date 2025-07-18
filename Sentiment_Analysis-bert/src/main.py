# Entry point of the application
from data.load_data import load_data
from models.bert_classifier import BertClassifier
from training.train import train_model
from evaluation.evaluate import evaluate_model
from utils.tokenizer import get_tokenizer

def main():
    # Load the dataset
    df = load_data('IMDB Dataset.csv')

    # Prepare the tokenizer
    tokenizer = get_tokenizer()

    # Initialize the BERT classifier
    model = BertClassifier()

    # Split the data into train and test sets
    train_df, test_df = df[:int(0.8*len(df))], df[int(0.8*len(df)):]

    # Train the model
    train_model(model, train_df, tokenizer)

    # Evaluate the model
    report = evaluate_model(model, test_df, tokenizer)
    print(report)

if __name__ == "__main__":
    main()