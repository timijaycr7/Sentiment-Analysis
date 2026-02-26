import argparse
from pathlib import Path

from data.load_data import create_hf_datasets, load_data
from evaluation.evaluate import evaluate_model
from models.bert_classifier import get_model
from training.train import create_trainer
from utils.tokenizer import get_tokenizer, tokenize_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a BERT sentiment classifier on IMDB data.")
    parser.add_argument(
        "--data-path",
        default=str(Path(__file__).resolve().parent.parent / "IMDB Dataset.csv"),
        help="Path to IMDB CSV dataset",
    )
    parser.add_argument(
        "--tuned",
        action="store_true",
        help="Use the tuned training configuration (lower LR + early stopping)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on number of training samples",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on number of test samples",
    )
    parser.add_argument(
        "--save-dir",
        default=str(Path(__file__).resolve().parent.parent / "saved_model"),
        help="Directory to save the trained model and tokenizer",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_data(args.data_path)
    train_dataset, test_dataset = create_hf_datasets(df)

    tokenizer = get_tokenizer()
    train_dataset, test_dataset = tokenize_datasets(train_dataset, test_dataset, tokenizer)

    if args.max_train_samples is not None:
        train_limit = min(args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(train_limit))
    if args.max_test_samples is not None:
        test_limit = min(args.max_test_samples, len(test_dataset))
        test_dataset = test_dataset.select(range(test_limit))

    model = get_model()
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        use_tuned_config=args.tuned,
    )

    trainer.train()

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    metrics, report = evaluate_model(trainer, test_dataset)

    print("Evaluation metrics:")
    print(metrics)
    print("\nClassification report:")
    print(report)
    print(f"\nSaved model and tokenizer to: {save_path}")


if __name__ == "__main__":
    main()