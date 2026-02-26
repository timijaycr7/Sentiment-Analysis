def get_tokenizer():
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer


def tokenize_datasets(train_dataset, test_dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["review"], truncation=True, padding="max_length", max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    return train_dataset, test_dataset