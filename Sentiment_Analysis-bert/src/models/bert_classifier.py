class BertClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        from transformers import BertForSequenceClassification
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def train(self, train_dataset, eval_dataset, training_args):
        from transformers import Trainer
        from transformers import DataCollatorWithPadding

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        trainer.train()

    def evaluate(self, test_dataset):
        from transformers import Trainer

        trainer = Trainer(model=self.model)
        predictions = trainer.predict(test_dataset)
        return predictions