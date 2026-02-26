from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


def create_training_args(use_tuned_config=False):
    if use_tuned_config:
        return TrainingArguments(
            output_dir="./results_tuned",
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=1e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs_tuned",
            logging_steps=10,
            dataloader_pin_memory=False,
            load_best_model_at_end=False,
            report_to="none",
        )

    return TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        dataloader_pin_memory=False,
        load_best_model_at_end=False,
        report_to="none",
    )


def create_trainer(model, tokenizer, train_dataset, test_dataset, use_tuned_config=False):
    training_args = create_training_args(use_tuned_config=use_tuned_config)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )