from transformers import BertForSequenceClassification


def get_model(model_name="bert-base-uncased", num_labels=2):
    return BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)