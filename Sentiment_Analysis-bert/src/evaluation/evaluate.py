import numpy as np
from sklearn.metrics import classification_report


def evaluate_model(trainer, test_dataset):
    metrics = trainer.evaluate()
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = test_dataset["label"]
    report = classification_report(true_labels, predicted_labels)
    return metrics, report