def evaluate_model(trainer, test_dataset):
    from sklearn.metrics import classification_report
    import numpy as np

    # Get predictions from the trained model on the test set
    predictions = trainer.predict(test_dataset)

    # The predictions are in the 'predictions' attribute, taking the max for classification
    predicted_labels = np.argmax(predictions.predictions, axis=1)

    # Get the true labels from the test dataset
    true_labels = test_dataset['label']

    # Return the classification report
    return classification_report(true_labels, predicted_labels)