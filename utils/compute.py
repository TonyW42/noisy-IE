from datasets import load_dataset, load_metric
import numpy as np

metric = load_metric("seqeval")
wnut = load_dataset("wnut_17")

label_list = wnut["train"].features[f"ner_tags"].feature.names
# label_list


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels,)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
