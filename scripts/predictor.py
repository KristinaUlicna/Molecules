import numpy as np

from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

def compute_metrics_and_matrix_classification(data_loader, pre_trained_model):
    y_pred_results = []
    y_true_results = []

    for batch in data_loader:
    
        # Passing the node features and the connection info
        predictions, _ = pre_trained_model(batch.x, batch.edge_index, batch.batch)
        y_pred = predictions.argmax(dim=1)
        y_true = batch.y.squeeze()
        
        # Store the result:
        y_pred_results.extend(y_pred)
        y_true_results.extend(y_true)

    y_pred_results = np.stack(y_pred_results)
    y_true_results = np.stack(y_true_results)

    accuracy = accuracy_score(y_true=y_true_results, y_pred=y_pred_results)
    precision, recall, f1score, _ = precision_recall_fscore_support(
        y_true=y_true_results, y_pred=y_pred_results, pos_label=1, 
    )
    metrics_dictionary = {
        "accuracy" : accuracy, 
        "precision" : precision[1], 
        "recall" : recall[1], 
        "f1score" : f1score[1],
}
    conf_matrix = ConfusionMatrixDisplay.from_predictions(
        y_true=y_true_results, 
        y_pred=y_pred_results,
        normalize="true",
    )

    auroc_curve = RocCurveDisplay.from_predictions(
        y_true=y_true_results, 
        y_pred=y_pred_results,
    )

    avg_pred_curve = PrecisionRecallDisplay.from_predictions(
        y_true=y_true_results,
        y_pred=y_pred_results,
    )
    
    return metrics_dictionary, conf_matrix, auroc_curve, avg_pred_curve


def compute_metrics_and_matrix_regression(data_loader, pre_trained_model, cutoff_limit: float = 8.0):
    y_pred_results = []
    y_true_results = []

    for batch in data_loader:
    
        # Passing the node features and the connection info
        predictions, _ = pre_trained_model(batch.x, batch.edge_index, batch.batch)
        y_pred = predictions.squeeze()
        y_true = batch.y.squeeze()
        
        # Process into integer predictions:
        y_pred = [value > cutoff_limit for value in y_pred]
        y_true = [value > cutoff_limit for value in y_true]
        
        # Store the result:
        y_pred_results.extend(y_pred)
        y_true_results.extend(y_true)

    y_pred_results = np.stack(y_pred_results)
    y_true_results = np.stack(y_true_results)

    accuracy = accuracy_score(y_true=y_true_results, y_pred=y_pred_results)
    precision, recall, f1score, _ = precision_recall_fscore_support(
        y_true=y_true_results, y_pred=y_pred_results, pos_label=1, 
    )

    metrics_dictionary = {
        "accuracy" : accuracy, 
        "precision" : precision, 
        "recall" : recall, 
        "f1score" : f1score,
}
    conf_matrix = ConfusionMatrixDisplay.from_predictions(
        y_true=y_true_results, 
        y_pred=y_pred_results,
        normalize="true",
        cmap="copper"
    )
    
    return metrics_dictionary, conf_matrix
