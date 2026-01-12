import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def compute_confusion_matrix(self, y_true, y_pred):
        """
        Standard Binary Matrix (TP/TN/FP/FN).
        Only used for Binary Classification (Breast Cancer).
        """
        y_true = y_true.flatten().astype(int)
        y_pred = y_pred.flatten().astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        return TP, TN, FP, FN

    # --- THIS IS THE MISSING FUNCTION ---
    def compute_multiclass_matrix(self, y_true_idx, y_pred_idx, num_classes=10):
        """
        Computes a KxK Confusion Matrix for Multi-Class (MNIST).
        Rows: Actual, Cols: Predicted
        """
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true_idx, y_pred_idx):
            matrix[t, p] += 1
        return matrix
    # ------------------------------------

    def evaluate(self, x_test, y_test):
        # 1. Get Predictions
        y_pred = self.pipeline.predict(x_test)
        
        # --- CHECK: Binary or Multi-Class? ---
        is_binary = (y_test.shape[1] == 1) if y_test.ndim > 1 else True

        # 2. Calculate Accuracy
        if is_binary:
            acc = np.mean(y_pred.flatten() == y_test.flatten())
        else:
            true_indices = np.argmax(y_test, axis=1)
            acc = np.mean(y_pred.flatten() == true_indices)

        # 3. Retrieve History
        train_loss = self.pipeline.loss_history[-1] if self.pipeline.loss_history else 0.0
        train_acc  = self.pipeline.accuracy_history[-1] if self.pipeline.accuracy_history else 0.0
        
        if len(self.pipeline.val_loss_history) > 0:
            val_loss_str = f"{self.pipeline.val_loss_history[-1]:.4f}"
            val_acc_str  = f"{self.pipeline.val_accuracy_history[-1]:.2f}%"
        else:
            val_loss_str = "N/A"
            val_acc_str  = "N/A"

        # 4. Print Report
        print("\n" + "="*45)
        print(f"       FINAL PERFORMANCE REPORT ({'Binary' if is_binary else 'Multi-Class'})")
        print("="*45)
        print(f"{'Metric':<10} | {'Train':<10} | {'Val':<10} | {'Test (New)':<10}")
        print("-" * 50)
        print(f"{'Loss':<10} | {train_loss:.4f}     | {val_loss_str:<10} | N/A")
        print(f"{'Accuracy':<10} | {train_acc:.2f}%     | {val_acc_str:<10} | {acc*100:.2f}%")
        
        metrics = {'accuracy': acc, 'f1_score': 0.0}

        if is_binary:
            TP, TN, FP, FN = self.compute_confusion_matrix(y_test, y_pred)
            epsilon = 1e-15
            precision = TP / (TP + FP + epsilon)
            recall = TP / (TP + FN + epsilon)
            f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
            
            print("\n--- 2. CLASSIFICATION METRICS (Test Set) ---")
            print(f"Confusion Matrix:\n [[TN={TN}  FP={FP}]\n  [FN={FN}  TP={TP}]]")
            print(f"{'-'*30}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1_score:.4f}")
            metrics['f1_score'] = f1_score
        else:
            print("\n--- 2. CLASSIFICATION METRICS (Test Set) ---")
            print("(Detailed F1/Precision skipped for Multi-Class output)")
            print(f"Accuracy: {acc*100:.2f}%")

        print("="*45 + "\n")
        return metrics

    def plot_history(self):
        """Plots Training vs Validation Curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1.plot(self.pipeline.loss_history, label='Training Loss', color='blue')
        if len(self.pipeline.val_loss_history) > 0:
            ax1.plot(self.pipeline.val_loss_history, label='Validation Loss', color='orange', linestyle='--')
        ax1.set_title('Model Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(self.pipeline.accuracy_history, label='Training Acc', color='green')
        if len(self.pipeline.val_accuracy_history) > 0:
            ax2.plot(self.pipeline.val_accuracy_history, label='Validation Acc', color='red', linestyle='--')
        ax2.set_title('Model Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.show()

    def plot_confusion_matrix(self, x_test, y_test):
        """Smart Plotting: Handles both 2x2 (Cancer) and 10x10 (MNIST) matrices."""
        y_pred = self.pipeline.predict(x_test)
        
        # --- LOGIC SWITCH ---
        is_binary = (y_test.shape[1] == 1) if y_test.ndim > 1 else True

        if is_binary:
            TP, TN, FP, FN = self.compute_confusion_matrix(y_test, y_pred)
            cm = np.array([[TN, FP], [FN, TP]])
            labels = ['Class 0', 'Class 1']
        else:
            # Multi-Class Logic
            y_true_idx = np.argmax(y_test, axis=1)
            y_pred_idx = y_pred.flatten()
            cm = self.compute_multiclass_matrix(y_true_idx, y_pred_idx, num_classes=10)
            labels = [str(i) for i in range(10)]

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_curve(self, x_test, y_test):
        """Plots ROC only for Binary problems to avoid crashes."""
        is_binary = (y_test.shape[1] == 1) if y_test.ndim > 1 else True
        
        if not is_binary:
            print("   [Visuals] Note: ROC Curve skipped for Multi-Class (MNIST).")
            return

        y_probs = self.pipeline.predict_proba(x_test)
        y_true = y_test.flatten().astype(int)
        y_probs = y_probs.flatten()
        thresholds = np.sort(np.unique(y_probs))[::-1]
        
        tpr_list = []
        fpr_list = []

        for thresh in thresholds:
            y_pred_temp = (y_probs >= thresh).astype(int)
            TP = np.sum((y_true == 1) & (y_pred_temp == 1))
            TN = np.sum((y_true == 0) & (y_pred_temp == 0))
            FP = np.sum((y_true == 0) & (y_pred_temp == 1))
            FN = np.sum((y_true == 1) & (y_pred_temp == 0))
            
            tpr = TP / (TP + FN + 1e-15)
            fpr = FP / (FP + TN + 1e-15)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            
        tpr_list = [0] + tpr_list + [1]
        fpr_list = [0] + fpr_list + [1]
        auc = np.trapz(tpr_list, fpr_list)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()