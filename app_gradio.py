import gradio as gr
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import ast  # Safely evaluate strings like "[128, 64]"

# Add project root to path
sys.path.append(os.getcwd())

# Import your custom modules
from data_pipeline.data_loader import DataLoader
from data_pipeline.preprocessor import Preprocessor
from pipeline.pipeline import NeuralNetworkPipeline
from evaluation.evaluator import Evaluator
from data_pipeline.data_generator import DataGenerator

# --- GLOBAL STATE ---
current_pipeline = None
current_preprocessor = None
current_config = None
test_data_bundle = None # Store (x_test, y_test) for evaluation later

def load_default_config():
    """Loads the base config to populate initial UI values"""
    with open('config/config.yml', 'r') as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------
# LOGIC: TRAINING HANDLER
# ---------------------------------------------------------
def train_model(
    dataset_name, 
    # Model Params
    arch_str, activ_str, dropout_val, 
    # Training Params
    epochs, lr, batch_size, 
    # Optimizer Params
    opt_method, beta1, beta2, epsilon,
    # Regularization Params
    reg_type, lambda_val,
    # Progress Bar
    progress=gr.Progress()
):
    global current_pipeline, current_preprocessor, current_config, test_data_bundle
    
    # 1. Load Base Config
    config = load_default_config()
    dataset_cfg = config['datasets'][dataset_name]
    
    # --- 2. APPLY UI OVERRIDES (The "Control Everything" Logic) ---
    
    # A. Parse Architecture String "[128, 64]" -> List [128, 64]
    try:
        architecture = ast.literal_eval(arch_str)
        if not isinstance(architecture, list): raise ValueError
    except:
        return None, f"Error: Architecture must be a list like [128, 64]. You typed: {arch_str}"

    # B. Parse Activations "[relu, relu, sigmoid]" -> List
    try:
        activations = ast.literal_eval(activ_str)
        if not isinstance(activations, list): raise ValueError
    except:
         return None, f"Error: Activations must be a list like ['relu', 'sigmoid']. You typed: {activ_str}"

    # C. Update Config Dictionary
    dataset_cfg['model']['architecture'] = architecture
    dataset_cfg['model']['activations'] = activations
    dataset_cfg['model']['dropout_rates'] = [float(dropout_val)] * len(architecture) # Apply uniform dropout for UI simplicity
    
    dataset_cfg['training']['epochs'] = int(epochs)
    dataset_cfg['training']['learning_rate'] = float(lr)
    dataset_cfg['training']['batch_size'] = int(batch_size)
    
    dataset_cfg['model']['regularization'] = reg_type
    dataset_cfg['model']['l_lambda'] = float(lambda_val)

    # D. Update Global Optimizer Settings
    config['optimizer']['method'] = opt_method
    config['optimizer']['beta1'] = float(beta1)
    config['optimizer']['beta2'] = float(beta2)
    config['optimizer']['epsilon'] = float(epsilon)

    # 3. Load Data
    progress(0.1, desc="Loading Data...")
    loader = DataLoader(config_path=None, data_dir=config['paths']['raw_data'])
    
    # Wrap config for loader
    loader_friendly_config = dataset_cfg.copy()
    if 'data_config' in dataset_cfg: loader_friendly_config.update(dataset_cfg['data_config'])
    loader.config = {'data_pipeline': {'datasets': {dataset_name: loader_friendly_config}}}
    
    try:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = loader.load_dataset(dataset_name)
    except Exception as e:
        return None, f"Data Load Error: {e}"

    # 4. Preprocess
    progress(0.2, desc="Preprocessing...")
    prep = Preprocessor()
    prep.fit(x_train)
    current_preprocessor = prep 
    
    flatten_needed = (dataset_cfg['type'] == 'image_flattened')
    x_train = prep.transform(x_train, flatten=flatten_needed)
    x_val = prep.transform(x_val, flatten=flatten_needed)
    x_test = prep.transform(x_test, flatten=flatten_needed) # Prep test for later

    # Handle Labels
    if dataset_name == 'breast_cancer':
        mapping = {'M': 1, 'B': 0} # Should match config really, but hardcoded for UI safety
        y_train = np.array([mapping[y] for y in y_train]).reshape(-1, 1)
        y_val = np.array([mapping[y] for y in y_val]).reshape(-1, 1)
        y_test = np.array([mapping[y] for y in y_test]).reshape(-1, 1)
    else:
        y_train = prep.fit_encode_labels(y_train)
        y_val = prep.encode_labels(y_val)
        y_test = prep.encode_labels(y_test)

    # SAVE TEST DATA FOR LATER EVALUATION
    test_data_bundle = (x_test, y_test)

    # 5. Initialize Pipeline
    input_dim = x_train.shape[1]
    full_arch = [input_dim] + architecture
    
    current_pipeline = NeuralNetworkPipeline(
        layer_sizes=full_arch,
        activations=activations,
        loss_type=dataset_cfg['model']['loss_function'],
        optimizer_method=opt_method,
        lr=float(lr),
        beta1=float(beta1),
        beta2=float(beta2),
        epsilon=float(epsilon),
        regularization=reg_type,
        l_lambda=float(lambda_val),
        dropout_rates=dataset_cfg['model']['dropout_rates']
    )
    
    # 6. Training Loop with Live Updates
    train_gen = DataGenerator(x_train, y_train, batch_size=int(batch_size))
    
    history_loss = []
    history_acc = []
    
    t = 0
    for epoch in range(int(epochs)):
        for x_batch, y_batch in train_gen:
            t += 1
            y_pred = current_pipeline.mlp.forward(x_batch, traning=True)
            dl_doutput = current_pipeline.total_backward(y_pred, y_batch)
            gradients = current_pipeline.mlp.backward(dl_doutput)
            current_pipeline.mlp.update_parameters(gradients, t)
            
        # Metrics
        y_pred_train = current_pipeline.mlp.forward(x_train, traning=False)
        train_loss = current_pipeline.compute_total_loss(y_pred_train, y_train)
        train_acc = current_pipeline.accuracy(y_pred_train, y_train)
        
        history_loss.append(train_loss)
        history_acc.append(train_acc)
        
        # Update Progress
        progress((epoch+1)/int(epochs), desc=f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(history_loss, label='Loss', color='blue')
        ax1.set_title("Training Loss")
        ax1.grid(True, alpha=0.3)
        ax2.plot(history_acc, label='Accuracy', color='green')
        ax2.set_title("Training Accuracy")
        ax2.grid(True, alpha=0.3)
        plt.close(fig)
        
        yield fig, f"Epoch {epoch+1}: Accuracy {train_acc:.2f}%"

    return fig, f"Training Complete! Final Accuracy: {train_acc:.2f}%"

# ---------------------------------------------------------
# LOGIC: EVALUATION HANDLER
# ---------------------------------------------------------
# ---------------------------------------------------------
# LOGIC: EVALUATION HANDLER (DEBUG VERSION)
# ---------------------------------------------------------
# ---------------------------------------------------------
# LOGIC: EVALUATION HANDLER (CLEAN VERSION - NO HISTORY)
# ---------------------------------------------------------
def run_full_evaluation():
    global current_pipeline, test_data_bundle
    
    if current_pipeline is None or test_data_bundle is None:
        return None, "⚠️ Train the model first!"
        
    x_test, y_test = test_data_bundle
    evaluator = Evaluator(current_pipeline)
    
    try:
        # 1. Capture Text Report
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            metrics = evaluator.evaluate(x_test, y_test)
        report_text = f.getvalue()

        # 2. Determine Dataset Type
        is_binary = (y_test.shape[1] == 1) if y_test.ndim > 1 else True
        y_pred = current_pipeline.predict(x_test)

        # 3. Create Plots (Single Row only)
        if is_binary:
            # === BREAST CANCER: Confusion Matrix + ROC ===
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Confusion Matrix
            TP, TN, FP, FN = evaluator.compute_confusion_matrix(y_test, y_pred)
            cm = np.array([[TN, FP], [FN, TP]])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                        xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
            ax1.set_title('Confusion Matrix')

            # ROC Curve
            y_probs = current_pipeline.predict_proba(x_test).flatten()
            y_true = y_test.flatten()
            thresholds = np.sort(np.unique(y_probs))[::-1]
            tpr_list, fpr_list = [0], [0]
            for thresh in thresholds:
                temp_pred = (y_probs >= thresh).astype(int)
                tp_ = np.sum((y_true == 1) & (temp_pred == 1))
                tn_ = np.sum((y_true == 0) & (temp_pred == 0))
                fp_ = np.sum((y_true == 0) & (temp_pred == 1))
                fn_ = np.sum((y_true == 1) & (temp_pred == 0))
                tpr_list.append(tp_ / (tp_ + fn_ + 1e-15))
                fpr_list.append(fp_ / (fp_ + tn_ + 1e-15))
            tpr_list.append(1); fpr_list.append(1)
            auc = np.trapz(tpr_list, fpr_list)
            
            ax2.plot(fpr_list, tpr_list, color='darkorange', label=f'AUC = {auc:.3f}')
            ax2.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax2.set_title('ROC Curve')
            ax2.legend()

        else:
            # === MNIST: PER-CLASS ACCURACY BAR CHART ===
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            y_true_idx = np.argmax(y_test, axis=1)
            y_pred_idx = y_pred.flatten()
            classes = np.arange(10)
            accuracies = []
            
            for cls in classes:
                mask = (y_true_idx == cls)
                acc = np.mean(y_pred_idx[mask] == cls) if np.sum(mask) > 0 else 0
                accuracies.append(acc * 100)
            
            colors = ['#2ca02c' if a > 95 else '#ff7f0e' if a > 80 else '#d62728' for a in accuracies]
            bars = ax.bar(classes, accuracies, color=colors, alpha=0.8)
            
            ax.set_title('Performance by Digit (Which number is hardest?)')
            ax.set_xlabel('Digit (0-9)')
            ax.set_ylabel('Accuracy (%)')
            ax.set_xticks(classes)
            ax.set_ylim(0, 105)
            ax.grid(axis='y', alpha=0.3)
            
            # Labels on bars
            for bar, acc in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                         f"{acc:.0f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        return fig, report_text

    except Exception as e:
        import traceback
        return None, f"❌ Evaluation Error: {str(e)}\n{traceback.format_exc()}"

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
css = """
.gradio-container {background-color: #f0f2f6}
h1 {color: #2d3748; text-align: center}
"""

with gr.Blocks(title="Neural Network Master", css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Neural Network Master Control Dashboard")
    
    with gr.Tab("1. Training & Configuration"):
        with gr.Row():
            # --- LEFT COLUMN: CONTROLS ---
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 🛠 Model Configuration")
                dataset_dd = gr.Dropdown(["breast_cancer", "mnist"], label="Dataset", value="mnist")
                
                with gr.Accordion("Architecture & Layers", open=True):
                    # Default for MNIST
                    arch_input = gr.Textbox(value="[128, 64, 10]", label="Layer Sizes (Python List)")
                    activ_input = gr.Textbox(value="['relu', 'relu', 'sigmoid']", label="Activations (Python List)")
                    dropout_sl = gr.Slider(0.0, 0.9, value=0.0, step=0.1, label="Dropout Rate")

                with gr.Accordion("Optimizer & Regularization", open=False):
                    opt_dd = gr.Dropdown(["adam", "sgd", "rmsprop", "momentum"], label="Optimizer", value="adam")
                    lr_sl = gr.Number(value=0.001, label="Learning Rate", precision=4)
                    beta1_sl = gr.Slider(0.0, 1.0, value=0.9, label="Beta 1 (Momentum)")
                    beta2_sl = gr.Slider(0.0, 1.0, value=0.999, label="Beta 2 (RMSProp)")
                    eps_num = gr.Number(value=1e-8, label="Epsilon")
                    
                    reg_dd = gr.Radio(["L2", "L1", "None"], label="Regularization", value="L2")
                    lambda_num = gr.Number(value=0.0001, label="Lambda (Strength)", precision=5)

                with gr.Accordion("Training Loop", open=True):
                    epochs_sl = gr.Slider(1, 200, value=10, step=1, label="Epochs")
                    batch_sl = gr.Slider(16, 256, value=64, step=16, label="Batch Size")
                
                train_btn = gr.Button("🚀 Train Model", variant="primary")

            # --- RIGHT COLUMN: VISUALIZATION ---
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Live Training Metrics")
                plot_output = gr.Plot(label="Training Curves")
                status_output = gr.Textbox(label="Logs", interactive=False)

        # Connect Training Button
        train_btn.click(
            train_model, 
            inputs=[
                dataset_dd, arch_input, activ_input, dropout_sl,
                epochs_sl, lr_sl, batch_sl,
                opt_dd, beta1_sl, beta2_sl, eps_num,
                reg_dd, lambda_num
            ], 
            outputs=[plot_output, status_output]
        )

    with gr.Tab("2. Evaluation & Testing"):
        gr.Markdown("### 📊 Test Set Performance")
        
        with gr.Row():
            eval_btn = gr.Button("Run Full Evaluation on Test Set", variant="primary")
        
        with gr.Row():
            with gr.Column():
                conf_matrix_plot = gr.Plot(label="Confusion Matrix")
            with gr.Column():
                report_output = gr.Textbox(label="Detailed Performance Report", lines=15, interactive=False)
        
        eval_btn.click(
            run_full_evaluation,
            inputs=[],
            outputs=[conf_matrix_plot, report_output]
        )

if __name__ == "__main__":
    demo.queue().launch()
