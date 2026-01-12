import gradio as gr
import numpy as np
import pickle
import sys
import os
import cv2

# Ensure project modules are available
sys.path.append(os.getcwd())

# ---------------------------------------------------------
# MODEL LOADER UTILITY
# ---------------------------------------------------------
def load_saved_model(dataset_name):
    """Loads the pickled pipeline from the saved_models directory"""
    model_path = f"saved_models/{dataset_name}_model.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# ---------------------------------------------------------
# PREDICTION LOGIC
# ---------------------------------------------------------
def predict_handler(dataset_name, input_image, *tabular_inputs):
    # 1. Load the specific model
    pipeline = load_saved_model(dataset_name)
    if pipeline is None:
        return f"❌ Error: Model for {dataset_name} not found in saved_models/"

    try:
        if dataset_name == "mnist":
            if input_image is None: return "Please draw a digit."
            
            # --- MNIST LOGIC ---
            img = input_image['composite'] if isinstance(input_image, dict) else input_image
            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            resized = cv2.resize(gray, (28, 28))
            resized = cv2.bitwise_not(resized) # Fix color inversion
            flattened = resized.reshape(1, 784) / 255.0
            
            probs = pipeline.predict_proba(flattened)
            prediction = np.argmax(probs)
            confidence = probs[0][prediction]
            
            return f"🔢 Predicted Digit: {prediction}\n✅ Confidence: {confidence:.2%}"

        elif dataset_name == "breast_cancer":
            # --- BREAST CANCER LOGIC ---
            # Now we receive exactly 30 inputs from the UI
            features = np.array(tabular_inputs).reshape(1, -1)
            
            # Predict
            prob = pipeline.predict_proba(features)[0][0]
            result = "Malignant (High Risk)" if prob >= 0.5 else "Benign (Low Risk)"
            
            # Color code the result for better UX
            emoji = "🔴" if prob >= 0.5 else "🟢"
            
            return (f"{emoji} DIAGNOSIS: {result}\n"
                    f"📊 Probability of Malignancy: {prob:.4%}")

    except Exception as e:
        import traceback
        return f"❌ Prediction Error: {str(e)}\n{traceback.format_exc()}"

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
# The 30 Feature Names in standard order
feature_names = [
    "Radius", "Texture", "Perimeter", "Area", "Smoothness",
    "Compactness", "Concavity", "Concave Points", "Symmetry", "Fractal Dim"
]

with gr.Blocks(title="Model Inference Center", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 AI Diagnostic Center")
    gr.Markdown("Select a trained model to run real-time inference.")

    dataset_choice = gr.Radio(["mnist", "breast_cancer"], label="Select Model", value="mnist")

    # --- MNIST UI ---
    with gr.Column(visible=True) as mnist_ui:
        gr.Markdown("### ✍️ Digit Recognition")
        with gr.Row():
            with gr.Column():
                sketch = gr.Sketchpad(label="Draw Here", layers=False, type="numpy", brush=gr.Brush(colors=["#000000"], color_mode="fixed"))
                mnist_btn = gr.Button("Classify Digit", variant="primary")
            with gr.Column():
                # Spacer
                pass

    # --- BREAST CANCER UI (30 Features) ---
    cancer_inputs = [] # List to store the 30 slider components
    
    with gr.Column(visible=False) as cancer_ui:
        gr.Markdown("### 🩺 Patient Data Entry")
        cancer_btn = gr.Button("Run Diagnostic Analysis", variant="primary")
        
        with gr.Row():
            # COLUMN 1: Mean Values (Features 0-9)
            with gr.Column():
                gr.Markdown("#### Mean Values")
                for name in feature_names:
                    # We set reasonable defaults to avoid '0.0' which might be unrealistic
                    slider = gr.Slider(0, 1000, value=10, label=f"Mean {name}") 
                    cancer_inputs.append(slider)

            # COLUMN 2: Standard Error (Features 10-19)
            with gr.Column():
                gr.Markdown("#### Standard Error")
                for name in feature_names:
                    slider = gr.Slider(0, 100, value=1, label=f"SE {name}")
                    cancer_inputs.append(slider)

            # COLUMN 3: Worst Values (Features 20-29)
            with gr.Column():
                gr.Markdown("#### Worst (Largest) Values")
                for name in feature_names:
                    slider = gr.Slider(0, 1000, value=20, label=f"Worst {name}")
                    cancer_inputs.append(slider)

    # --- OUTPUT SECTION ---
    gr.Markdown("### 📋 Analysis Results")
    output_text = gr.Textbox(label="Result", lines=2)

    # --- EVENTS ---
    
    # 1. Toggle UI Visibility
    def toggle_ui(choice):
        if choice == "mnist":
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True)

    dataset_choice.change(toggle_ui, inputs=[dataset_choice], outputs=[mnist_ui, cancer_ui])

    # 2. Button Actions
    # Notice we pass the *list* `cancer_inputs` which expands to 30 arguments
    mnist_btn.click(
        predict_handler, 
        inputs=[dataset_choice, sketch] + cancer_inputs, 
        outputs=output_text
    )
    
    cancer_btn.click(
        predict_handler, 
        inputs=[dataset_choice, sketch] + cancer_inputs, 
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()