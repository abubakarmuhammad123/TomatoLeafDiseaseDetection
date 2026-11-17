
import gradio as gr
import sys
from pathlib import Path

# Locate project directories
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

# Make the prediction module accessible
sys.path.insert(0, str(SRC_DIR))

from prediction import predict_tomato


def gradio_predict(image):

    if image is None:
        return "Please upload a tomato leaf image."

    result = predict_tomato(image)

    if result["status"] == "rejected":
        return (
            "Image rejected.\n\n"
            "This image did not pass the tomato-leaf "
            "verification stage.\n"
            f"Non-tomato confidence: {result['confidence']:.2%}"
        )

    disease = result["disease"].replace("Tomato___", "").replace("_", " ")

    return (
        "Tomato leaf detected!\n\n"
        f"Predicted condition: {disease}\n"
        f"Classification confidence: {result['confidence']:.2%}\n\n"
        "Note: AI prediction, not a definitive plant-health diagnosis."
    )


demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Upload a leaf image"),
    outputs=gr.Textbox(label="Prediction result"),
    title="Tomato Leaf Disease Detection System",
    description=(
        "An AI-powered two-stage system that verifies tomato leaves "
        "and predicts five tomato conditions."
    )
)


if __name__ == "__main__":
    demo.launch()
