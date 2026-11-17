
from pathlib import Path

import torch
from torchvision import transforms

from models import TomatoCNN, TomatoValidatorCNN


# Locate project folders
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load saved model checkpoints
disease_checkpoint = torch.load(
    MODEL_DIR / "tomato_cnn_final.pth",
    map_location=device,
    weights_only=True
)

validator_checkpoint = torch.load(
    MODEL_DIR / "tomato_validator.pth",
    map_location=device,
    weights_only=True
)


# Rebuild and load the disease model
disease_model = TomatoCNN().to(device)

disease_model.load_state_dict(
    disease_checkpoint["model_state_dict"]
)

disease_model.eval()


# Rebuild and load the validator model
validator_model = TomatoValidatorCNN().to(device)

validator_model.load_state_dict(
    validator_checkpoint["model_state_dict"]
)

validator_model.eval()


# Image preprocessing
disease_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=disease_checkpoint["mean"],
        std=disease_checkpoint["std"]
    )
])

validator_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=validator_checkpoint["mean"],
        std=validator_checkpoint["std"]
    )
])


# Complete two-stage prediction pipeline
def predict_tomato(image):

    image = image.convert("RGB")

    # Stage 1: Verify tomato leaf
    validator_input = validator_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        validator_output = validator_model(validator_input)
        validator_probs = torch.softmax(validator_output, dim=1)

        tomato_probability = validator_probs[0, 1].item()

    if tomato_probability < 0.50:
        return {
            "status": "rejected",
            "message": "This image did not pass the tomato-leaf verification stage.",
            "confidence": 1 - tomato_probability
        }

    # Stage 2: Predict tomato condition
    disease_input = disease_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        disease_output = disease_model(disease_input)
        disease_probs = torch.softmax(disease_output, dim=1)

        disease_prediction = disease_probs.argmax(dim=1).item()
        disease_confidence = disease_probs[0, disease_prediction].item()

    disease_name = disease_checkpoint["class_names"][disease_prediction]

    return {
        "status": "accepted",
        "disease": disease_name,
        "confidence": disease_confidence
    }
