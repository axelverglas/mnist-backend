import sys
import os
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from model.convnet import ConvNet

# Ajouter le parent au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Initialize FastAPI
app = FastAPI(title="MNIST Digit Recognition API", version="1.0.0")

# Configure CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the saved model"""
    global MODEL
    if MODEL is None:
        try:
            # Create an instance of the model
            MODEL = ConvNet(input_size=(1, 28, 28), n_kernels=6, output_size=10)

            # Load the saved weights
            model_path = os.path.join(
                os.path.dirname(__file__), "../../model/mnist-0.0.1.pt"
            )
            MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
            MODEL.to(DEVICE)
            MODEL.eval()

            print(f"Model loaded successfully on {DEVICE}")
        except Exception as e:
            print(f"Error loading the model: {e}")
            raise e

    return MODEL


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess the image for prediction
    - Convert to grayscale
    - Resize to 28x28
    - Normalize between 0 and 1
    - Convert to np.array float32
    """
    try:
        # Open the image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to grayscale
        image = image.convert("L")
        # Resize to 28x28
        image = image.resize((28, 28))
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        # Normalize between 0 and 1
        image_array = image_array / 255.0
        # Apply the same normalization as during training
        image_array = (image_array - 0.1307) / 0.3081
        return image_array

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error during preprocessing: {str(e)}"
        )


def predict(image_array: np.ndarray, model: torch.nn.Module) -> dict:
    """
    Do the prediction on the image
    """
    try:
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(image_array).float()
        # Add batch and channel dimensions: (1, 1, 28, 28)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        # Send to the device
        tensor = tensor.to(DEVICE)
        # Prediction
        with torch.no_grad():
            logits = model(tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        # Return the result as a dictionary
        result = {
            "predicted_digit": int(predicted_class),
            "confidence": float(confidence),
            "probabilities": probabilities[0].cpu().numpy().tolist(),
        }
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@app.get("/")
async def root():
    """Base endpoint to check if the API is running"""
    return {"message": "MNIST Digit Recognition API is running!"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "device": str(DEVICE)}


@app.post("/api/v1/predict")
async def predict_digit(file: UploadFile = File(...)):
    """
    Main endpoint for digit recognition
    """
    # Check the file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The file must be an image")

    try:
        # Read the file content
        image_bytes = await file.read()
        # Preprocess the image
        image_array = preprocess_image(image_bytes)
        # Load t
        model = load_model()
        # Do the prediction
        result = predict(image_array, model)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
