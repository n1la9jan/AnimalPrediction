import streamlit as st

st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("PyTorch Image Classification App")
st.write("Upload an image to get a prediction for whether it's a cat, dog, or wild animal.")

import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image
import pickle

class Net(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding = 1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding = 1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding = 1)

    self.pooling = nn.MaxPool2d(2,2)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()
    self.linear = nn.Linear((128*16*16), 128)
    self.output = nn.Linear(128, 3)

  def forward(self, x):
    x = self.conv1(x)
    x = self.pooling(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.pooling(x)
    x = self.relu(x)

    x = self.conv3(x)
    x = self.pooling(x)
    x = self.relu(x)

    x = self.flatten(x)
    x = self.linear(x)
    x = self.output(x)

    return x

# Load the trained model
@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load('./trained_model/animal_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Load the label encoder
@st.cache_resource
def load_label_encoder():
    with open('./trained_model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder

label_encoder = load_label_encoder()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
])


def predict_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    return label_encoder.inverse_transform(predicted.cpu().numpy())[0]

# Streamlit UI for image upload and prediction
st.subheader("Upload an Image for Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=200)
    st.write("")
    st.write("Classifying...")

    prediction = predict_image(image)
    st.success(f"Prediction: {prediction.capitalize()}")