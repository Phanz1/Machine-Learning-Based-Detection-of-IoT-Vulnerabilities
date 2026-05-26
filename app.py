# =========================================================
# IoT Vulnerability Assessment System
# Streamlit + PyTorch Deployment
# =========================================================

# ==========================
# IMPORT LIBRARIES
# ==========================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# ==========================
# PAGE CONFIGURATION
# ==========================

st.set_page_config(
    page_title="IoT Vulnerability Assessment",
    layout="centered"
)

st.title("IoT Vulnerability Assessment System")

st.write(
    "Machine Learning-Based Detection of IoT Vulnerabilities"
)

# =========================================================
# DEFINE AUTOENCODER MODEL
# =========================================================

class Autoencoder(nn.Module):

    def __init__(self, input_dim):

        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

# Number of input features
input_dim = 44

model = Autoencoder(input_dim)

model.load_state_dict(
    torch.load("iot_autoencoder.pth")
)

model.eval()

# =========================================================
# USER INPUT SECTION
# =========================================================

st.subheader("Enter IoT Traffic Features")

packet_size = st.number_input(
    "Packet Size",
    min_value=0.0
)

flow_duration = st.number_input(
    "Flow Duration",
    min_value=0.0
)

connection_frequency = st.number_input(
    "Connection Frequency",
    min_value=0.0
)

# =========================================================
# PREDICTION
# =========================================================

if st.button("Assess Vulnerability"):

    # Convert input to tensor
    input_data = np.array([
        packet_size,
        flow_duration,
        connection_frequency
    ])

    input_tensor = torch.tensor(
        input_data,
        dtype=torch.float32
    )

    input_tensor = input_tensor.unsqueeze(0)

    # Model prediction
    with torch.no_grad():

        reconstruction = model(input_tensor)

        mse = torch.mean(
            (input_tensor - reconstruction) ** 2
        )

    # Threshold
    threshold = 0.1

    st.subheader("Assessment Result")

    if mse.item() > threshold:

        st.error(
            "Vulnerability Detected"
        )

        st.write(
            f"Anomaly Score: {mse.item():.4f}"
        )

    else:

        st.success(
            "Normal IoT Traffic"
        )

        st.write(
            f"Anomaly Score: {mse.item():.4f}"
        )

# =========================================================
# FOOTER
# =========================================================

st.write("---")

st.write(
    "PGD Research Project: Vulnerability Assessment "
    "of IoT Devices Using Machine Learning"
)