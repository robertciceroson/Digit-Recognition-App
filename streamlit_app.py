import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognition App",
    page_icon="🔢",
    layout="centered",
)

# ── CNN architecture (matches the notebook) ───────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ── Train once, cache for the session ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    progress = st.progress(0, text="Training CNN on MNIST — first launch only (~60 s)…")
    total = len(loader)

    model.train()
    for i, (images, labels) in enumerate(loader):
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            progress.progress(min(i / total, 1.0), text=f"Training… batch {i}/{total}")

    progress.empty()
    model.eval()
    return model


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔢 Digit Recognition App")
st.markdown(
    "Draw a digit (0–9) in the box below, then click **Predict** to see the CNN's answer."
)
st.markdown("---")

# ── Load / train model ────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    model = get_model()

# ── Drawing canvas (HTML5 canvas via streamlit-drawable-canvas) ───────────────
try:
    from streamlit_drawable_canvas import st_canvas

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### ✏️ Draw here")
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=18,
            stroke_color="black",
            background_color="white",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col2:
        st.markdown("### 📊 Prediction")

        if st.button("🔍 Predict", use_container_width=True):
            if canvas_result.image_data is not None:
                # Convert RGBA numpy array → PIL grayscale → invert → 28×28
                img_array = canvas_result.image_data.astype(np.uint8)
                img_pil = Image.fromarray(img_array, mode="RGBA").convert("L")
                img_pil = ImageOps.invert(img_pil)          # MNIST: white digit on black
                img_pil = img_pil.resize((28, 28), Image.LANCZOS)

                # Normalise and add batch + channel dims
                img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0)

                with torch.no_grad():
                    logits = model(img_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze().numpy()
                    pred = int(np.argmax(probs))

                st.markdown(f"## Predicted digit: **{pred}**")
                st.markdown(f"Confidence: **{probs[pred]*100:.1f}%**")

                # Probability bar chart
                fig, ax = plt.subplots(figsize=(4, 3))
                colors = ["#2E75B6" if i == pred else "#D9E8F5" for i in range(10)]
                ax.bar(range(10), probs, color=colors)
                ax.set_xticks(range(10))
                ax.set_xlabel("Digit")
                ax.set_ylabel("Probability")
                ax.set_title("Class probabilities")
                ax.set_ylim(0, 1)
                st.pyplot(fig)
            else:
                st.info("Draw a digit first, then click Predict.")
        else:
            st.info("Draw a digit on the left, then click **Predict**.")

except ImportError:
    st.error(
        "`streamlit-drawable-canvas` is not installed. "
        "Add `streamlit-drawable-canvas` to requirements.txt."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "CNN trained on 60,000 MNIST images · ~98–99% test accuracy · "
    "[GitHub repo](https://github.com/robertciceroson/Digit-Recognition-App)"
)
