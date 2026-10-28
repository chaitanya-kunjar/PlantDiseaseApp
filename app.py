import html
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.space()
st.set_page_config(
    page_title="PlantCare | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

IMG_SIZE = (160, 160)
MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"


# --------------------------------------------------
# 2. CUSTOM CSS — FOREST GREEN DESIGN
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --forest-950: #0B2619;
        --forest-900: #123B27;
        --forest-800: #1D5635;
        --forest-700: #287545;
        --leaf: #B9E986;
        --leaf-soft: #E6F5D8;
        --cream: #F2F7EC;
        --surface: #FFFFFF;
        --text: #14291C;
        --muted: #405849;
        --border: #C6DCC5;
    }

    .stApp {
        background: linear-gradient(180deg, #EAF4E3 0%, #F5F8F1 340px, #F5F8F1 100%);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stMain"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 1.8rem;
    }

    h1, h2, h3, h4, p, li, label, [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.4px;
    }

    p, li, .stMarkdown, .stCaption, [data-testid="stCaptionContainer"] {
        color: #263F30;
    }

    /* Full-width, high-contrast top navigation */
    div[data-testid="stRadio"] {
        width: 100%;
        margin: 0 0 28px 0;
    }

    div[data-testid="stRadio"] > label {
        display: none !important;
    }

    div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: row !important;
        gap: 0 !important;
        width: 100% !important;
        box-sizing: border-box;
        background: #D9EBCF;
        border: 2px solid #8BB58A;
        border-radius: 16px;
        padding: 7px;
        margin: 0;
        box-shadow: 0 6px 16px rgba(11, 38, 25, 0.10);
    }

    div[data-testid="stRadio"] > div > label {
        flex: 1 1 0 !important;
        display: flex !important;
        align-items: center;
        justify-content: center;
        text-align: center;
        border-radius: 11px;
        padding: 15px 12px;
        font-size: 1.18rem !important;
        line-height: 1.3;
        font-weight: 800 !important;
        color: #143A25 !important;
        background: transparent;
        border: 1px solid transparent;
        min-height: 54px;
        cursor: pointer;
        transition: background .18s ease, color .18s ease;
    }

    /* Hide radio bullets/icons; the whole tab acts as the navigation control. */
    div[data-testid="stRadio"] > div > label > div:first-child,
    div[data-testid="stRadio"] > div > label svg,
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }

    div[data-testid="stRadio"] > div > label:has(input:checked) {
        background: #164B2D !important;
        color: #FFFFFF !important;
        border-color: #164B2D !important;
        box-shadow: 0 3px 9px rgba(11, 38, 25, 0.22);
    }

    div[data-testid="stRadio"] > div > label:has(input:checked) *,
    div[data-testid="stRadio"] > div > label:has(input:checked) p,
    div[data-testid="stRadio"] > div > label:has(input:checked) span {
        color: #FFFFFF !important;
    }

    .brand-title {
        color: #103B25 !important;
    }

    .hero {
        background: linear-gradient(125deg, #0B2F1D 0%, #155331 58%, #287545 100%);
        border: 1px solid #287545;
        border-radius: 23px;
        padding: 36px 38px;
        margin: 4px 0 26px 0;
        box-shadow: 0 12px 30px rgba(11, 38, 25, 0.18);
    }

    .eyebrow {
        color: #D6F3B9 !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #FFFFFF !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: clamp(1.9rem, 3vw, 2.8rem);
        line-height: 1.15;
        margin: 0 0 14px 0;
    }

    .hero-text {
        color: #F0F8EC !important;
        font-size: 1.04rem;
        line-height: 1.75;
        max-width: 740px;
        margin-bottom: 0;
    }

    .section-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.24rem;
        font-weight: 800;
        color: #153B25 !important;
        margin: 18px 0 5px 0;
    }

    .section-subtitle {
        color: #405849 !important;
        font-size: 0.96rem;
        margin-bottom: 18px;
    }

    .panel {
        background: #FFFFFF;
        color: #183624;
        border: 1px solid #C5DCC3;
        border-radius: 18px;
        padding: 23px;
        margin-bottom: 16px;
        box-shadow: 0 5px 16px rgba(18, 59, 39, 0.07);
    }

    .panel h3 {
        color: #17482D !important;
    }

    .panel p {
        color: #304A39 !important;
    }

    .result-panel {
        background: linear-gradient(145deg, #FFFFFF 0%, #E6F4DE 100%);
        border: 1px solid #B8D6B3;
        border-radius: 19px;
        padding: 25px;
        margin: 12px 0 18px 0;
        box-shadow: 0 7px 22px rgba(18, 59, 39, 0.10);
    }

    .result-label {
        color: #405849 !important;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .result-name {
        color: #17482D !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        line-height: 1.4;
        overflow-wrap: anywhere;
        margin-bottom: 12px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #C6DCC5;
        border-radius: 15px;
        padding: 18px;
        min-height: 105px;
        margin-bottom: 12px;
    }

    .metric-label {
        color: #405849 !important;
        font-size: 0.88rem;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #17482D !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1.65rem;
        overflow-wrap: anywhere;
    }

    .tip {
        background: #E1F1D7;
        border: 1px solid #B8D6AD;
        border-left: 5px solid #287545;
        border-radius: 9px;
        padding: 14px 16px;
        color: #1B4028 !important;
        font-size: 0.94rem;
        line-height: 1.65;
        margin: 14px 0;
    }

    .tip strong, .tip p {
        color: #1B4028 !important;
    }

    div.stButton > button[kind="primary"] {
        background: #1D5635 !important;
        color: #FFFFFF !important;
        border: 1px solid #1D5635 !important;
        border-radius: 11px;
        min-height: 48px;
        font-weight: 800;
        width: 100%;
        transition: 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #0B2619 !important;
        border-color: #0B2619 !important;
        color: #FFFFFF !important;
    }

    div.stButton > button:not([kind="primary"]) {
        color: #17482D !important;
        background: #FFFFFF !important;
        border: 1px solid #91B992 !important;
        border-radius: 10px;
        font-weight: 700;
    }

    div.stButton > button:not([kind="primary"]):hover {
        color: #0B2619 !important;
        background: #E6F4DE !important;
        border-color: #287545 !important;
    }

    [data-testid="stFileUploader"] {
        background: #fff;
        color: #183624;
        border: 2px dashed #6F9D70;
        border-radius: 15px;
        padding: 12px;
    }

    [data-testid="stFileUploader"] * {
        color: #183624;
    }

    [data-testid="stProgress"] > div > div > div > div {
        background-color: #287545;
    }

    [data-testid="stAlert"] {
        color: #173B2A;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
            padding-top: 1rem;
        }
        div[data-testid="stRadio"] > div {
            gap: 4px !important;
            padding: 5px;
        }
        div[data-testid="stRadio"] > div > label {
            font-size: .91rem !important;
            padding: 11px 4px;
            min-height: 46px;
        }
        .hero { padding: 25px 21px; }
        .panel, .result-panel { padding: 18px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. LOAD MODEL AND CLASS NAMES
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    # Supports a JSON list or a dictionary of class names.
    # For dictionaries, insertion order must match the model output order.
    if isinstance(names, dict):
        names = list(names.values())

    if not isinstance(names, list) or not names:
        raise ValueError(
            "class_names.json must contain a non-empty list of class names."
        )

    return [str(name) for name in names]


def prepare_image(image: Image.Image) -> np.ndarray:
    """Resize an image to the model's expected 160x160 RGB input."""
    resized = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def get_display_scores(raw_predictions: np.ndarray) -> np.ndarray:
    """
    Use model probabilities when they are already a valid distribution.
    Otherwise apply softmax to create scores for display. Displayed scores
    are not guaranteed to be calibrated probabilities.
    """
    scores = np.asarray(raw_predictions, dtype=np.float64).reshape(-1)

    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("The model returned empty or invalid prediction scores.")

    looks_like_probabilities = (
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and np.isclose(scores.sum(), 1.0, atol=1e-3)
    )

    if looks_like_probabilities:
        return scores

    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    total = exp_scores.sum()

    if not np.isfinite(total) or total == 0:
        raise ValueError("Could not convert model outputs into display scores.")

    return exp_scores / total


# --------------------------------------------------
# 4. BRAND HEADER AND TOP NAVIGATION
# --------------------------------------------------
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:14px;margin:0 0 18px 0;">
        <div style="font-size:2.35rem;line-height:1;">🌿</div>
        <div>
            <div style="font-family:'Manrope',sans-serif;font-size:2rem;line-height:1.15;"
                 class="brand-title"><strong>PlantCare</strong></div>
            <div style="color:#405849;font-size:1.02rem;font-weight:600;margin-top:4px;">
                Computer Vision Based Solution
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# The three navigation options are equal-width and displayed across the top.
page = st.radio(
    "Navigation",
    ["Home", "Disease Detection", "About the model"],
    horizontal=True,
    label_visibility="collapsed",
    key="navigation_page",
)


def go_to_detection():
    st.session_state["navigation_page"] = "Disease Detection"


def go_to_about():
    st.session_state["navigation_page"] = "About the model"


# --------------------------------------------------
# 5. HOME PAGE
# --------------------------------------------------
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">SMART AGRICULTURE · DEEP LEARNING</div>
            <div class="hero-title">
                Grow healthier.<br>Detect earlier.
            </div>
            <p class="hero-text">
                Welcome to PlantCare — a plant disease prediction demo
                that analyzes a leaf image with a trained MobileNetV2 model.
                Explore predicted classes and model scores in a clear,
                easy-to-use interface.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Your plant-health workspace</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Choose a section below or use the navigation bar at the top.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size:2.1rem;">🍃</div>
                <h3 style="margin:10px 0 7px;">Disease Detection</h3>
                <p style="color:#66786B;line-height:1.7;">
                    Upload a plant leaf photo, view the predicted class,
                    review its score, and compare the top three predictions.
                </p>
                <div style="color:#2D6A45;font-weight:700;font-size:.86rem;">
                    IMAGE ANALYSIS →
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Open Disease Detection →",
            key="home_open_detection",
            use_container_width=True,
            on_click=go_to_detection,
        )

    with col2:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size:2.1rem;">🧠</div>
                <h3 style="margin:10px 0 7px;">About the Model</h3>
                <p style="color:#66786B;line-height:1.7;">
                    Learn how the MobileNetV2 model works, how images are
                    prepared, and what the evaluation score means.
                </p>
                <div style="color:#2D6A45;font-weight:700;font-size:.86rem;">
                    MODEL OVERVIEW →
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Explore the Model →",
            key="home_open_about",
            use_container_width=True,
            on_click=go_to_about,
        )

    st.markdown(
        """
        <div class="tip">
            <strong>Tips for a useful prediction:</strong> Upload a sharp,
            well-lit image with the leaf clearly visible. Results may be less
            reliable for photos that differ from the training dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 6. DISEASE DETECTION PAGE
# --------------------------------------------------
elif page == "Disease Detection":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">LEAF IMAGE ANALYSIS</div>
            <div class="hero-title">Plant Disease Detection</div>
            <p class="hero-text">
                Upload a leaf image to see the model's highest-scoring class
                and the top three predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            '<div class="section-title">01 · Upload your image</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">Supported formats: JPG, JPEG, PNG</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            help="Choose a clear image showing a plant leaf.",
            key="leaf_image_uploader",
        )

        image = None
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(
                    image,
                    caption="Your uploaded leaf image",
                    use_container_width=True,
                )
                st.caption(f"Image dimensions: {image.width} × {image.height}")
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("This file could not be read as an image.")

        if image is None:
            st.markdown(
                """
                <div class="panel" style="text-align:center;padding:28px 18px;">
                    <div style="font-size:3rem;">🌿</div>
                    <h3 style="margin:12px 0 6px;">Your leaf image goes here</h3>
                    <p style="color:#66786B;">
                        Select an image above to preview it before prediction.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        predict_clicked = st.button(
            "Analyze Leaf Image  →",
            type="primary",
            disabled=image is None,
            key="predict_button",
        )

    with right:
        st.markdown(
            '<div class="section-title">02 · Prediction results</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">Your model output will appear here.</div>',
            unsafe_allow_html=True,
        )

        if predict_clicked and image is not None:
            try:
                with st.spinner("Loading model and analyzing leaf..."):
                    model = load_model()
                    class_names = load_class_names()

                    # Preserve the trained model's 160x160 input size.
                    image_array = prepare_image(image)

                    # The model is assumed to include MobileNetV2 preprocessing,
                    # as in the original trained architecture. Do not preprocess twice.
                    raw_predictions = model.predict(image_array, verbose=0)[0]
                    predictions = get_display_scores(raw_predictions)

                    if len(predictions) != len(class_names):
                        raise ValueError(
                            "The number of model outputs does not match the "
                            "number of class names in class_names.json. Confirm "
                            "the class names and order match the training setup."
                        )

                    predicted_index = int(np.argmax(predictions))
                    predicted_class = class_names[predicted_index]
                    confidence = float(predictions[predicted_index])
                    top_indices = np.argsort(predictions)[::-1][:3]

                safe_class = html.escape(predicted_class)

                st.markdown(
                    f"""
                    <div class="result-panel">
                        <div class="result-label">Predicted class</div>
                        <div class="result-name">{safe_class}</div>
                        <div style="color:#66786B;font-size:.9rem;">
                            Highest-scoring class from the model
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Model score</div>
                            <div class="metric-value">{confidence * 100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with metric_col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Recognized classes</div>
                            <div class="metric-value">{len(class_names)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    """
                    <div class="tip">
                        <strong>Please note:</strong> This score represents
                        the model's output, not a guarantee of correctness.
                        It is not necessarily a calibrated probability of disease.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("### Top 3 predictions")

                for rank, index in enumerate(top_indices, start=1):
                    index = int(index)
                    label = html.escape(class_names[index])
                    score = float(predictions[index])

                    st.markdown(
                        f"""
                        <div style="display:flex;justify-content:space-between;
                                    align-items:flex-start;gap:12px;
                                    margin-top:14px;margin-bottom:6px;">
                            <span style="font-weight:600;color:#1C3024;">
                                {rank}. {label}
                            </span>
                            <span style="font-weight:800;color:#205137;white-space:nowrap;">
                                {score * 100:.2f}%
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(score, 0.0), 1.0))

                st.caption(
                    "The model can only predict classes it learned during training. "
                    "For important crop-health decisions, seek expert assessment."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")
                st.info(
                    "Check that plant_disease_model.keras and class_names.json "
                    "are in the repository and match the trained model."
                )
        else:
            st.markdown(
                """
                <div class="panel" style="padding:35px 20px;">
                    <div style="font-size:2rem;">📊</div>
                    <h3 style="margin:12px 0 6px;">Awaiting analysis</h3>
                    <p style="color:#66786B;line-height:1.7;">
                        Upload a leaf image and select
                        <strong>Analyze Leaf Image</strong> to see the result.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------
# 7. ABOUT THE MODEL PAGE
# --------------------------------------------------
elif page == "About the model":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">ABOUT THIS PROJECT</div>
            <div class="hero-title">The science behind the leaf scan</div>
            <p class="hero-text">
                PlantCare uses transfer learning with MobileNetV2 to classify
                plant leaf images into the categories represented in its training data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Model overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Architecture</div>
                <div style="font-size:1.1rem;font-weight:800;color:#205137;">
                    MobileNetV2
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Image input</div>
                <div style="font-size:1.1rem;font-weight:800;color:#205137;">
                    160 × 160
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Test accuracy</div>
                <div style="font-size:1.1rem;font-weight:800;color:#205137;">
                    87.31%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### How the prediction works")
    st.markdown("""
        1. **Image upload:** A user submits a plant leaf photo.
        2. **Preprocessing:** The image is converted to RGB and resized to
           160 × 160 pixels.
        3. **Inference:** The trained model calculates scores for its learned
           output classes.
        4. **Results:** The app displays the highest-scoring class and the
           three highest-scoring predictions.
        """)

    st.markdown("### Model limitations")
    st.markdown("""
        - The model can only predict the classes represented in its training.
        - Predictions may be less reliable for unfamiliar plant varieties,
          backgrounds, lighting conditions, or real-world photos unlike the training images.
        - The displayed score is not a calibrated measure of diagnostic certainty.
        - This educational demo should not replace expert agricultural advice.
        """)

    st.markdown("### Technologies used")
    st.markdown("""
        - **Python** — application programming
        - **TensorFlow / Keras** — model loading and inference
        - **MobileNetV2** — transfer-learning architecture
        - **Streamlit** — interactive web interface
        - **NumPy and Pillow** — image processing
        """)
