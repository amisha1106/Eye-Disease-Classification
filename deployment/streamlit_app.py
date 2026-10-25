import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "recovered_eye_model.keras"

CLASS_LABELS = [
    "Cataract",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Normal"
]

EYE_CARE_GUIDANCE = {
    "Cataract": {
        "overview": "The model's highest-scoring category is Cataract.",
        "tips": [
            "Arrange an eye examination with an optometrist or ophthalmologist.",
            "Use adequate lighting when reading and avoid driving if your vision is impaired.",
            "Wear UV-protective sunglasses outdoors.",
            "If cataracts are confirmed and affect daily activities, discuss treatment options with an eye specialist."
        ]
    },
    "Diabetic Retinopathy": {
        "overview": "The model's highest-scoring category is Diabetic Retinopathy.",
        "tips": [
            "Arrange a comprehensive dilated retinal examination, particularly if you have diabetes.",
            "Follow your clinician's advice on blood glucose, blood pressure, and cholesterol management.",
            "Keep scheduled diabetic eye-screening appointments, even if your vision seems normal.",
            "Seek urgent eye care for sudden vision changes, new floaters, flashes, or a curtain-like shadow."
        ]
    },
    "Glaucoma": {
        "overview": "The model's highest-scoring category is Glaucoma.",
        "tips": [
            "Arrange an eye examination that includes eye-pressure measurement and optic-nerve assessment.",
            "Ask an eye specialist whether you need visual-field testing or other glaucoma investigations.",
            "If glaucoma has already been diagnosed, follow your prescribed treatment and monitoring schedule.",
            "Severe eye pain with a red eye, blurred vision, or nausea requires emergency medical assessment."
        ]
    },
    "Normal": {
        "overview": "The model's highest-scoring category is Normal.",
        "tips": [
            "Continue routine eye examinations at intervals recommended by your eye-care professional.",
            "Wear appropriate eye protection during sports, work, or activities with a risk of eye injury.",
            "Protect your eyes from excessive sunlight with UV-protective sunglasses.",
            "Do not rely on this result to rule out an eye condition; some diseases can be present without obvious symptoms."
        ]
    }
}

st.set_page_config(
    page_title="Eye Image Classification",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Eye Image Classification")
st.write(
    "Upload a retinal image to explore the output of this experimental "
    "machine-learning model."
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

uploaded_file = st.file_uploader(
    "Choose a retinal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", width=350)

        if st.button("Analyze image", type="primary"):
            with st.spinner("Processing image..."):
                model = load_model()
                resized = image.resize((224, 224))
                array = np.asarray(resized, dtype=np.float32) / 255.0
                scores = model.predict(array[None, ...], verbose=0)[0]

            if (
                not np.isfinite(scores).all()
                or len(scores) != len(CLASS_LABELS)
            ):
                st.error("The model returned invalid results. Please try again.")
            else:
                index = int(np.argmax(scores))
                predicted_class = CLASS_LABELS[index]
                guidance = EYE_CARE_GUIDANCE[predicted_class]

                st.divider()
                st.subheader("Model result")
                st.markdown(f"### {predicted_class}")
                st.write(guidance["overview"])

                st.subheader("Eye-care tips and next steps")
                for tip in guidance["tips"]:
                    st.markdown(f"- {tip}")

                st.info(
                    "If you have concerns about your vision, consult a qualified "
                    "eye-care professional. Sudden vision loss or severe eye pain "
                    "requires urgent medical attention."
                )

    except Exception:
        st.error(
            "Unable to process this image. Please try another supported image file."
        )
