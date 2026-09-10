import tempfile
from PIL import Image
import streamlit as st
import llm_Image_detector as m

st.title("LLM Image Detector")
st.write("Upload an image to run the prediction model.")

# File uploader widget supporting common image formats
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  # Display the uploaded image
  image = Image.open(uploaded_file)
  st.image(image, caption="Uploaded Image", width=300)

  # Run prediction on button click
  if st.button("Run Prediction"):
    with st.spinner("Analyzing image..."):
      # Save the uploaded file temporarily so the detector can read it like a path
      with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

      # Get prediction using your library
      r = m.predict_image(tmp_path)

      # Display the results
      st.success("Analysis Complete!")
      st.write("### Result:")
      st.write(r)