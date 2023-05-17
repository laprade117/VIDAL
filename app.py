import os
import glob
import requests

import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from venomai import unet, predictor, preprocess

st.set_page_config(
    page_title='AI-assisted Necrosis Analysis',
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
 )

def download_models():
    # print('Downloading models...')
    for i in range(5):
        filename = f'models/unet_inference_{i}.ckpt'
        if os.path.exists(filename):
            # print(f'Inference model {i} is already downloaded. Skipping...')
            continue
        URL = f'https://github.com/laprade117/VIDAL/releases/download/inference-models/unet_inference_{i}.ckpt'
        response = requests.get(URL)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        open(f'models/unet_inference_{i}.ckpt', 'wb').write(response.content)
        # print(f'Downloading inference model {i}...')

if __name__ == '__main__':
    
    download_models()

    st.title('AI-assisted Necrosis Analysis')
    st.text('1. Upload a photo to analyize using the \'Browse files\' button the left.')
    st.text('2. Click on each lesion in the image. There should be a single white dot on each lesion you want a necrosis score computed.')
    st.text('3. Press the \'Compute\' button below to calculate the average necrosis score for the selected lesions. Individiual lesion scores are also displayed.')

    uploaded_file = st.sidebar.file_uploader("Upload an image with the template and black sheet of paper separating the mice. After uploading, wait a few seconds while the tool computes the severity scores.", type=['.jpg','.png','.tif'], accept_multiple_files=False)
    
    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image_shape = np.array(image).shape

        height = 600
        width = int(np.round((height / image_shape[0]) * image_shape[1]))

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1.0)",  # Fixed fill color with some opacity
            stroke_width=1,
            stroke_color="rgba(255, 255, 255, 1.0)",
            background_color="rgba(255, 255, 255, 1.0)",
            background_image=image,
            update_streamlit=True,
            width=width,
            height=height,
            drawing_mode="point",
            point_display_radius=5,
            key="canvas",
        )

        if canvas_result.json_data is not None:

            if st.button('Compute'):

                image = np.array(image)
                image = preprocess.preprocess_image(image, target_res=5)

                targets = canvas_result.json_data['objects']
                centers = []
                for target in targets:
                    centerY = int(np.round(image.shape[1] * (target['left'] / width)))
                    centerX = int(np.round(image.shape[0] * (target['top'] / height)))
                    centers.append([centerX, centerY])
                centers = np.array(centers)

                final_predictions = None
                for i in range(5):
                    model = unet.UNet.load_from_checkpoint(f'models/unet_inference_{i}.ckpt')
                    predictions, windows = predictor.predict_image(model, image, centers=centers, apply_preprocessing=False)
                    if i == 0:
                        final_predictions = predictions
                    else:
                        final_predictions += predictions
                predictions = final_predictions / 5
                
                nus, light_real_areas, dark_real_areas = predictor.compute_necrotic_units(predictions, windows, return_stats=True)

                masks = np.zeros((windows.shape[0], windows.shape[1], windows.shape[2], 3))
                masks[:,:,:,0] = (np.argmax(predictions, 1) == 2)
                masks[:,:,:,2] = (np.argmax(predictions, 1) == 1)
                masks = (masks * 255).astype('uint8')

                windows = list(np.array(windows, dtype=object))
                masks = list(np.array(masks, dtype=object))

                captions_nu = [f'NU: {nus[i]:.02f}' for i in range(len(windows))]
                captions = [f'Light area: {light_real_areas[i]:.02f} mm\N{SUPERSCRIPT TWO},\nDark area: {dark_real_areas[i]:.02f} mm\N{SUPERSCRIPT TWO}' for i in range(len(windows))]

                col1, col2, col3 = st.columns(3)
                col1.metric("Necrotic unit", f"{np.nanmean(nus):.02f}")
                col2.metric("Light area", f"{np.nanmean(light_real_areas):.02f} mm\N{SUPERSCRIPT TWO}")
                col3.metric("Dark area", f"{np.nanmean(dark_real_areas):.02f} mm\N{SUPERSCRIPT TWO}")

                st.image(windows, caption=captions_nu, width=174, clamp=[0,255])
                st.image(masks, caption=captions, width=174, clamp=[0,255])
                st.image(image)
