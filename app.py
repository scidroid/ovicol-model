import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_cropper import st_cropper

# Configuration
TILE_SIZE = 250
OVERLAP_RATIO = 0

# Set page config
st.set_page_config(page_title="OviCol", page_icon="🥚", layout="wide")

# Add title and description
st.title("🥚 OviCol")
st.markdown("""
Cuenta con nosotros para detectar huevos de Aedes aegypti en imágenes de mosquitos.
""")


@st.cache_resource
def load_model():
    """Load the YOLO model"""
    return YOLO("model_output/weights/best.pt")


def create_tiles(image):
    """Create non-overlapping tiles from the input image"""
    height, width = image.shape[:2]
    tiles = []
    tile_positions = []

    # With no overlap, step size equals tile size
    step_size = TILE_SIZE

    # Calculate number of tiles needed in each dimension
    num_tiles_h = int(np.ceil(height / TILE_SIZE))
    num_tiles_w = int(np.ceil(width / TILE_SIZE))

    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            # Calculate tile boundaries
            x = j * TILE_SIZE
            y = i * TILE_SIZE
            x_end = min(x + TILE_SIZE, width)
            y_end = min(y + TILE_SIZE, height)

            tile = image[y:y_end, x:x_end]
            current_height, current_width = tile.shape[:2]

            # If tile is smaller than TILE_SIZE, pad it
            if current_height < TILE_SIZE or current_width < TILE_SIZE:
                pad_h = TILE_SIZE - current_height
                pad_w = TILE_SIZE - current_width
                tile = cv2.copyMakeBorder(tile,
                                          0,
                                          pad_h,
                                          0,
                                          pad_w,
                                          cv2.BORDER_CONSTANT,
                                          value=[0, 0, 0])

            tiles.append(tile)
            tile_positions.append((x, y, x_end, y_end))

    return tiles, tile_positions


def merge_predictions(original_image, tiles_predictions, tile_positions):
    """Merge predictions from tiles back into a single image"""
    height, width = original_image.shape[:2]
    merged_image = original_image.copy()
    total_detections = 0

    for pred, (x, y, x_end, y_end) in zip(tiles_predictions, tile_positions):
        if pred is not None and len(pred.boxes) > 0:
            # Get the plotted tile with detections, but without labels
            plot = pred.plot(labels=False, conf=False)
            # Crop the plot to match the original tile size (remove padding if any)
            plot = plot[:y_end - y, :x_end - x]
            # Place the plot in the merged image
            merged_image[y:y_end, x:x_end] = plot
            total_detections += len(pred.boxes)

    return merged_image, total_detections


def process_image(image, model):
    """Process the image and return results"""
    try:
        # Create a status container
        status_container = st.empty()
        progress_bar = st.progress(0)

        status_container.text("Preparando imagen...")
        # Handle different image formats
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 3 and image.dtype == np.uint8:
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                 cv2.COLOR_BGR2RGB)

        # Create tiles
        status_container.text("Creando segmentos de imagen...")
        progress_bar.progress(10)
        tiles, tile_positions = create_tiles(image)

        # Process each tile
        tiles_predictions = []
        total_tiles = len(tiles)
        status_container.text("Procesando segmentos...")

        for idx, tile in enumerate(tiles):
            # Update progress
            progress = int(10 + (idx + 1) / total_tiles * 70)
            progress_bar.progress(progress)
            status_container.text(
                f"Procesando segmento {idx + 1}/{total_tiles}...")

            # Make prediction on tile
            results = model.predict(tile)
            result = results[0]
            tiles_predictions.append(result)

        # Merge predictions for full image
        status_container.text("Combinando resultados...")
        progress_bar.progress(90)
        processed_image, total_detections = merge_predictions(
            image, tiles_predictions, tile_positions)
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        # Complete the progress
        progress_bar.progress(100)
        status_container.empty()

        return processed_image_rgb, total_detections

    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
        return None, 0


# Load the model
try:
    with st.spinner("Cargando modelo..."):
        model = load_model()
        st.success("¡Modelo cargado exitosamente! ✅")
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Selecciona una imagen...",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the uploaded image
        image = Image.open(uploaded_file)

        # Add cropping functionality
        st.write("Ajusta el recorte de la imagen:")
        cropped_image = st_cropper(image,
                                   realtime_update=True,
                                   box_color='#0000FF',
                                   aspect_ratio=None)

        # Convert to RGB mode
        cropped_image = cropped_image.convert('RGB')

        # Process image
        image_array = np.array(cropped_image)
        processed_image, num_detections = process_image(image_array, model)

        if processed_image is not None:
            # Display detection count with large, centered text
            st.markdown(f"""
            <h2 style='text-align: center; color: #1f77b4;'>
                {num_detections} huevo{'s' if num_detections != 1 else ''} detectado{'s' if num_detections != 1 else ''}
            </h2>
            """,
                        unsafe_allow_html=True)

            # Create columns for before/after images
            col1, col2 = st.columns(2)

            col1.header("Imagen Original")
            col1.image(cropped_image, use_column_width=True)

            # Display results
            col2.header("Imagen Procesada")
            col2.image(processed_image, use_column_width=True)

    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
        st.info("Por favor, intenta subir una imagen diferente.")
