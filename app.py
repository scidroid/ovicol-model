import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import math

# Configuration
TILE_SIZE = 250
OVERLAP_RATIO = 0.2

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
    """Create tiles from the input image"""
    height, width = image.shape[:2]
    tiles = []
    tile_positions = []

    step_size = int(TILE_SIZE * (1 - OVERLAP_RATIO))
    if step_size == 0:
        step_size = TILE_SIZE

    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            y_end = min(y + TILE_SIZE, height)
            x_end = min(x + TILE_SIZE, width)

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
            # Get the plotted tile with detections
            plot = pred.plot()
            # Crop the plot to match the original tile size (remove padding if any)
            plot = plot[:y_end - y, :x_end - x]
            # Place the plot in the merged image
            merged_image[y:y_end, x:x_end] = plot
            total_detections += len(pred.boxes)

    return merged_image, total_detections


def process_image(image, model):
    """Process the image and return results"""
    try:
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
        tiles, tile_positions = create_tiles(image)

        # Process each tile
        tiles_predictions = []
        processed_tiles = []
        tile_detections = []

        for tile in tiles:
            # Make prediction on tile
            results = model.predict(tile)
            result = results[0]
            tiles_predictions.append(result)

            # Get processed tile
            processed_tile = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            processed_tiles.append(processed_tile)
            tile_detections.append(len(result.boxes))

        # Merge predictions for full image
        processed_image, total_detections = merge_predictions(
            image, tiles_predictions, tile_positions)
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        return processed_image_rgb, total_detections, processed_tiles, tile_detections, tile_positions
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, 0, [], [], []


# Load the model
try:
    model = load_model()
    st.success("Model loaded successfully! ✅")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Create columns for before/after images
        col1, col2 = st.columns(2)

        # Read and display original image
        image = Image.open(uploaded_file)
        # Convert to RGB mode
        image = image.convert('RGB')
        col1.header("Original Image")
        col1.image(image, use_column_width=True)

        # Process image
        with st.spinner("Processing image..."):
            # Convert PIL Image to numpy array
            image_array = np.array(image)

            # Process the image
            processed_image, num_detections, processed_tiles, tile_detections, tile_positions = process_image(
                image_array, model)

            if processed_image is not None:
                # Display results
                col2.header("Processed Image")
                col2.image(processed_image, use_column_width=True)

                # Display detection count with large, centered text
                st.markdown(f"""
                <h2 style='text-align: center; color: #1f77b4;'>
                    Detected {num_detections} egg{'s' if num_detections != 1 else ''}
                </h2>
                """,
                            unsafe_allow_html=True)

                # Display individual tiles
                st.header("Individual Tiles")
                st.markdown("Each tile shows its individual detections")

                # Calculate number of columns (aim for tiles of about 200px width)
                page_width = 1000  # Approximate page width in pixels
                tile_width = 200  # Desired display width for each tile
                num_cols = min(4, len(processed_tiles))  # Maximum 4 columns

                # Create columns for tiles
                for i in range(0, len(processed_tiles), num_cols):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        if i + j < len(processed_tiles):
                            with cols[j]:
                                x, y, x_end, y_end = tile_positions[i + j]
                                st.image(
                                    processed_tiles[i + j],
                                    caption=
                                    f"Tile at ({x}, {y})\nDetected: {tile_detections[i + j]} eggs"
                                )

    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")
        st.info("Please try uploading a different image.")
