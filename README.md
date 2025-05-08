# 🥚 OviCol

OviCol is a tool for detecting Aedes aegypti eggs in images using YOLOv8. The project consists of two main components:
- A training script (`yolo.py`) for creating and training the egg detection model
- A web interface (`app.py`) for using the trained model to detect eggs in new images

## Features

- Image tiling for better detection of small objects
- Support for various image formats (JPG, JPEG, PNG)
- Automatic handling of different color spaces (RGB, RGBA, Grayscale)
- Individual tile analysis view
- Total egg count across the image

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- ultralytics>=8.0.0
- streamlit>=1.28.0
- opencv-python-headless>=4.8.0
- numpy>=1.24.0
- Pillow>=10.0.0

## Training the Model

1. Configure the parameters in `yolo.py`:
   ```python
   API_URL = "https://aedes.almanza.cc/api/annotations/export-data?format=yolo"
   API_KEY = "secret ;)"
   ...
   TILE_SIZE = 250
   OVERLAP_RATIO = 0.2
   EPOCHS = 50
   BATCH_SIZE = 16
   ```

2. Run the training script:
   ```bash
   python yolo.py
   ```

The script will:
- Download images from the API
- Create tiles from the images
- Split the dataset into train/val/test
- Train the YOLOv8 model
- Save the best model in `model_output/weights/best.pt`

## Using the Web Interface

1. Make sure the trained model is in `model_output/weights/best.pt`

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to the displayed URL

4. Upload an image to detect eggs

## Model Architecture

- Base: YOLOv8s
- Input size: 250x250 pixels
- Tiling with 20% overlap
- Single class: egg bounding boxes

## Project Structure

```
.
├── app.py              # Streamlit web interface
├── yolo.py            # Training script
├── requirements.txt   # Python dependencies
├── dataset/          # Training data and configuration
│   ├── images/      # Training images
│   ├── labels/      # Training annotations
│   └── dataset.yaml # YOLO dataset configuration
└── model_output/    # Trained model and results
    └── weights/
        └── best.pt  # Best trained model
```

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

[MIT License](LICENSE) 