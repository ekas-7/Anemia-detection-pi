# CNN Segmentation Pipeline

This folder contains the deep-learning (CNN-based) segmentation pipeline components.

Structure:
- data/: Datasets (Eyes Defy, CP, Conjunctiva)
- preprocessing/: Image normalization and noise removal scripts
- segmentation/: U-Net / Mask R-CNN implementations and annotations
- models/: CNN architectures (ResNet, EfficientNet) and checkpoints
- training/: Training scripts and configuration files
- inference/: Model serving and inference scripts

Notes:
- Intended for cloud or high-performance GPU systems.
CNN Conjunctiva Segmentation Pipeline
This project contains a U-Net based pipeline to segment the conjunctiva from eye images. It is structured to separate data loading, model definition, training, and prediction into distinct modules.

Project Structure
cnn_segmentation_pipeline/
│
├── src/
│   ├── __init__.py           # Makes src a Python package
│   ├── data_loader.py      # Handles loading and processing of image data
│   └── model.py              # Defines the U-Net architecture and custom loss
│
├── train.py                  # Main script to train the model
├── predict.py                # Script to run inference on a new image
├── requirements.txt          # Project dependencies
└── README.md                 # This file

Setup
Clone the Repository:

git clone <your-repo-url>
cd <your-repo-name>

Install Dependencies:
Navigate into the cnn_segmentation_pipeline directory and install the required packages.

cd cnn_segmentation_pipeline
pip install -r requirements.txt

Place Dataset:
Make sure your eyes-defy-anemia-dataset folder is located outside the cnn_segmentation_pipeline directory, at the root of your project.

How to Use
1. Train the Model
To start the training process, run the train.py script from within the cnn_segmentation_pipeline folder:

python train.py

This will process the data, train the U-Net, and save two model files: unet_conjunctiva.keras and unet_conjunctiva.tflite.

2. Run Prediction
To segment the conjunctiva on a new image, use the predict.py script. Make sure the model files from the training step are present in the same directory.

python predict.py --image "path/to/your/eye_image.jpg"

Replace "path/to/your/eye_image.jpg" with the actual path to your image. This will display the original image alongside the colored segmentation results from both the Keras and TFLite models.

Pushing to GitHub
Once you have placed these new files in your local cnn_segmentation_pipeline folder, you can push them to your repository using the standard Git workflow:

# From the root of your project directory
git add cnn_segmentation_pipeline/
git commit -m "Refactor segmentation code into a structured pipeline"
git push
