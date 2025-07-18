# SARDrone
Governor's School of Engineering Research Project

The goal of this project is to create a communication pipeline using a laptop as a basestation and an off-the-shelf model drone, utilizing LoRa to communicate vital information back to the basestation.

1. Initialize virtual environment using ".\venv\Scripts\activate"
2. Run "pip install -r requirements.txt" for the needed dependencies
3. Run unzip_annotations.py for data set
4. Download the required model file. The trained model (`model.safetensors`) is too large to upload to GitHub. Please download it from the link below and place it in the following directory:
https://drive.google.com/file/d/1t4Qq5y6GzNWc2BIDAB38h8Kmd8QZ6MZj/view?usp=sharing
After downloading, place it in:
basestation_code/gpt2_v2/model.safetensors
n. Run (...)