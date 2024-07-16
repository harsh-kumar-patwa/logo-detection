# Logo Detection 
This project sets up a machine learning pipeline to detect Pepsi and CocaCola logos in video files using the YOLO object detection model. The pipeline extracts frames from a video, detects the logos, and returns timestamps of detected logos in a JSON file.

## Table of Contents
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Approach](#approach)
- [Demo Video](#demo-video)

## Requirements

- Python 3.8+
- `ultralytics` library
- `roboflow` library
- `cv2` (OpenCV)
- `yolov9`

## Setup

1. Clone the repository:

    ```sh
    git clone https://github.com/harsh-kumar-patwa/logo-detection.git
    cd logo-detection
    ```

2. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Download the datasets:

    The datasets for Pepsi and CocaCola logo detection are downloaded using the `roboflow` library. The dataset includes images and annotations for training the YOLO model.

    ```python
    from roboflow import Roboflow

    # When you will try to download the dataset, you will be prompted to download via code. Just copy the code generated after that. And instead of this use your lines of code.

    rf = Roboflow(api_key="your_api_key")

    # Download Pepsi dataset {You can change this as per your project dataset}
    project_pepsi = rf.workspace("your_workspace").project("pepsi-cola-detection")
    version_pepsi = project_pepsi.version(1)
    dataset_pepsi = version_pepsi.download("yolov9")

    # Download CocaCola dataset {You can change this as per your project dataset}
    project_coca_cola = rf.workspace("your_workspace").project("cocacola-puhys")
    version_coca_cola = project_coca_cola.version(3)
    dataset_coca_cola = version_coca_cola.download("yolov9")
    ```

4. Update class IDs:

    Update the class IDs in the downloaded datasets to avoid conflicts if any classid is same.

    ```python
    import os

    def update_class_ids(label_path, old_class_id, new_class_id):
        for label_file in os.listdir(label_path):
            if label_file.endswith(".txt"):
                file_path = os.path.join(label_path, label_file)
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if int(parts[0]) == old_class_id:
                        parts[0] = str(new_class_id)
                    updated_lines.append(" ".join(parts))
                with open(file_path, 'w') as file:
                    file.write("\\n".join(updated_lines) + "\\n")

    # Define the paths to the datasets
    dataset1_path = "path_to_pepsi_dataset"
    dataset2_path = "path_to_cocacola_dataset"

    # Define the subdirectories to be processed
    subdirs = ['train', 'valid', 'test']

    # Update class IDs
    for subdir in subdirs:
        labels_path = os.path.join(dataset1_path, subdir, "labels")
        if os.path.exists(labels_path):
            update_class_ids(labels_path, old_class_id=0, new_class_id=1)
    ```

## Usage

1. Run the pipeline:

    The pipeline processes the video file to detect Pepsi and CocaCola logos and generates a JSON file with the timestamps.

    ```python
    # After running this you will be asked to enter video path (only mp4) and model path.
    # I have already given my model and a test video file
    python detect_logos.py
    ```

2. Output:

    The output will be a JSON file `output_timestamps.json` containing the timestamps of detected logos:

    ```json
    {
        "Pepsi_pts": [10.1, 10.2, 10.3, ...],
        "CocaCola_pts": [20.3, 31.8, 40.12, ...]
    }
    ```

## Approach

1. **Dataset Preparation**:
    - Downloaded datasets for Pepsi and CocaCola logos using Roboflow.
        - [Cococola Dataset](https://universe.roboflow.com/hawkeg/cocacola-puhys/dataset/3)
        - [Pepsi Dataset](https://universe.roboflow.com/seagrass-1m5bu/pepsi-cola-detection)
    - Updated class IDs to ensure unique identifiers for each logo.

2. **Model Training**:
    - Trained the YOLOv9 model on the datasets.

3. **Pipeline Implementation**:
    - Extracted frames from the video using the `av` library.
    - Used the trained YOLOv9 model to detect logos in each frame.
    - Calculated timestamps and saved them in a JSON file.
    - Also added the size of the detected frame and its distance from center 


- **Approach Document**: Details of approaches considered, choices made, and reasons.
- **Functional Demo**: Video recording and code demonstration.

## Demo Video

A demonstration video is available at [loom.com](https://loom.com).
