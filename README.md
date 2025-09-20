# Face Recognition System

A complete **Face Recognition Application** built with **OpenCV** and **Python** that can detect, train, and recognize faces in real-time using your webcam. This system combines dataset generation, machine learning training, and live face recognition into a comprehensive solution.

## 🎯 Project Overview

This face recognition system consists of three main components:
1. **Dataset Generation** - Capture and store face images for training
2. **Model Training** - Train an LBPH (Local Binary Pattern Histogram) face recognizer
3. **Real-time Recognition** - Recognize faces in live video feed with confidence scoring

## ✨ Features

### Core Functionality
- **Real-time Face Detection** using Haar Cascade Classifier
- **Automated Dataset Generation** with webcam input
- **LBPH Face Recognition** for accurate identification
- **Confidence Scoring** to filter reliable recognitions
- **Multi-user Support** with user management system
- **Live Video Recognition** with bounding boxes and labels

### Technical Features
- Automatic face cropping and resizing to 200x200 pixels
- Grayscale image processing for optimal performance
- Configurable confidence threshold (default: 72%)
- Clean exit handling and resource management
- Structured data organization

## 🚀 Advantages

1. **Real-time Performance** - Fast face detection and recognition
2. **High Accuracy** - LBPH algorithm provides reliable results
3. **Easy to Use** - Simple Jupyter notebook interface
4. **Scalable** - Support for multiple users
5. **Lightweight** - Minimal dependencies and resource usage
6. **Customizable** - Adjustable confidence thresholds and parameters
7. **Open Source** - Free to use and modify

## 📋 Requirements

- **Python 3.7+**
- **OpenCV** (`opencv-contrib-python`)
- **NumPy**
- **Pillow (PIL)**
- **Jupyter Notebook**
- **Webcam** for image capture

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kalharapasan/Face-Recognition-App-01.git
   cd Face-Recognition-App-01
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-contrib-python numpy pillow jupyter
   ```

3. **Verify webcam access** (optional):
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam OK' if cap.read()[0] else 'Webcam Error'); cap.release()"
   ```

## 📖 How to Use

### Step 1: Generate Dataset
1. Open the Jupyter notebook:
   ```bash
   jupyter notebook App1.ipynb
   ```

2. Run the first cell to import OpenCV:
   ```python
   import cv2
   ```

3. Execute the dataset generation function:
   - The webcam will open automatically
   - Position your face in the frame
   - The system will capture 200 images automatically
   - Images are saved in `data/user.{id}.{img_id}.jpg` format
   - Press `Enter` key to stop early, or let it capture all 200 images

### Step 2: Train the Model
1. Run the training cell in the notebook
2. The system will:
   - Load all images from the `data/` folder
   - Extract facial features using LBPH algorithm
   - Train the classifier
   - Save the trained model as `classifier.xml`

### Step 3: Real-time Recognition
1. Execute the recognition cell
2. The webcam will open showing:
   - **Green rectangle** around detected faces
   - **Name label** for recognized users (confidence > 72%)
   - **"UNKNOWN"** label for unrecognized faces
3. Press `'q'` key to exit the recognition mode

## 📁 Project Structure

```
Face-Recognition-App-01/
├── App1.ipynb                          # Main Jupyter notebook
├── classifier.xml                      # Trained face recognition model
├── haarcascade_frontalface_default.xml # Face detection classifier
├── users.json                         # User ID to name mapping
├── README.md                          # Project documentation
└── data/                              # Training dataset folder
    ├── user.1.1.jpg                  # User 1, Image 1
    ├── user.1.2.jpg                  # User 1, Image 2
    └── ...                           # More training images
```

## ⚙️ Configuration

### Adding New Users
1. Update the `users.json` file:
   ```json
   {
     "1": "Kalhara",
     "2": "John Doe",
     "3": "Jane Smith"
   }
   ```

2. Generate dataset with appropriate user ID in the code
3. Retrain the model with new data

### Adjusting Recognition Sensitivity
- **Increase confidence threshold** (>72) for stricter recognition
- **Decrease threshold** (<72) for more lenient recognition
- Modify the threshold in the recognition function

## 🎮 Controls

- **Dataset Generation**: Press `Enter` to stop capturing images early
- **Recognition Mode**: Press `'q'` to quit the application
- **Training**: Automatic process, no user interaction required

## 🔧 Troubleshooting

### Common Issues
1. **Webcam not detected**: Check camera permissions and connections
2. **Poor recognition**: Ensure good lighting and clear face visibility
3. **Low confidence scores**: Retrain with more diverse dataset images
4. **Module errors**: Verify all dependencies are installed correctly

### Performance Tips
- Use good lighting for better face detection
- Keep face centered and at reasonable distance
- Capture diverse angles during dataset generation
- Regularly retrain model with new images

## 📊 Technical Details

- **Face Detection**: Haar Cascade Classifier
- **Recognition Algorithm**: LBPH (Local Binary Pattern Histogram)
- **Image Processing**: OpenCV
- **Input Resolution**: 200x200 grayscale images
- **Default Confidence Threshold**: 72%
- **Dataset Size**: Up to 200 images per user

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs and issues
- Suggesting new features
- Submitting pull requests
- Improving documentation

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Kalhara** - [GitHub Profile](https://github.com/Kalharapasan)

## Possible Improvements
- Add support for multiple users with unique IDs.
- Extend dataset size automatically (e.g., stop after 100 samples).
- Preprocess faces (histogram equalization, normalization).
- Integrate with a face recognition model.

---