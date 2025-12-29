# Two-Way Sign Language Communication System

## Overview
This is a functional prototype for a two-way sign language communication system that combines:
- **Gesture Recognition**: Static and dynamic gestures using rule-based classifier (MediaPipe hand landmarks) and trained CNN/LSTM models for improved accuracy.
- **Speech Processing**: Speech-to-Text (STT) using Google API (with offline fallback) and Text-to-Speech (TTS) using pyttsx3 (offline) or gTTS (online).
- **NLP Cleanup**: Text normalization, sentence building from gestures, and basic grammar rules.
- **Hand Tracking**: Real-time detection with MediaPipe.
- **Video Processing**: Frame enhancement and buffering for better accuracy.
- **Streamlit GUI**: Web interface with camera input, mode switching, and real-time feedback.
- **ML Training Pipeline**: Complete workflow for training CNN and LSTM models on sign language datasets (WLASL/MS-ASL).

The system supports two modes:
- **Sign-to-Speech**: Recognize gestures from camera → Translate to text → Optional TTS output.
- **Speech-to-Sign**: Speech input → Recognize text → Display corresponding gestures (text-based visualization).

The system now includes trained ML models for superior gesture recognition accuracy compared to the rule-based MVP.

## Features
- Real-time webcam capture and hand landmark detection.
- **Dual Recognition System**: Rule-based classifier for quick setup + trained CNN/LSTM models for high accuracy.
- Static gesture recognition (CNN) and dynamic sequence recognition (LSTM).
- Data augmentation and hyperparameter tuning for model training.
- Builds coherent sentences from gesture sequences.
- STT with microphone calibration and error handling (works without mic by disabling feature).
- TTS playback with voice configuration.
- Conversation history with timestamps.
- Confidence-based detection and UI feedback.
- Model evaluation with confusion matrices and performance metrics.
- Error handling for containerized environments (e.g., no audio hardware).

Supported Gestures (Static + Dynamic):
- **Static Gestures (20)**: hello, goodbye, thank_you, please, yes, no, help, stop, ok, good, bad, water, food, eat, drink, more, finished, sorry, love, peace
- **Dynamic Gestures (15)**: how_are_you, nice_to_meet_you, what_is_your_name, where_is_bathroom, i_need_help, thank_you_very_much, have_a_good_day, see_you_later, i_am_hungry, i_am_thirsty, excuse_me, i_am_sorry, i_love_you, good_morning, good_night

See `data/gesture_labels.py` for full mappings and categories.

## Quickstart

### Prerequisites
- Python 3.11+
- Webcam and microphone (optional; app handles missing hardware).
- Browser for Streamlit interface.

### Installation
1. Clone or download the repository.
2. Install dependencies using uv (recommended) or pip:
   ```
   uv sync  # Installs from pyproject.toml
   ```
   Or with pip:
   ```
   pip install -r requirements.txt  # If you generate one from pyproject.toml
   ```
   
   Key packages: `streamlit`, `opencv-python`, `mediapipe`, `tensorflow`, `speechrecognition`, `pyttsx3`, `gtts`, `pygame`, `numpy`, `pillow`, `joblib`.

3. (Optional) For offline STT, install PocketSphinx: `pip install pocketsphinx`.

### Running the App
1. Start the Streamlit server:
   ```
   streamlit run app.py --server.port 5000
   ```
2. Open your browser to `http://localhost:5000`.
3. Grant camera/microphone permissions.
4. Use sidebar controls:
   - Select mode (Sign-to-Speech or Speech-to-Sign).
   - Enable camera and start detection.
   - Adjust confidence threshold (0.7 recommended).
   - For Speech-to-Sign: Click "Listen for Speech".
   - Click "Speak Translation" for TTS output.
5. Test gestures in front of the camera (e.g., wave for "hello").

### Usage Examples
- **Sign-to-Speech**: Make gestures like wave → thumbs_up. App detects, builds sentence (e.g., "Hello good"), displays text, and speaks on button press.
- **Speech-to-Sign**: Speak "Hello, how are you?" → App recognizes, cleans text, shows gesture sequence (e.g., "hello how_are_you").
- Clear history with the sidebar button.

## Project Structure
- `app.py`: Main Streamlit application with ML model integration.
- `models/`: Gesture recognition modules (CNN, LSTM, rule-based classifier, unified interface).
- `utils/`: Hand tracking, speech processing, NLP cleanup, video processing.
- `data/`: Gesture labels, dataset loader, and processed data.
- `scripts/`: Training and evaluation scripts (download_dataset.py, preprocess_dataset.py, train_cnn.py, evaluate_cnn.py, test_realtime.py).
- `pyproject.toml`: Dependencies (managed with uv).

## ML Model Training

### Prerequisites for Training
- **Dataset**: WLASL (21K videos) or MS-ASL (25K videos) - download manually or use provided scripts
- **Hardware**: GPU recommended for faster training (CPU works but slower)
- **Storage**: ~50GB free space for datasets and models

### Training Workflow

1. **Download Dataset**:
   ```bash
   python scripts/download_dataset.py
   # Choose WLASL or MS-ASL (manual download required for full datasets)
   ```

2. **Preprocess Dataset**:
   ```bash
   python scripts/preprocess_dataset.py
   # Extracts hand landmarks from videos, saves processed features
   ```

3. **Train CNN Model (Static Gestures)**:
   ```bash
   python scripts/train_cnn.py --dataset data/processed_wlasl/processed_features.pkl --epochs 100 --augmentation
   # Includes data augmentation, cross-validation, hyperparameter search options
   ```

4. **Evaluate Model**:
   ```bash
   python scripts/evaluate_cnn.py --model models/trained_models/cnn_gesture_model.h5
   # Generates confusion matrices, accuracy reports, and performance plots
   ```

5. **Test Real-time Performance**:
   ```bash
   python scripts/test_realtime.py --model models/trained_models/cnn_gesture_model.h5
   # Webcam testing with FPS and detection rate metrics
   ```

### Training Options
- **Data Augmentation**: `--augmentation --aug_factor 3`
- **Cross-validation**: `--cross_val --cv_folds 5`
- **Hyperparameter Search**: `--hp_search --hp_evals 10`
- **Model Comparison**: `--compare model1.h5 model2.h5`

### Expected Performance
- **CNN Static Gestures**: 85-95% accuracy on test set (depends on dataset quality)
- **Real-time FPS**: 15-30 FPS on modern hardware
- **Training Time**: 2-8 hours on GPU, 8-24 hours on CPU

## Development
- **Training Models**: Use the complete pipeline above with WLASL/MS-ASL datasets.
- **Expanding Gestures**: Add new gesture classes to `data/gesture_labels.py` and retrain models.
- **Enhancements**:
  - Train LSTM models for dynamic sequences (`scripts/train_lstm.py` - placeholder).
  - Integrate Whisper for better STT.
  - Visual gesture animations in Speech-to-Sign mode.
  - Mobile responsiveness.
- **Testing**: Run training scripts, evaluate models, test real-time performance.

## Known Limitations
- **Model Training**: Requires large datasets (WLASL/MS-ASL) and significant compute resources.
- **Real-time Performance**: ML models may be slower than rule-based on low-end hardware.
- **Dynamic Gestures**: LSTM training script is placeholder; needs implementation for sequence data.
- **Dataset Availability**: Full WLASL/MS-ASL downloads require manual setup.
- **Accuracy**: Model performance depends on training data quality and gesture variations.
- **STT**: Requires internet for Google API; offline fallback is basic.
- **Containerized Environments**: May have camera/audio issues (app handles gracefully).
- **Performance**: FPS ~15-30 on standard hardware; lower with complex models.

## Troubleshooting
- **Camera not working**: Check browser permissions; ensure OpenCV/MediaPipe installed.
- **Microphone errors**: App falls back to disabled STT; install pyaudio if needed (`pip install pyaudio`).
- **TTS silent**: Check audio output; try switching to gTTS (internet required).
- **Model loading errors**: Ensure model paths in `app.py` are correct; check TensorFlow version compatibility.
- **Training failures**: Verify dataset preprocessing completed; check GPU memory if using CUDA.
- **Low accuracy**: Try data augmentation, hyperparameter tuning, or different dataset subsets.
- **Slow inference**: Use model optimization or fallback to rule-based classifier.
- **Import errors**: Run `uv sync` or `pip install -e .`; ensure all dependencies installed.
- **Streamlit issues**: Update Streamlit (`pip install --upgrade streamlit`).

## Contributing
Fork the repo, create a branch, make changes, and submit a PR. Focus on model training, new gestures, or UI improvements.

## License
MIT License. See LICENSE file for details.

## Acknowledgments
- MediaPipe for hand tracking.
- Streamlit for the web interface.
- SpeechRecognition, pyttsx3, gTTS for audio processing.
- TensorFlow for ML models.

For questions or issues, open a GitHub issue.
