# Traffic Congestion Prediction System

## Overview

An AI-powered traffic congestion prediction system that leverages satellite imagery and machine learning to forecast traffic conditions 30-60 minutes in advance. The system uses a hybrid CNN-LSTM architecture to analyze spatial and temporal patterns from satellite data, combined with real-time vehicle detection and Gemini AI integration for enhanced insights. Built with Flask for the web interface and designed for lightweight deployment on Replit.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Machine Learning Pipeline

**Hybrid CNN-LSTM Architecture**: The core prediction model combines Convolutional Neural Networks (CNN) for spatial feature extraction from satellite imagery with Long Short-Term Memory (LSTM) networks for temporal pattern analysis. This addresses the need to understand both what's happening in a specific location (spatial) and how traffic patterns evolve over time (temporal).

- **CNN Component**: Processes 224x224 satellite images to extract visual features like road networks, vehicle density, and traffic patterns
- **LSTM Component**: Analyzes time-series traffic data to identify temporal trends and predict future congestion states
- **Attention Mechanism**: Custom attention layer (128 units) focuses the model on congestion-prone areas, improving prediction accuracy by weighting important features
- **Multi-class Output**: Predicts three congestion levels (Low, Medium, High) with confidence scores

**Lazy Loading**: Model is loaded on first prediction request rather than at startup to optimize memory usage and reduce initialization time in resource-constrained environments.

### Computer Vision System

**Vehicle Detection**: YOLO-inspired vehicle detection using classical computer vision techniques rather than deep learning to minimize computational overhead.

- **Image Processing Pipeline**: Grayscale conversion → Gaussian blur → Otsu's thresholding → morphological operations → contour detection
- **Vehicle Identification**: Detects objects between 3-150 pixel area as potential vehicles, with confidence scores based on size
- **Synthetic Image Generation**: Creates satellite-like images based on traffic parameters (vehicle count, congestion level) for testing and demonstration when real satellite data is unavailable

### Data Management

**Synthetic Dataset Generator**: Creates realistic traffic data for training and testing without requiring actual satellite imagery.

- **Geographic Bounds**: Simulates traffic within Bangalore coordinates (lat: 12.97-13.08, lon: 77.59-77.77)
- **Weather Distribution**: Generates realistic weather patterns (Clear: 41.3%, Cloudy: 26.4%, Rainy: 21.9%, Foggy: 10.4%)
- **Congestion Distribution**: Balanced distribution across three levels (Low: 34.9%, Medium: 35.6%, High: 29.5%)
- **Feature Generation**: Creates correlated features where high congestion correlates with high vehicle counts, low speeds, and high road occupancy
- **Storage Format**: JSON-based local file storage for simplicity and portability

**Trade-off**: JSON file storage chosen over PostgreSQL for simpler deployment and no database setup requirements, though this limits scalability for production use.

### Web Application Architecture

**Flask Backend**: Lightweight Python web framework serving REST API endpoints and HTML templates.

- **Route Structure**: Separate routes for overview (/), predictions (/dashboard), and analytics (/analytics)
- **Lazy Initialization**: Dataset generated only if missing, model loaded on first prediction
- **Session Management**: Uses environment variable for secret key with fallback for development

**Frontend Design**: Dark-themed responsive interface with three main views.

- **Static Assets**: Vanilla CSS with CSS variables for theming, Chart.js for visualizations
- **No Heavy Frameworks**: Uses plain JavaScript instead of React/Vue to minimize bundle size
- **Responsive Layout**: Sidebar navigation with grid-based card layouts for different screen sizes

### Input/Output Flow

**Prediction Request Flow**:
1. User inputs traffic parameters (vehicle count, speed, occupancy, weather, location)
2. System generates synthetic satellite image based on parameters
3. Vehicle detector analyzes image for vehicle density
4. Data processor prepares features for model input
5. CNN-LSTM model generates congestion prediction with confidence score
6. Gemini AI (if enabled) provides contextual insights and recommendations
7. Results displayed with visualizations and actionable recommendations

**Data Features**: 
- Spatial: latitude, longitude, vehicle count, road occupancy
- Temporal: average speed, weather conditions
- Derived: vehicle density level, detected vehicles from image processing

## External Dependencies

### Third-Party APIs

**Google Gemini AI Integration**: Optional AI-enhanced traffic analysis providing contextual insights.

- **Purpose**: Generates natural language explanations of traffic predictions and actionable recommendations
- **Authentication**: API key via GEMINI_API_KEY environment variable
- **Graceful Degradation**: System functions without Gemini, displaying message when API key not configured
- **Prompt Engineering**: Structured prompts request concise analysis (2-3 sentences), key contributing factors, and management recommendations

### Machine Learning Libraries

**TensorFlow/Keras**: Core deep learning framework for CNN-LSTM model.

- **Custom Components**: AttentionLayer implemented as custom Keras layer with trainable weights
- **Model Serialization**: Models saved in Keras format (.keras extension)
- **Version Compatibility**: Designed for TensorFlow 2.x with Keras as integrated API

**OpenCV (cv2)**: Computer vision library for image processing and vehicle detection.

- **Use Cases**: Image resizing, color space conversion, thresholding, morphological operations, contour detection
- **Alternative to Deep Learning**: Classical CV techniques reduce computational requirements

**NumPy/Pandas**: Numerical computing and data manipulation.

- **NumPy**: Array operations, random data generation, mathematical transformations
- **Pandas**: Dataset organization, time series handling (limited use due to JSON storage)

### Frontend Libraries

**Chart.js**: JavaScript charting library for data visualizations.

- **CDN Delivery**: Loaded from jsdelivr CDN to avoid bundling
- **Use Cases**: Traffic distribution charts, time series graphs, confidence visualizations

### Development Tools

**PIL (Pillow)**: Python Imaging Library for advanced image manipulation operations beyond OpenCV capabilities.

**Python Standard Library**: Extensive use of os, json, datetime for file handling, data serialization, and time operations.

### Environment Configuration

**Environment Variables**:
- `GEMINI_API_KEY`: Optional Gemini AI authentication
- `SESSION_SECRET`: Flask session security (defaults to dev key)

**File Paths**:
- Dataset: `data/traffic_dataset.json`
- Model: `models/cnn_lstm_traffic_model.keras`
- Static assets: `static/css/`, `static/js/`, `static/images/`
- Templates: `templates/*.html`