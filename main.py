from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import json
import numpy as np
from datetime import datetime
import sys

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn_lstm_model import CNNLSTMModel, TrafficPredictionEngine
from models.data_processor import SatelliteImageProcessor, TrafficDataProcessor
from models.vehicle_detector import VehicleDetector
from data.synthetic_dataset import SyntheticDatasetGenerator
from gemini_integration import GeminiTrafficAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key-change-in-production')

prediction_engine = TrafficPredictionEngine()
image_processor = SatelliteImageProcessor()
data_processor = TrafficDataProcessor()
vehicle_detector = VehicleDetector()
gemini_analyzer = GeminiTrafficAnalyzer()

DATASET_PATH = 'data/traffic_dataset.json'
MODEL_PATH = 'models/cnn_lstm_traffic_model.keras'

def initialize_system():
    """Initialize dataset (model loaded on first prediction)"""
    if not os.path.exists(DATASET_PATH):
        print("Generating synthetic dataset...")
        generator = SyntheticDatasetGenerator(num_records=1000, num_days=42)
        generator.save_dataset(DATASET_PATH)
    
    print("System initialized successfully (model will load on first prediction)!")

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Real-time prediction dashboard"""
    return render_template('dashboard.html')

@app.route('/analytics')
def analytics():
    """Statistical analytics page"""
    return render_template('analytics.html')

@app.route('/api/dataset/stats', methods=['GET'])
def get_dataset_stats():
    """Get dataset statistics"""
    try:
        data = data_processor.load_dataset(DATASET_PATH)
        if not data:
            return jsonify({'error': 'Dataset not found'}), 404
        
        congestion_counts = {}
        weather_counts = {}
        vehicle_counts = []
        speeds = []
        occupancies = []
        
        for record in data:
            cong = record['congestion_level']
            congestion_counts[cong] = congestion_counts.get(cong, 0) + 1
            
            weather = record['weather_condition']
            weather_counts[weather] = weather_counts.get(weather, 0) + 1
            
            vehicle_counts.append(record['vehicle_count'])
            speeds.append(record['average_speed'])
            occupancies.append(record['road_occupancy'])
        
        stats = {
            'total_records': len(data),
            'congestion_distribution': congestion_counts,
            'weather_distribution': weather_counts,
            'vehicle_count_stats': {
                'mean': float(np.mean(vehicle_counts)),
                'min': float(np.min(vehicle_counts)),
                'max': float(np.max(vehicle_counts)),
                'std': float(np.std(vehicle_counts))
            },
            'speed_stats': {
                'mean': float(np.mean(speeds)),
                'min': float(np.min(speeds)),
                'max': float(np.max(speeds)),
                'std': float(np.std(speeds))
            },
            'occupancy_stats': {
                'mean': float(np.mean(occupancies)),
                'min': float(np.min(occupancies)),
                'max': float(np.max(occupancies)),
                'std': float(np.std(occupancies))
            }
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/records', methods=['GET'])
def get_dataset_records():
    """Get dataset records with pagination"""
    try:
        data = data_processor.load_dataset(DATASET_PATH)
        if not data:
            return jsonify({'error': 'Dataset not found'}), 404
        
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        records = data[start_idx:end_idx]
        
        return jsonify({
            'records': records,
            'total': len(data),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(data) + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_congestion():
    """Predict traffic congestion"""
    try:
        if not prediction_engine.model_loaded:
            print("Loading prediction model...")
            prediction_engine.initialize(MODEL_PATH)
            print("Model loaded successfully!")
        
        data = request.get_json()
        
        vehicle_count = float(data.get('vehicle_count', 50))
        average_speed = float(data.get('average_speed', 30))
        road_occupancy = float(data.get('road_occupancy', 0.5))
        weather_condition = data.get('weather_condition', 'Clear')
        latitude = float(data.get('latitude', 13.0))
        longitude = float(data.get('longitude', 77.6))
        congestion_level = data.get('congestion_level', 'Medium')
        
        synthetic_image = image_processor.generate_synthetic_satellite_image(
            vehicle_count, congestion_level
        )
        
        traffic_features = np.array([
            vehicle_count,
            average_speed,
            road_occupancy,
            data_processor.weather_mapping.get(weather_condition, 0),
            latitude,
            longitude
        ], dtype=np.float32)
        
        prediction = prediction_engine.predict_congestion(synthetic_image, traffic_features)
        
        vehicle_count_detected = vehicle_detector.count_vehicles(synthetic_image)
        density_level, density_value = vehicle_detector.estimate_density(synthetic_image)
        
        result = {
            'prediction': prediction,
            'input_data': {
                'vehicle_count': vehicle_count,
                'average_speed': average_speed,
                'road_occupancy': road_occupancy,
                'weather_condition': weather_condition,
                'latitude': latitude,
                'longitude': longitude
            },
            'vehicle_detection': {
                'detected_vehicles': vehicle_count_detected,
                'density_level': density_level,
                'density_value': float(density_value)
            },
            'timestamp': datetime.now().isoformat(),
            'forecast_window': '30-60 minutes'
        }
        
        gemini_insights = gemini_analyzer.analyze_traffic_prediction(result)
        result['ai_insights'] = gemini_insights
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction for multiple locations"""
    try:
        if not prediction_engine.model_loaded:
            print("Loading prediction model...")
            prediction_engine.initialize(MODEL_PATH)
            print("Model loaded successfully!")
        
        data = request.get_json()
        locations = data.get('locations', [])
        
        results = []
        for location in locations:
            vehicle_count = float(location.get('vehicle_count', 50))
            average_speed = float(location.get('average_speed', 30))
            road_occupancy = float(location.get('road_occupancy', 0.5))
            weather_condition = location.get('weather_condition', 'Clear')
            latitude = float(location.get('latitude', 13.0))
            longitude = float(location.get('longitude', 77.6))
            congestion_level = location.get('congestion_level', 'Medium')
            
            synthetic_image = image_processor.generate_synthetic_satellite_image(
                vehicle_count, congestion_level
            )
            
            traffic_features = np.array([
                vehicle_count,
                average_speed,
                road_occupancy,
                data_processor.weather_mapping.get(weather_condition, 0),
                latitude,
                longitude
            ], dtype=np.float32)
            
            prediction = prediction_engine.predict_congestion(synthetic_image, traffic_features)
            
            results.append({
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                },
                'prediction': prediction
            })
        
        return jsonify({
            'predictions': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """Analyze traffic from image"""
    try:
        data = request.get_json()
        vehicle_count = float(data.get('vehicle_count', 100))
        congestion_level = data.get('congestion_level', 'Medium')
        
        synthetic_image = image_processor.generate_synthetic_satellite_image(
            vehicle_count, congestion_level
        )
        
        vehicles = vehicle_detector.detect_vehicles(synthetic_image)
        density_level, density_value = vehicle_detector.estimate_density(synthetic_image)
        spatial_features = image_processor.extract_spatial_features(synthetic_image)
        
        return jsonify({
            'vehicle_count': len(vehicles),
            'density_level': density_level,
            'density_value': float(density_value),
            'spatial_features': spatial_features,
            'detected_vehicles': vehicles[:10]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/gemini/summary', methods=['GET'])
def get_gemini_summary():
    """Get AI-generated traffic pattern summary"""
    try:
        data = data_processor.load_dataset(DATASET_PATH)
        if not data:
            return jsonify({'error': 'Dataset not found'}), 404
        
        stats_response = get_dataset_stats()
        if isinstance(stats_response, tuple):
            return stats_response
        
        stats = stats_response.get_json()
        summary = gemini_analyzer.get_traffic_forecast_summary(stats)
        
        return jsonify({
            'summary': summary,
            'gemini_enabled': gemini_analyzer.enabled,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': prediction_engine.model_loaded,
        'dataset_available': os.path.exists(DATASET_PATH),
        'gemini_enabled': gemini_analyzer.enabled
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    initialize_system()
    app.run(host='0.0.0.0', port=5000, debug=False)
