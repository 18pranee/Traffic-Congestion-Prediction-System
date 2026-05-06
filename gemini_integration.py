import google.generativeai as genai
import os

class GeminiTrafficAnalyzer:
    """Gemini AI integration for enhanced traffic prediction insights"""
    
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.enabled = True
        else:
            self.model = None
            self.enabled = False
    
    def analyze_traffic_prediction(self, prediction_data):
        """Get AI-enhanced insights for traffic prediction"""
        if not self.enabled:
            return {
                'enabled': False,
                'insights': 'Gemini AI not configured. Set GEMINI_API_KEY to enable enhanced insights.',
                'recommendations': []
            }
        
        try:
            prompt = f"""Analyze this traffic congestion prediction data and provide insights:

Congestion Level: {prediction_data['prediction']['congestion_level']}
Confidence: {prediction_data['prediction']['confidence']:.2%}

Input Data:
- Vehicle Count: {prediction_data['input_data']['vehicle_count']}
- Average Speed: {prediction_data['input_data']['average_speed']} km/h
- Road Occupancy: {prediction_data['input_data']['road_occupancy']:.2%}
- Weather: {prediction_data['input_data']['weather_condition']}
- Location: {prediction_data['input_data']['latitude']}, {prediction_data['input_data']['longitude']}

Vehicle Detection:
- Detected Vehicles: {prediction_data['vehicle_detection']['detected_vehicles']}
- Density Level: {prediction_data['vehicle_detection']['density_level']}

Provide:
1. Brief traffic situation analysis (2-3 sentences)
2. Key factors contributing to this congestion level
3. Actionable recommendations for traffic management

Keep response concise and actionable."""
            
            response = self.model.generate_content(prompt)
            
            insights_text = response.text if response.text else "No insights generated"
            
            recommendations = self._extract_recommendations(insights_text)
            
            return {
                'enabled': True,
                'insights': insights_text,
                'recommendations': recommendations,
                'ai_model': 'gemini-pro'
            }
            
        except Exception as e:
            return {
                'enabled': True,
                'error': str(e),
                'insights': 'Error generating AI insights',
                'recommendations': []
            }
    
    def _extract_recommendations(self, text):
        """Extract key recommendations from insights text"""
        recommendations = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (
                line.startswith('-') or 
                line.startswith('•') or 
                line.startswith('*') or
                'recommend' in line.lower()
            ):
                clean_line = line.lstrip('-•*').strip()
                if len(clean_line) > 10:
                    recommendations.append(clean_line)
        
        return recommendations[:5]
    
    def get_traffic_forecast_summary(self, dataset_stats):
        """Generate AI summary of traffic patterns from dataset"""
        if not self.enabled:
            return "Gemini AI not configured"
        
        try:
            prompt = f"""Summarize the traffic patterns from this dataset:

Total Records: {dataset_stats['total_records']}

Congestion Distribution:
{json.dumps(dataset_stats['congestion_distribution'], indent=2)}

Weather Distribution:
{json.dumps(dataset_stats['weather_distribution'], indent=2)}

Vehicle Count: Mean {dataset_stats['vehicle_count_stats']['mean']:.0f}, Range {dataset_stats['vehicle_count_stats']['min']:.0f}-{dataset_stats['vehicle_count_stats']['max']:.0f}

Speed: Mean {dataset_stats['speed_stats']['mean']:.1f} km/h, Range {dataset_stats['speed_stats']['min']:.1f}-{dataset_stats['speed_stats']['max']:.1f} km/h

Provide a brief summary (3-4 sentences) highlighting key patterns and trends."""
            
            response = self.model.generate_content(prompt)
            
            return response.text if response.text else "No summary generated"
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def suggest_optimal_routes(self, current_location, destination, congestion_data):
        """Suggest alternative routes based on congestion predictions"""
        if not self.enabled:
            return []
        
        try:
            prompt = f"""Based on current traffic congestion data, suggest optimal routes:

Current Location: {current_location['latitude']}, {current_location['longitude']}
Destination: {destination['latitude']}, {destination['longitude']}
Current Congestion: {congestion_data['level']}

Provide 2-3 brief route suggestions considering traffic conditions."""
            
            response = self.model.generate_content(prompt)
            
            text = response.text if response.text else ""
            return self._extract_recommendations(text)
            
        except Exception as e:
            return [f"Error generating route suggestions: {str(e)}"]
