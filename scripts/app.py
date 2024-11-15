from flask import Flask, request, jsonify
from recommendation import HybridRecommender
from scripts.agent import SchemeRecommenderAgent
from feedback import FeedbackProcessor
import os

app = Flask(__name__)

# Initialize components
recommender = HybridRecommender(
    schemes_data_path='data/schemes.json',
    chroma_persist_dir='models/chromadb'
)

agent = SchemeRecommenderAgent(
    recommender=recommender,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

feedback_processor = FeedbackProcessor()

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Endpoint for getting recommendations"""
    data = request.json
    user_profile = data.get('user_profile', {})
    query = data.get('query', '')
    
    # Run agent
    conversation = agent.run(user_profile, query)
    
    return jsonify({
        'conversation': conversation,
        'recommendations': agent.state.current_recommendations
    })

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Endpoint for submitting feedback"""
    data = request.json
    recommendation_id = data.get('recommendation_id')
    user_feedback = data.get('feedback', {})
    
    feedback = feedback_processor.collect_feedback(
        recommendation_id=recommendation_id,
        user_feedback=user_feedback
    )
    
    # Get insights and potential improvements
    insights = feedback_processor.get_scheme_insights(recommendation_id)
    improvements = feedback_processor.get_recommendation_improvements()
    
    return jsonify({
        'status': 'success',
        'insights': insights,
        'improvements': improvements
    })

if __name__ == '__main__':
    app.run(debug=True) 