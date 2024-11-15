from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class RecommendationFeedback:
    """Structure for storing recommendation feedback"""
    recommendation_id: str
    relevance_score: float
    user_comments: str
    applied: bool
    successful: Optional[bool] = None

class FeedbackProcessor:
    def __init__(self):
        """Initialize feedback processor"""
        self.feedback_history = []
        self.scheme_performance = {}

    def collect_feedback(
        self,
        recommendation_id: str,
        user_feedback: Dict
    ) -> RecommendationFeedback:
        """Collect and structure user feedback"""
        feedback = RecommendationFeedback(
            recommendation_id=recommendation_id,
            relevance_score=user_feedback.get('relevance_score', 0.0),
            user_comments=user_feedback.get('comments', ''),
            applied=user_feedback.get('applied', False),
            successful=user_feedback.get('successful')
        )
        
        self.feedback_history.append(feedback)
        self.update_scheme_performance(feedback)
        
        return feedback

    def update_scheme_performance(self, feedback: RecommendationFeedback):
        """Update scheme performance metrics"""
        scheme_id = feedback.recommendation_id
        if scheme_id not in self.scheme_performance:
            self.scheme_performance[scheme_id] = {
                'relevance_scores': [],
                'application_rate': 0,
                'success_rate': 0
            }
        
        perf = self.scheme_performance[scheme_id]
        perf['relevance_scores'].append(feedback.relevance_score)
        
        # Update application and success rates
        total_feedbacks = len(perf['relevance_scores'])
        applications = sum(1 for f in self.feedback_history 
                         if f.recommendation_id == scheme_id and f.applied)
        successes = sum(1 for f in self.feedback_history 
                       if f.recommendation_id == scheme_id and f.successful)
        
        perf['application_rate'] = applications / total_feedbacks
        perf['success_rate'] = successes / applications if applications > 0 else 0

    def get_scheme_insights(self, scheme_id: str) -> Dict:
        """Get performance insights for a specific scheme"""
        if scheme_id not in self.scheme_performance:
            return None
            
        perf = self.scheme_performance[scheme_id]
        return {
            'average_relevance': np.mean(perf['relevance_scores']),
            'application_rate': perf['application_rate'],
            'success_rate': perf['success_rate']
        }

    def get_recommendation_improvements(self) -> List[Dict]:
        """Generate insights for improving recommendations"""
        improvements = []
        for scheme_id, perf in self.scheme_performance.items():
            avg_relevance = np.mean(perf['relevance_scores'])
            if avg_relevance < 0.7:
                improvements.append({
                    'scheme_id': scheme_id,
                    'issue': 'low_relevance',
                    'score': avg_relevance,
                    'suggestion': 'Review scheme targeting criteria'
                })
            if perf['application_rate'] < 0.3:
                improvements.append({
                    'scheme_id': scheme_id,
                    'issue': 'low_application_rate',
                    'rate': perf['application_rate'],
                    'suggestion': 'Simplify application process or improve scheme explanation'
                })
        return improvements 