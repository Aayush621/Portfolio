from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import numpy as np

class HybridRecommender:
    def __init__(self, schemes_data_path: str, chroma_persist_dir: str):
        """
        Initialize the hybrid recommender with both vector and semantic search capabilities.
        
        Args:
            schemes_data_path (str): Path to schemes data
            chroma_persist_dir (str): Directory to persist ChromaDB
        """
        self.schemes = self.load_schemes(schemes_data_path)
        
        # Initialize embeddings and models
        self.semantic_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Initialize vector stores
        self.setup_vector_stores(chroma_persist_dir)

    def setup_vector_stores(self, persist_dir: str):
        """Set up vector stores for both search approaches"""
        # Prepare documents
        documents = [
            Document(
                page_content=scheme['description'],
                metadata={
                    'id': scheme['id'],
                    'name': scheme['name'],
                    'max_income': scheme['max_income'],
                    'locations': scheme['applicable_locations'],
                    'benefits': scheme['benefits']
                }
            )
            for scheme in self.schemes
        ]

        # Create text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Split documents
        split_docs = text_splitter.split_documents(documents)
        
        # Initialize vector stores
        self.semantic_store = Chroma(
            persist_directory=f"{persist_dir}/semantic",
            embedding_function=self.semantic_embeddings,
            collection_name="semantic_schemes"
        )
        
        # Add documents if store is empty
        if len(self.semantic_store.get()) == 0:
            self.semantic_store.add_documents(split_docs)

    def verify_eligibility(self, user_profile: Dict[str, Any], scheme: Dict[str, Any]) -> bool:
        """Check if user is eligible for a scheme"""
        if user_profile['income'] > scheme['max_income']:
            return False
        if user_profile['location'] not in scheme['applicable_locations']:
            return False
        return True

    def get_vector_recommendations(self, query: str, eligible_schemes: List[Dict], top_k: int = 5) -> List[Dict]:
        """Get recommendations using vector search"""
        query_embedding = self.vector_model.encode(query)
        
        scheme_embeddings = self.vector_model.encode(
            [scheme['description'] for scheme in eligible_schemes]
        )
        
        # Calculate similarities
        similarities = np.dot(scheme_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [eligible_schemes[i] for i in top_indices]

    def get_semantic_recommendations(self, query: str, eligible_schemes: List[Dict], top_k: int = 5) -> List[Dict]:
        """Get recommendations using semantic search"""
        results = self.semantic_store.similarity_search_with_score(
            query,
            k=top_k,
            filter={
                "id": {"$in": [str(scheme['id']) for scheme in eligible_schemes]}
            }
        )
        
        recommended_schemes = []
        for doc, score in results:
            scheme = next(
                (s for s in eligible_schemes if str(s['id']) == doc.metadata['id']),
                None
            )
            if scheme:
                recommended_schemes.append((scheme, score))
        
        return [scheme for scheme, _ in sorted(recommended_schemes, key=lambda x: x[1])]

    def get_hybrid_recommendations(
        self,
        user_profile: Dict[str, Any],
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.5
    ) -> List[Dict]:
        """
        Get hybrid recommendations combining vector and semantic search results
        
        Args:
            user_profile: User information
            query: Search query
            top_k: Number of recommendations to return
            vector_weight: Weight for vector search results (0-1)
        """
        # Get eligible schemes
        eligible_schemes = [
            scheme for scheme in self.schemes
            if self.verify_eligibility(user_profile, scheme)
        ]
        
        if not eligible_schemes:
            return []

        # Get recommendations from both approaches
        vector_recommendations = self.get_vector_recommendations(query, eligible_schemes, top_k)
        semantic_recommendations = self.get_semantic_recommendations(query, eligible_schemes, top_k)

        # Combine recommendations with weighted scoring
        scheme_scores = {}
        
        # Score vector recommendations
        for i, scheme in enumerate(vector_recommendations):
            scheme_scores[scheme['id']] = vector_weight * (top_k - i)
            
        # Score semantic recommendations
        for i, scheme in enumerate(semantic_recommendations):
            current_score = scheme_scores.get(scheme['id'], 0)
            scheme_scores[scheme['id']] = current_score + (1 - vector_weight) * (top_k - i)

        # Sort schemes by final scores
        ranked_schemes = sorted(
            eligible_schemes,
            key=lambda x: scheme_scores.get(x['id'], 0),
            reverse=True
        )

        return ranked_schemes[:top_k]

    def analyze_user_context(self, user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Analyze user context to adjust recommendation weights"""
        context_weights = {
            'vector_weight': 0.5,  # Default weight
            'semantic_weight': 0.5
        }
        
        # Adjust weights based on user profile completeness
        profile_completeness = sum(1 for v in user_profile.values() if v) / len(user_profile)
        
        if profile_completeness > 0.8:
            # More complete profiles benefit from semantic search
            context_weights['semantic_weight'] = 0.7
            context_weights['vector_weight'] = 0.3
        elif profile_completeness < 0.5:
            # Less complete profiles rely more on vector search
            context_weights['vector_weight'] = 0.7
            context_weights['semantic_weight'] = 0.3
            
        return context_weights

    def get_recommendations(
        self,
        user_profile: Dict[str, Any],
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Main recommendation method that analyzes context and returns hybrid recommendations
        
        Args:
            user_profile: User information
            query: Search query
            top_k: Number of recommendations to return
        """
        # Analyze user context
        weights = self.analyze_user_context(user_profile)
        
        # Get hybrid recommendations with analyzed weights
        recommendations = self.get_hybrid_recommendations(
            user_profile=user_profile,
            query=query,
            top_k=top_k,
            vector_weight=weights['vector_weight']
        )
        
        return recommendations
