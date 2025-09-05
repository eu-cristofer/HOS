"""
Machine Learning Classifier for Rotordynamics Fault Detection

This module provides machine learning tools for classifying fault conditions
in rotordynamic systems using extracted spectral features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import joblib
import warnings


class FaultClassifier:
    """
    A machine learning classifier for rotordynamics fault detection.
    
    This class provides methods for training and evaluating various ML models
    for fault classification using spectral features.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the fault classifier.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('random_forest', 'svm', 'neural_network', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the machine learning model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            )
        elif self.model_type == 'neural_network':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_data(self, features_list: List[Dict], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Parameters:
        -----------
        features_list : list
            List of feature dictionaries
        labels : list
            List of fault labels
            
        Returns:
        --------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded labels
        """
        # Convert features to DataFrame
        df = pd.DataFrame(features_list)
        self.feature_names = df.columns.tolist()
        
        # Extract features and labels
        X = df.values
        y = self.label_encoder.fit_transform(labels)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the classifier.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded labels
        test_size : float
            Fraction of data to use for testing
            
        Returns:
        --------
        results : dict
            Training results including accuracy scores
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted labels (encoded)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        probabilities : np.ndarray
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (for tree-based models).
        
        Returns:
        --------
        importance : dict
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}
    
    def plot_feature_importance(self, top_n: int = 10) -> None:
        """
        Plot feature importance.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to plot
        """
        importance = self.get_feature_importance()
        if not importance:
            print("Feature importance not available for this model type")
            return
        
        # Get top N features
        top_features = list(importance.items())[:top_n]
        features, scores = zip(*top_features)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance - {self.model_type.title()}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True labels (encoded)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting confusion matrix")
        
        y_pred = self.predict(X)
        
        # Get class names
        class_names = self.label_encoder.classes_
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {self.model_type.title()}')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


def compare_models(features_list: List[Dict], labels: List[str]) -> Dict[str, Dict]:
    """
    Compare different machine learning models.
    
    Parameters:
    -----------
    features_list : list
        List of feature dictionaries
    labels : list
        List of fault labels
        
    Returns:
    --------
    results : dict
        Comparison results for different models
    """
    model_types = ['random_forest', 'svm', 'neural_network', 'gradient_boosting']
    results = {}
    
    for model_type in model_types:
        print(f"Training {model_type}...")
        
        # Create classifier
        classifier = FaultClassifier(model_type)
        
        # Prepare data
        X, y = classifier.prepare_data(features_list, labels)
        
        # Train
        train_results = classifier.train(X, y)
        
        # Store results
        results[model_type] = train_results
        
        print(f"{model_type}: Test Accuracy = {train_results['test_accuracy']:.3f}")
    
    return results


def plot_model_comparison(results: Dict[str, Dict]) -> None:
    """
    Plot comparison of different models.
    
    Parameters:
    -----------
    results : dict
        Results from compare_models function
    """
    models = list(results.keys())
    test_accuracies = [results[model]['test_accuracy'] for model in models]
    cv_means = [results[model]['cv_mean'] for model in models]
    cv_stds = [results[model]['cv_std'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, cv_means, width, yerr=cv_stds, 
                   label='CV Mean ± Std', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([model.replace('_', ' ').title() for model in models])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    from rotordynamics import generate_rotor_dataset
    
    # Generate dataset
    print("Generating rotor dataset...")
    dataset = generate_rotor_dataset(n_samples_per_fault=100)
    
    # Compare models
    print("Comparing models...")
    results = compare_models(dataset['features'], dataset['labels'])
    
    # Plot comparison
    plot_model_comparison(results)
    
    # Train best model (assuming Random Forest is best)
    print("Training best model...")
    classifier = FaultClassifier('random_forest')
    X, y = classifier.prepare_data(dataset['features'], dataset['labels'])
    classifier.train(X, y)
    
    # Plot feature importance
    classifier.plot_feature_importance()
    
    # Plot confusion matrix
    X_test = X[int(0.8*len(X)):]  # Use last 20% as test set
    y_test = y[int(0.8*len(y)):]
    classifier.plot_confusion_matrix(X_test, y_test)

