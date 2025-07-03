import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import logging
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class MillBayesianOptimizer:
    """
    Bayesian optimization for mill parameter tuning.
    Uses a trained XGBoost model to find optimal parameter settings.
    """
    
    def __init__(self, xgboost_model, target_col='PSI80', maximize=True):
        """
        Initialize the Bayesian optimizer
        
        Args:
            xgboost_model: Trained XGBoost model instance
            target_col: Target column to optimize (default: 'PSI80')
            maximize: Whether to maximize (True) or minimize (False) the target
        """
        self.model = xgboost_model
        self.target_col = target_col
        self.maximize = maximize
        self.optimizer = None
        self.pbounds = {}
        self.param_constraints = {}
        self.best_params = None
        self.optimization_history = []
        
        logger.info(f"Bayesian optimizer initialized for {'maximizing' if maximize else 'minimizing'} {target_col}")
    
    def _black_box_function(self, **kwargs):
        """
        Black box function that predicts the outcome using the XGBoost model
        
        Args:
            **kwargs: Parameter values to optimize
            
        Returns:
            Predicted target value (negated if minimizing)
        """
        try:
            # Create a DataFrame with the parameters
            param_values = {k: float(v) for k, v in kwargs.items()}
            data = pd.DataFrame([param_values])
            
            # Check if parameters meet constraints
            if not self._check_constraints(param_values):
                # Return a very poor score to discourage invalid combinations
                return -float('inf') if self.maximize else float('inf')
            
            # Make prediction using the model
            prediction = self.model.predict(data)[0]
            
            # Log this iteration
            self.optimization_history.append({
                'params': param_values,
                'prediction': float(prediction),
                'timestamp': datetime.now().isoformat()
            })
            
            # Return prediction (negative if minimizing)
            return prediction if self.maximize else -prediction
            
        except Exception as e:
            logger.error(f"Error in black box function: {e}")
            # Return a very bad value in case of error
            return -float('inf') if self.maximize else float('inf')
    
    def set_parameter_bounds(self, pbounds=None, data=None):
        """
        Set the parameter bounds for optimization
        
        Args:
            pbounds: Dictionary mapping parameter names to (min, max) tuples
            data: Optional DataFrame to derive bounds from
            
        Returns:
            Dictionary of parameter bounds
        """
        if pbounds is not None:
            self.pbounds = pbounds
        elif data is not None:
            # Derive bounds from data
            self.pbounds = {}
            for feature in self.model.features:
                if feature in data.columns:
                    # Add a small buffer to min/max values
                    min_val = data[feature].min() * 0.95  # 5% below minimum
                    max_val = data[feature].max() * 1.05  # 5% above maximum
                    self.pbounds[feature] = (min_val, max_val)
        else:
            # Default bounds for common parameters
            self.pbounds = {
                'Ore': (160.0, 200.0),
                'WaterMill': (12.0, 18.0),
                'WaterZumpf': (140.0, 240.0),
                'PressureHC': (0.3, 0.5),
                'DensityHC': (1500, 1800),
                'MotorAmp': (170.0, 220.0)
            }
        
        logger.info(f"Parameter bounds set: {self.pbounds}")
        return self.pbounds
    
    def set_constraints(self, constraints=None):
        """
        Set constraints on parameter combinations
        
        Args:
            constraints: Dictionary of constraint functions
                        Each key is a constraint name, value is a function that
                        takes parameter dictionary and returns True if valid
        """
        self.param_constraints = constraints or {}
        logger.info(f"Set {len(self.param_constraints)} parameter constraints")
    
    def _check_constraints(self, params):
        """
        Check if parameters meet all constraints
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            True if all constraints are met, False otherwise
        """
        for name, constraint_func in self.param_constraints.items():
            if not constraint_func(params):
                logger.debug(f"Constraint '{name}' not met with params: {params}")
                return False
        return True
    
    def optimize(self, init_points=5, n_iter=25, acq='ei', kappa=2.5, xi=0.0, save_dir=None):
        """
        Run the Bayesian optimization process
        
        Args:
            init_points: Number of random exploration steps before exploitation
            n_iter: Number of optimization iterations
            acq: Acquisition function type ('ei', 'ucb', or 'poi')
            kappa: Parameter for 'ucb' acquisition function
            xi: Parameter for 'ei' and 'poi' acquisition functions
            save_dir: Optional directory to save optimization results
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Check if parameter bounds are set
            if not self.pbounds:
                logger.warning("Parameter bounds not set, using defaults")
                self.set_parameter_bounds()
            
            # Initialize optimizer
            self.optimizer = BayesianOptimization(
                f=self._black_box_function,
                pbounds=self.pbounds,
                random_state=42
            )
            
            # Set utility function
            utility = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
            
            # Clear history
            self.optimization_history = []
            
            # Run optimization
            logger.info(f"Starting optimization with {init_points} initial points and {n_iter} iterations")
            
            for i in range(init_points):
                # Random exploration
                next_point = self.optimizer.suggest(utility)
                target = self._black_box_function(**next_point)
                self.optimizer.register(params=next_point, target=target)
                logger.info(f"Exploration step {i+1}/{init_points}, target: {target:.4f}")
            
            for i in range(n_iter):
                # Guided optimization
                next_point = self.optimizer.suggest(utility)
                target = self._black_box_function(**next_point)
                self.optimizer.register(params=next_point, target=target)
                logger.info(f"Optimization step {i+1}/{n_iter}, target: {target:.4f}")
            
            # Get the best parameters
            self.best_params = self.optimizer.max['params']
            best_target = self.optimizer.max['target']
            
            # Calculate actual prediction for the best parameters
            best_prediction = best_target if self.maximize else -best_target
            
            logger.info(f"Optimization complete. Best target: {best_prediction:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
            
            # Save results if directory provided
            if save_dir:
                self._save_optimization_results(save_dir)
            
            return {
                'best_params': self.best_params,
                'best_target': best_prediction,
                'maximize': self.maximize,
                'target_col': self.target_col,
                'history': self.optimization_history
            }
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            raise
    
    def _save_optimization_results(self, directory):
        """
        Save optimization results to disk
        
        Args:
            directory: Directory to save results
        """
        os.makedirs(directory, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(directory, f"optimization_results_{timestamp}.json")
        
        results = {
            'target_col': self.target_col,
            'maximize': self.maximize,
            'bounds': self.pbounds,
            'best_params': self.best_params,
            'best_target': self.optimizer.max['target'],
            'history': self.optimization_history,
            'timestamp': timestamp
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {filename}")
    
    def recommend_parameters(self, n_recommendations=3):
        """
        Get top N parameter recommendations from optimization history
        
        Args:
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of dictionaries with parameter recommendations
        """
        if not self.optimization_history:
            logger.warning("No optimization history available")
            return []
        
        # Convert history to DataFrame
        df = pd.DataFrame([
            {**item['params'], 'prediction': item['prediction']}
            for item in self.optimization_history
        ])
        
        # Sort by prediction (high to low if maximizing, low to high if minimizing)
        ascending = not self.maximize
        df_sorted = df.sort_values('prediction', ascending=ascending)
        
        # Get top N recommendations
        top_n = df_sorted.head(n_recommendations)
        
        # Format recommendations
        recommendations = []
        for _, row in top_n.iterrows():
            params = {col: row[col] for col in row.index if col != 'prediction'}
            recommendations.append({
                'params': params,
                'predicted_value': row['prediction']
            })
        
        return recommendations
