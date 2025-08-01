
# """
# Rainfall Prediction Pipeline using Keras
# =========================================

# This script implements a complete machine learning pipeline to predict
# chances of rainfall based on weather features.


# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import warnings
# warnings.filterwarnings('ignore')

# # Set random seeds for reproducibility
# np.random.seed(42)
# tf.random.set_seed(42)

# class RainfallPredictor:
#     """Complete rainfall prediction pipeline using Keras."""
    
#     def __init__(self, data_path='rainfall_dataset.csv'):
#         self.data_path = data_path
#         self.model = None
#         self.scaler = None
#         self.feature_columns = None
#         self.target_column = 'Chances of rain'
        
#     def load_and_explore_data(self):
#         """Load the dataset and perform basic exploration."""
#         print("=" * 50)
#         print("LOADING AND EXPLORING DATA")
#         print("=" * 50)
        
#         # Load data
#         self.df = pd.read_csv(self.data_path)
#         print(f"Dataset shape: {self.df.shape}")
#         print(f"Columns: {list(self.df.columns)}")
        
#         # Basic info
#         print("\nDataset Info:")
#         print(self.df.info())
        
#         print("\nBasic Statistics:")
#         print(self.df.describe())
        
#         # Check for missing values
#         print(f"\nMissing values:\n{self.df.isnull().sum()}")
        
#         return self.df
    
#     def visualize_data(self):
#         """Create visualizations to understand the data."""
#         print("\n" + "=" * 50)
#         print("DATA VISUALIZATION")
#         print("=" * 50)
        
#         # Create output directory for plots
#         os.makedirs('plots', exist_ok=True)
        
#         # 1. Distribution of target variable
#         plt.figure(figsize=(15, 10))
        
#         plt.subplot(2, 3, 1)
#         plt.hist(self.df[self.target_column], bins=30, alpha=0.7, color='skyblue')
#         plt.title('Distribution of Chances of Rain')
#         plt.xlabel('Chances of Rain (%)')
#         plt.ylabel('Frequency')
        
#         # 2. Correlation matrix
#         plt.subplot(2, 3, 2)
#         correlation_matrix = self.df.corr()
#         sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
#         plt.title('Feature Correlation Matrix')
        
#         # 3. Feature distributions
#         numeric_cols = self.df.select_dtypes(include=[np.number]).columns
#         for i, col in enumerate(numeric_cols[:4], 3):
#             plt.subplot(2, 3, i)
#             plt.hist(self.df[col], bins=20, alpha=0.7)
#             plt.title(f'Distribution of {col}')
#             plt.xlabel(col)
#             plt.ylabel('Frequency')
        
#         plt.tight_layout()
#         plt.savefig('plots/data_exploration.png', dpi=300, bbox_inches='tight')
#         plt.show()
        
#         # Feature vs Target relationships
#         plt.figure(figsize=(15, 10))
#         feature_cols = [col for col in self.df.columns if col != self.target_column]
        
#         for i, col in enumerate(feature_cols, 1):
#             plt.subplot(2, 3, i)
#             plt.scatter(self.df[col], self.df[self.target_column], alpha=0.5)
#             plt.xlabel(col)
#             plt.ylabel('Chances of Rain')
#             plt.title(f'{col} vs Chances of Rain')
            
#             # Add trend line
#             z = np.polyfit(self.df[col], self.df[self.target_column], 1)
#             p = np.poly1d(z)
#             plt.plot(self.df[col], p(self.df[col]), "r--", alpha=0.8)
        
#         plt.tight_layout()
#         plt.savefig('plots/feature_relationships.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def preprocess_data(self):
#         """Preprocess the data for training."""
#         print("\n" + "=" * 50)
#         print("DATA PREPROCESSING")
#         print("=" * 50)
        
#         # Separate features and target
#         self.feature_columns = [col for col in self.df.columns if col != self.target_column]
#         X = self.df[self.feature_columns].copy()
#         y = self.df[self.target_column].copy()
        
#         print(f"Features: {self.feature_columns}")
#         print(f"Target: {self.target_column}")
        
#         # Handle Wind direction as categorical (if needed)
#         # Since Wind direction has values 1-4, we might want to treat it as categorical
#         # For now, we'll keep it as numeric, but you could one-hot encode it
        
#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=None
#         )
        
#         print(f"\nTraining set size: {X_train.shape}")
#         print(f"Test set size: {X_test.shape}")
        
#         # Scale the features
#         self.scaler = StandardScaler()
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_test_scaled = self.scaler.transform(X_test)
        
#         # Convert to numpy arrays
#         self.X_train = np.array(X_train_scaled, dtype=np.float32)
#         self.X_test = np.array(X_test_scaled, dtype=np.float32)
#         self.y_train = np.array(y_train, dtype=np.float32)
#         self.y_test = np.array(y_test, dtype=np.float32)
        
#         print(f"\nFeature scaling completed.")
#         print(f"Training features shape: {self.X_train.shape}")
#         print(f"Training targets shape: {self.y_train.shape}")
        
#         return self.X_train, self.X_test, self.y_train, self.y_test
    
#     def build_model(self):
#         """Build and compile the Keras model."""
#         print("\n" + "=" * 50)
#         print("BUILDING KERAS MODEL")
#         print("=" * 50)
        
#         # Model architecture
#         model = keras.Sequential([
#             # Input layer
#             layers.Dense(128, activation='relu', input_shape=(len(self.feature_columns),)),
#             layers.BatchNormalization(),
#             layers.Dropout(0.3),
            
#             # Hidden layers
#             layers.Dense(64, activation='relu'),
#             layers.BatchNormalization(),
#             layers.Dropout(0.2),
            
#             layers.Dense(32, activation='relu'),
#             layers.Dropout(0.1),
            
#             # Output layer for regression
#             layers.Dense(1, activation='linear')
#         ])
        
#         # Compile the model
#         model.compile(
#             optimizer=keras.optimizers.Adam(learning_rate=0.001),
#             loss='mse',
#             metrics=['mae']
#         )
        
#         self.model = model
        
#         print("Model Architecture:")
#         print(model.summary())
        
#         return model
    
#     def train_model(self, epochs=100, batch_size=32, validation_split=0.2):
#         """Train the model with callbacks."""
#         print("\n" + "=" * 50)
#         print("TRAINING MODEL")
#         print("=" * 50)
        
#         # Callbacks
#         callbacks = [
#             keras.callbacks.EarlyStopping(
#                 monitor='val_loss',
#                 patience=15,
#                 restore_best_weights=True
#             ),
#             keras.callbacks.ReduceLROnPlateau(
#                 monitor='val_loss',
#                 factor=0.5,
#                 patience=10,
#                 min_lr=1e-7
#             ),
#             keras.callbacks.ModelCheckpoint(
#                 'best_model.keras',
#                 monitor='val_loss',
#                 save_best_only=True
#             )
#         ]
        
#         # Train the model
#         history = self.model.fit(
#             self.X_train, self.y_train,
#             epochs=epochs,
#             batch_size=batch_size,
#             validation_split=validation_split,
#             callbacks=callbacks,
#             verbose=1
#         )
        
#         self.history = history
#         print("Training completed!")
        
#         return history
    
#     def evaluate_model(self):
#         """Evaluate the model performance."""
#         print("\n" + "=" * 50)
#         print("MODEL EVALUATION")
#         print("=" * 50)
        
#         # Make predictions
#         y_train_pred = self.model.predict(self.X_train).flatten()
#         y_test_pred = self.model.predict(self.X_test).flatten()
        
#         # Calculate metrics
#         train_mse = mean_squared_error(self.y_train, y_train_pred)
#         test_mse = mean_squared_error(self.y_test, y_test_pred)
#         train_mae = mean_absolute_error(self.y_train, y_train_pred)
#         test_mae = mean_absolute_error(self.y_test, y_test_pred)
#         train_r2 = r2_score(self.y_train, y_train_pred)
#         test_r2 = r2_score(self.y_test, y_test_pred)
        
#         print("PERFORMANCE METRICS:")
#         print(f"Training MSE: {train_mse:.4f}")
#         print(f"Test MSE: {test_mse:.4f}")
#         print(f"Training MAE: {train_mae:.4f}")
#         print(f"Test MAE: {test_mae:.4f}")
#         #print(f"Training R²: {train_r2:.4f}")
#         #print(f"Test R²: {test_r2:.4f}")
        
#         # Plot training history
#         self.plot_training_history()
        
#         # Plot predictions vs actual
#         self.plot_predictions(y_test_pred)
        
#         return {
#             'train_mse': train_mse, 'test_mse': test_mse,
#             'train_mae': train_mae, 'test_mae': test_mae,
#             'train_r2': train_r2, 'test_r2': test_r2
#         }
    
#     def plot_training_history(self):
#         """Plot training and validation loss curves."""
#         plt.figure(figsize=(12, 4))
        
#         plt.subplot(1, 2, 1)
#         plt.plot(self.history.history['loss'], label='Training Loss')
#         plt.plot(self.history.history['val_loss'], label='Validation Loss')
#         plt.title('Model Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
        
#         plt.subplot(1, 2, 2)
#         plt.plot(self.history.history['mae'], label='Training MAE')
#         plt.plot(self.history.history['val_mae'], label='Validation MAE')
#         plt.title('Model MAE')
#         plt.xlabel('Epoch')
#         plt.ylabel('MAE')
#         plt.legend()
        
#         plt.tight_layout()
#         plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def plot_predictions(self, y_pred):
#         """Plot predictions vs actual values."""
#         plt.figure(figsize=(10, 6))
        
#         plt.subplot(1, 2, 1)
#         plt.scatter(self.y_test, y_pred, alpha=0.5)
#         plt.plot([self.y_test.min(), self.y_test.max()], 
#                 [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
#         plt.xlabel('Actual Chances of Rain')
#         plt.ylabel('Predicted Chances of Rain')
#         plt.title('Predictions vs Actual Values')
        
#         plt.subplot(1, 2, 2)
#         residuals = self.y_test - y_pred
#         plt.scatter(y_pred, residuals, alpha=0.5)
#         plt.axhline(y=0, color='r', linestyle='--')
#         plt.xlabel('Predicted Chances of Rain')
#         plt.ylabel('Residuals')
#         plt.title('Residual Plot')
        
#         plt.tight_layout()
#         plt.savefig('plots/predictions.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def predict_new_sample(self, month, wind_speed, wind_direction, sun, cloud_cover):
#         """Make a prediction for new weather data."""
#         # Prepare the input
#         new_data = np.array([[month, wind_speed, wind_direction, sun, cloud_cover]])
#         new_data_scaled = self.scaler.transform(new_data)
        
#         # Make prediction
#         prediction = self.model.predict(new_data_scaled)[0][0]
        
#         print(f"\nPREDICTION:")
#         print(f"Input: Month={month}, Wind Speed={wind_speed}, Wind Direction={wind_direction}")
#         print(f"       Sun={sun}, Cloud Cover={cloud_cover}")
#         print(f"Predicted Chances of Rain: {prediction:.1f}%")
        
#         return prediction
    
#     def run_complete_pipeline(self):
#         """Run the complete ML pipeline."""
#         print("Starting Rainfall Prediction Pipeline...")
        
#         # 1. Load and explore data
#         self.load_and_explore_data()
        
#         # 2. Visualize data
#         self.visualize_data()
        
#         # 3. Preprocess data
#         self.preprocess_data()
        
#         # 4. Build model
#         self.build_model()
        
#         # 5. Train model
#         self.train_model()
        
#         # 6. Evaluate model
#         metrics = self.evaluate_model()
        
#         # 7. Example predictions
#         print("\n" + "=" * 50)
#         print("EXAMPLE PREDICTIONS")
#         print("=" * 50)
        
#         # Test with some example data
#         examples = [
#             (6, 45, 2, 70, 30),  # Summer, moderate wind, high sun, low clouds
#             (12, 60, 1, 20, 80), # Winter, high wind, low sun, high clouds
#             (3, 25, 3, 85, 40),  # Spring, low wind, high sun, moderate clouds
#         ]
        
#         for example in examples:
#             self.predict_new_sample(*example)
        
#         print("\n" + "=" * 50)
#         print("PIPELINE COMPLETED SUCCESSFULLY!")
#         print("=" * 50)
#         print("Model saved as 'best_model.keras'")
#         print("Plots saved in 'plots/' directory")
        
#         return metrics

# def main():
#     """Main function to run the rainfall prediction pipeline."""
#     # Initialize the predictor
#     predictor = RainfallPredictor('rainfall_dataset.csv')
    
#     # Run the complete pipeline
#     metrics = predictor.run_complete_pipeline()
    
#     return predictor, metrics

# if __name__ == "__main__":
#     # Run the pipeline
#     predictor, metrics = main()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class RainfallPredictor:
    """Complete rainfall prediction pipeline using Keras."""
    
    def __init__(self, data_path='rainfall_dataset.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.target_column = 'Chances of rain'
        
    def load_and_explore_data(self):
        """Load the dataset and perform basic exploration."""
        print("=" * 50)
        print("LOADING AND EXPLORING DATA")
        print("=" * 50)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Basic info
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        # Check for missing values
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        
        return self.df
    
    def visualize_data(self):
        """Create visualizations to understand the data."""
        print("\n" + "=" * 50)
        print("DATA VISUALIZATION")
        print("=" * 50)
        
        # Create output directory for plots
        os.makedirs('plots', exist_ok=True)
        
        # 1. Distribution of target variable
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.hist(self.df[self.target_column], bins=30, alpha=0.7, color='skyblue')
        plt.title('Distribution of Chances of Rain')
        plt.xlabel('Chances of Rain (%)')
        plt.ylabel('Frequency')
        
        # 2. Correlation matrix
        plt.subplot(2, 3, 2)
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        
        # 3. Feature distributions
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for i, col in enumerate(numeric_cols[:4], 3):
            plt.subplot(2, 3, i)
            plt.hist(self.df[col], bins=20, alpha=0.7)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('plots/data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature vs Target relationships
        plt.figure(figsize=(15, 10))
        feature_cols = [col for col in self.df.columns if col != self.target_column]
        
        for i, col in enumerate(feature_cols, 1):
            plt.subplot(2, 3, i)
            plt.scatter(self.df[col], self.df[self.target_column], alpha=0.5)
            plt.xlabel(col)
            plt.ylabel('Chances of Rain')
            plt.title(f'{col} vs Chances of Rain')
            
            # Add trend line
            z = np.polyfit(self.df[col], self.df[self.target_column], 1)
            p = np.poly1d(z)
            plt.plot(self.df[col], p(self.df[col]), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('plots/feature_relationships.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self):
        """Preprocess the data for training."""
        print("\n" + "=" * 50)
        print("DATA PREPROCESSING")
        print("=" * 50)
        
        # Separate features and target
        self.feature_columns = [col for col in self.df.columns if col != self.target_column]
        X = self.df[self.feature_columns].copy()
        y = self.df[self.target_column].copy()
        
        print(f"Features: {self.feature_columns}")
        print(f"Target: {self.target_column}")
        
        # Handle Wind direction as categorical (if needed)
        # Since Wind direction has values 1-4, we might want to treat it as categorical
        # For now, we'll keep it as numeric, but you could one-hot encode it
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"\nTraining set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to numpy arrays
        self.X_train = np.array(X_train_scaled, dtype=np.float32)
        self.X_test = np.array(X_test_scaled, dtype=np.float32)
        self.y_train = np.array(y_train, dtype=np.float32)
        self.y_test = np.array(y_test, dtype=np.float32)
        
        print(f"\nFeature scaling completed.")
        print(f"Training features shape: {self.X_train.shape}")
        print(f"Training targets shape: {self.y_train.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self):
        """Build and compile the Keras model."""
        print("\n" + "=" * 50)
        print("BUILDING KERAS MODEL")
        print("=" * 50)
        
        # Model architecture
        model = keras.Sequential([
            # Input layer
            layers.Dense(128, activation='relu', input_shape=(len(self.feature_columns),)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer for regression
            layers.Dense(1, activation='linear')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        print("Model Architecture:")
        model.summary(print_fn=lambda x: print(x))

        
        return model
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.2):
        """Train the model with callbacks."""
        print("\n" + "=" * 50)
        print("TRAINING MODEL")
        print("=" * 50)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        print("Training completed!")
        
        return history
    
    def evaluate_model(self):
        """Evaluate the model performance."""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train).flatten()
        y_test_pred = self.model.predict(self.X_test).flatten()
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print("PERFORMANCE METRICS:")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        # Plot training history
        self.plot_training_history()
        
        # Plot predictions vs actual
        self.plot_predictions(y_test_pred)
        
        return {
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_r2': train_r2, 'test_r2': test_r2
        }
    
    def plot_training_history(self):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, y_pred):
        """Plot predictions vs actual values."""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Chances of Rain')
        plt.ylabel('Predicted Chances of Rain')
        plt.title('Predictions vs Actual Values')
        
        plt.subplot(1, 2, 2)
        residuals = self.y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Chances of Rain')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.savefig('plots/predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_for_tensorflow_serving(self, model_name="my_model", base_path="./models"):
        """Save model for TensorFlow Serving in the required directory structure."""
        print("\n" + "=" * 50)
        print("SAVING MODEL FOR TENSORFLOW SERVING")
        print("=" * 50)
        
        # Create the required directory structure: /models/my_model/1/
        model_path = os.path.join(base_path, model_name)
        version_path = os.path.join(model_path, "1")
        
        # Create directories if they don't exist
        os.makedirs(version_path, exist_ok=True)
        
        print(f"Saving model to: {version_path}")
        
        try:
            # Use model.export() for TensorFlow Serving SavedModel format (Keras 3)
            self.model.export(version_path)
            print(f"✓ Model exported successfully in TensorFlow SavedModel format")
            
            # Verify the structure
            if os.path.exists(os.path.join(version_path, "saved_model.pb")):
                print(f"✓ saved_model.pb found")
            if os.path.exists(os.path.join(version_path, "variables")):
                print(f"✓ variables/ directory found")
                
        except Exception as e:
            print(f"❌ Error exporting model: {e}")
            raise
        
        # Save preprocessing components
        preprocessing_path = os.path.join(model_path, 'preprocessing')
        os.makedirs(preprocessing_path, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join(preprocessing_path, 'scaler.pkl'))
        joblib.dump(self.feature_columns, os.path.join(preprocessing_path, 'feature_columns.pkl'))
        
        # Save scaler parameters as JSON for easy access
        scaler_params = {
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'feature_names': self.feature_columns
        }
        with open(os.path.join(preprocessing_path, 'scaler_params.json'), 'w') as f:
            json.dump(scaler_params, f, indent=2)
        
        print(f"✓ Preprocessing components saved to: {preprocessing_path}")
        
        # Print directory structure
        print(f"\nDirectory structure created:")
        print(f"{model_path}/")
        print(f"  ├── 1/")
        print(f"  │   ├── saved_model.pb")
        print(f"  │   ├── variables/")
        print(f"  │   │   ├── variables.data-00000-of-00001")
        print(f"  │   │   └── variables.index")
        print(f"  └── preprocessing/")
        print(f"      ├── scaler.pkl")
        print(f"      ├── feature_columns.pkl")
        print(f"      └── scaler_params.json")
        
        # Docker command for TensorFlow Serving
        print(f"\nTo start TensorFlow Serving with Docker:")
        print(f"docker run -p 8501:8501 \\")
        print(f"  --mount type=bind,source={os.path.abspath(model_path)},target=/models/{model_name} \\")
        print(f"  -e MODEL_NAME={model_name} \\")
        print(f"  tensorflow/serving")
        
        return version_path
    
    def predict_new_sample(self, month, wind_speed, wind_direction, sun, cloud_cover):
        """Make a prediction for new weather data."""
        # Prepare the input
        new_data = np.array([[month, wind_speed, wind_direction, sun, cloud_cover]])
        new_data_scaled = self.scaler.transform(new_data)
        
        # Make prediction
        prediction = self.model.predict(new_data_scaled)[0][0]
        
        print(f"\nPREDICTION:")
        print(f"Input: Month={month}, Wind Speed={wind_speed}, Wind Direction={wind_direction}")
        print(f"       Sun={sun}, Cloud Cover={cloud_cover}")
        print(f"Predicted Chances of Rain: {prediction:.1f}%")
        
        return prediction
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline."""
        print("Starting Rainfall Prediction Pipeline...")
        
        # 1. Load and explore data
        self.load_and_explore_data()
        
        # 2. Visualize data
        self.visualize_data()
        
        # 3. Preprocess data
        self.preprocess_data()
        
        # 4. Build model
        self.build_model()
        
        # 5. Train model
        self.train_model()
        
        # 6. Evaluate model
        metrics = self.evaluate_model()
        
        # 7. Save for TensorFlow Serving
        self.save_for_tensorflow_serving()
        
        # 8. Example predictions
        print("\n" + "=" * 50)
        print("EXAMPLE PREDICTIONS")
        print("=" * 50)
        
        # Test with some example data
        examples = [
            (6, 45, 2, 70, 30),  # Summer, moderate wind, high sun, low clouds
            (12, 60, 1, 20, 80), # Winter, high wind, low sun, high clouds
            (3, 25, 3, 85, 40),  # Spring, low wind, high sun, moderate clouds
        ]
        
        for example in examples:
            self.predict_new_sample(*example)
        
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("Model saved as 'best_model.keras'")
        print("TensorFlow Serving model saved in 'serving_model/' directory")
        print("Plots saved in 'plots/' directory")
        
        return metrics

def main():
    """Main function to run the rainfall prediction pipeline."""
    # Initialize the predictor
    predictor = RainfallPredictor('rainfall_dataset.csv')
    
    # Run the complete pipeline
    metrics = predictor.run_complete_pipeline()
    
    return predictor, metrics

if __name__ == "__main__":
    # Run the pipeline
    predictor, metrics = main()