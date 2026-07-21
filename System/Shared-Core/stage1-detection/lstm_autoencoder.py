#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Autoencoder Detector
--------------------------
Deep learning reconstruction-based anomaly detector.

Uses LSTM autoencoder to learn normal patterns.
Anomalies have high reconstruction error.
"""

import numpy as np
from typing import Optional, List
from .base_detector import BaseDetector

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Create dummy classes for type hints
    Model = type('Model', (), {})
    EarlyStopping = type('EarlyStopping', (), {})


class LSTMAutoencoderDetector(BaseDetector):
    """
    LSTM Autoencoder anomaly detector.
    
    Learns to reconstruct normal patterns using LSTM layers.
    Anomalies have high reconstruction error (MSE).
    
    Parameters
    ----------
    encoder_layers : list of int, default=[64, 32]
        Number of LSTM units per encoder layer
    decoder_layers : list of int, default=[32, 64]
        Number of LSTM units per decoder layer
    dropout : float, default=0.2
        Dropout rate for regularization
    activation : str, default='tanh'
        Activation function: tanh, relu
    epochs : int, default=50
        Maximum training epochs
    batch_size : int, default=32
        Training batch size
    learning_rate : float, default=0.001
        Adam optimizer learning rate
    validation_split : float, default=0.2
        Validation data proportion
    early_stopping_patience : int, default=10
        Early stopping patience (epochs)
    threshold_percentile : float, default=98
        Percentile for reconstruction error threshold
    verbose : int, default=0
        Training verbosity (0=silent, 1=progress, 2=epoch)
    random_state : int, default=42
        Random seed
    """
    
    def __init__(
        self,
        encoder_layers: List[int] = None,
        decoder_layers: List[int] = None,
        dropout: float = 0.2,
        activation: str = 'tanh',
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        threshold_percentile: float = 98,
        verbose: int = 0,
        random_state: int = 42
    ):
        super().__init__(name='LSTMAutoencoder')
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM Autoencoder. "
                            "Install with: pip install tensorflow")
        
        self.encoder_layers = encoder_layers or [64, 32]
        self.decoder_layers = decoder_layers or [32, 64]
        self.dropout = dropout
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.threshold_percentile = threshold_percentile
        self.verbose = verbose
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.model = None
        self.threshold_ = None
        self.n_features_ = None
        
    def _build_model(self, n_features: int) -> Model:
        """
        Build LSTM autoencoder model.
        
        Parameters
        ----------
        n_features : int
            Number of input features
            
        Returns
        -------
        model : keras.Model
            Compiled autoencoder model
        """
        # Input layer (samples, timesteps=1, features)
        inputs = layers.Input(shape=(1, n_features))
        
        # Encoder
        x = inputs
        for i, units in enumerate(self.encoder_layers):
            return_sequences = (i < len(self.encoder_layers) - 1)
            x = layers.LSTM(
                units,
                activation=self.activation,
                return_sequences=return_sequences,
                dropout=self.dropout
            )(x)
        
        # Bottleneck (latent representation)
        encoded = x
        
        # Decoder
        x = layers.RepeatVector(1)(encoded)  # Repeat for timesteps
        for i, units in enumerate(self.decoder_layers):
            x = layers.LSTM(
                units,
                activation=self.activation,
                return_sequences=True,
                dropout=self.dropout
            )(x)
        
        # Output layer
        outputs = layers.TimeDistributed(layers.Dense(n_features))(x)
        
        # Build model
        model = Model(inputs=inputs, outputs=outputs, name='lstm_autoencoder')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def fit(self, X: np.ndarray) -> 'LSTMAutoencoderDetector':
        """
        Fit LSTM autoencoder on training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data (normalized)
            
        Returns
        -------
        self : LSTMAutoencoderDetector
            Fitted detector
        """
        self.n_features_ = X.shape[1]
        
        # Reshape for LSTM: (samples, timesteps=1, features)
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Build model
        self.model = self._build_model(self.n_features_)
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=self.verbose
        )
        
        # Train autoencoder
        self.model.fit(
            X_reshaped, X_reshaped,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stop],
            shuffle=True,
            verbose=self.verbose
        )
        
        # Compute reconstruction errors on training data
        reconstructions = self.model.predict(X_reshaped, verbose=0)
        reconstructions = reconstructions.reshape(X.shape)
        mse = np.mean((X - reconstructions) ** 2, axis=1)
        
        # Set threshold at specified percentile
        self.threshold_ = np.percentile(mse, self.threshold_percentile)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Binary predictions (0=normal, 1=anomaly)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")
        
        # Compute reconstruction errors
        errors = self.decision_function(X)
        
        # Flag if error > threshold
        predictions = (errors > self.threshold_).astype(int)
        
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (reconstruction MSE).
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to score
            
        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            Anomaly scores (reconstruction error, higher = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring")
        
        # Reshape for LSTM
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Reconstruct
        reconstructions = self.model.predict(X_reshaped, verbose=0)
        reconstructions = reconstructions.reshape(X.shape)
        
        # Compute MSE per sample
        mse = np.mean((X - reconstructions) ** 2, axis=1)
        
        return mse
    
    def get_params(self):
        """Get detector parameters."""
        params = super().get_params()
        params.update({
            'encoder_layers': self.encoder_layers,
            'decoder_layers': self.decoder_layers,
            'dropout': self.dropout,
            'activation': self.activation,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
            'threshold_percentile': self.threshold_percentile,
            'threshold': self.threshold_,
            'n_features': self.n_features_
        })
        return params
