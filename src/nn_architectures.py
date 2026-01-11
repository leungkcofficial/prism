"""
Neural Network Architectures for Survival Analysis

This module defines PyTorch neural network architectures for survival analysis models,
including MLP for DeepSurv, CNN-MLP for DeepHit, and LSTM for temporal sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any


class MLP(nn.Module):
    """
    Multi-layer Perceptron for DeepSurv model.
    
    Args:
        in_features: Number of input features
        hidden_dims: List of hidden layer dimensions
        out_features: Number of output features (default: 1 for DeepSurv)
        dropout: Dropout rate (default: 0.2)
        activation: Activation function (default: 'relu')
        batch_norm: Whether to use batch normalization (default: True)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        out_features: int = 1,
        dropout: float = 0.2,
        activation: str = 'relu',
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.out_features = out_features
        self.dropout_rate = dropout
        self.activation_name = activation
        self.batch_norm = batch_norm
        
        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = in_features
        
        for i, dim in enumerate(hidden_dims):
            # Add linear layer
            layers.append(nn.Linear(prev_dim, dim))
            
            # Add batch normalization if enabled
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            
            # Add activation
            layers.append(self.activation)
            
            # Add dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, out_features))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as a dictionary."""
        return {
            'in_features': self.in_features,
            'hidden_dims': self.hidden_dims,
            'out_features': self.out_features,
            'dropout': self.dropout_rate,
            'activation': self.activation_name,
            'batch_norm': self.batch_norm
        }


class CNNMLP(nn.Module):
    """
    CNN-MLP architecture for DeepHit model.
    
    This architecture uses a 1D CNN to extract features from the input,
    followed by an MLP for classification.
    
    Args:
        in_features: Number of input features
        hidden_dims: List of hidden layer dimensions for MLP
        out_features: Number of output features (number of time intervals for DeepHit)
        num_causes: Number of competing risks (default: 2)
        dropout: Dropout rate (default: 0.2)
        activation: Activation function (default: 'relu')
        batch_norm: Whether to use batch normalization (default: True)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        out_features: int,
        num_causes: int = 2,
        dropout: float = 0.2,
        activation: str = 'relu',
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.out_features = out_features
        self.num_causes = num_causes
        self.dropout_rate = dropout
        self.activation_name = activation
        self.batch_norm = batch_norm
        
        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # CNN feature extractor
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate CNN output size
        cnn_output_size = 64 * (in_features // 4)  # After 2 pooling layers with stride 2
        
        # MLP for each cause
        self.cause_networks = nn.ModuleList()
        for _ in range(num_causes):
            layers = []
            prev_dim = cnn_output_size
            
            for i, dim in enumerate(hidden_dims):
                # Add linear layer
                layers.append(nn.Linear(prev_dim, dim))
                
                # Add batch normalization if enabled
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dim))
                
                # Add activation
                layers.append(self.activation)
                
                # Add dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                
                prev_dim = dim
            
            # Add output layer
            layers.append(nn.Linear(prev_dim, out_features))
            
            # Create sequential model for this cause
            self.cause_networks.append(nn.Sequential(*layers))
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Single tensor of shape (batch_size, num_causes, out_features) for PyCox DeepHit compatibility
        """
        # Reshape for CNN (add channel dimension)
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, in_features)
        
        # CNN feature extraction
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Pass through cause-specific networks
        outputs = []
        for cause_net in self.cause_networks:
            outputs.append(cause_net(x))
        
        # Concatenate outputs and reshape for pycox DeepHit compatibility
        # pycox expects shape: (batch_size, num_causes, num_time_intervals)
        concatenated_output = torch.cat(outputs, dim=1)  # Shape: (batch_size, num_time_intervals * num_causes)
        
        # Reshape to (batch_size, num_causes, num_time_intervals)
        batch_size = concatenated_output.shape[0]
        reshaped_output = concatenated_output.view(batch_size, self.num_causes, self.out_features)
        
        return reshaped_output
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as a dictionary."""
        return {
            'in_features': self.in_features,
            'hidden_dims': self.hidden_dims,
            'out_features': self.out_features,
            'num_causes': self.num_causes,
            'dropout': self.dropout_rate,
            'activation': self.activation_name,
            'batch_norm': self.batch_norm
        }


class LSTMSurvival(nn.Module):
    """
    Pure LSTM architecture for survival analysis.
    
    This architecture uses LSTM layers to process temporal sequences of patient data,
    followed by a direct linear layer for output prediction. Suitable for both
    DeepSurv and DeepHit models.
    
    Args:
        input_dim: Number of input features per timestep
        sequence_length: Number of timesteps in each sequence
        lstm_hidden_dims: List of hidden dimensions for each LSTM layer, or single int for all layers
        output_dim: Number of output features (1 for DeepSurv, time_grid_size for DeepHit)
        dropout: Dropout rate (default: 0.2)
        bidirectional: Whether to use bidirectional LSTM (default: True)
        activation: Activation function for output layer (default: None for DeepSurv)
    """
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        lstm_hidden_dims: Union[int, List[int]],
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
        activation: str = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.activation_name = activation
        
        # Handle both single int and list of ints for hidden dimensions
        if isinstance(lstm_hidden_dims, int):
            # For backward compatibility, if single int provided, use it for all layers
            self.lstm_hidden_dims = [lstm_hidden_dims]
            self.lstm_num_layers = 1
        else:
            self.lstm_hidden_dims = lstm_hidden_dims
            self.lstm_num_layers = len(lstm_hidden_dims)
        
        # Build stacked LSTM layers
        self.lstm_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(self.lstm_hidden_dims):
            lstm_layer = nn.LSTM(
                input_size=prev_dim,
                hidden_size=hidden_dim,
                num_layers=1,  # Single layer per module for flexibility
                dropout=0,  # We'll add dropout between layers manually
                bidirectional=bidirectional,
                batch_first=True
            )
            self.lstm_layers.append(lstm_layer)
            
            # Next layer's input size is current layer's output size
            prev_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Calculate final LSTM output dimension
        final_hidden_dim = self.lstm_hidden_dims[-1]
        lstm_output_dim = final_hidden_dim * (2 if bidirectional else 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(lstm_output_dim, output_dim)
        
        # Activation function for output (if specified)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None
    
    def forward(self, x):
        """
        Forward pass through the stacked LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Pass through each LSTM layer sequentially
        current_input = x
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            # LSTM forward pass
            lstm_out, (hidden, cell) = lstm_layer(current_input)
            
            # Apply dropout between layers (except after the last layer)
            if i < len(self.lstm_layers) - 1 and self.dropout_rate > 0:
                lstm_out = self.dropout(lstm_out)
            
            # Use output as input for next layer
            current_input = lstm_out
        
        # Use the last timestep output from the final LSTM layer
        # lstm_out shape: (batch_size, sequence_length, hidden_dim * num_directions)
        # Debug: Print tensor shapes to understand the issue
        # print(f"[DEBUG LSTM] lstm_out shape: {lstm_out.shape}")
        # print(f"[DEBUG LSTM] lstm_out dimensions: {lstm_out.dim()}")
        
        # Handle different input shapes
        if lstm_out.dim() == 3:
            # Expected 3D tensor: (batch_size, sequence_length, hidden_dim * num_directions)
            last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_dim * num_directions)
        elif lstm_out.dim() == 2:
            # 2D tensor: (batch_size, hidden_dim * num_directions) - already the final output
            last_output = lstm_out
        else:
            raise ValueError(f"Unexpected LSTM output shape: {lstm_out.shape}")
        
        # Apply final dropout
        last_output = self.dropout(last_output)
        
        # Output layer
        output = self.output_layer(last_output)
        
        # Apply activation if specified
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as a dictionary."""
        return {
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'lstm_hidden_dims': self.lstm_hidden_dims,
            'lstm_num_layers': self.lstm_num_layers,
            'output_dim': self.output_dim,
            'dropout': self.dropout_rate,
            'bidirectional': self.bidirectional,
            'activation': self.activation_name
        }


class LSTMDeepHit(nn.Module):
    """
    LSTM architecture specifically designed for DeepHit competing risks model.
    
    This architecture uses LSTM layers to process temporal sequences, followed by
    separate output heads for each competing risk (cause).
    
    Args:
        input_dim: Number of input features per timestep
        sequence_length: Number of timesteps in each sequence
        lstm_hidden_dims: List of hidden dimensions for each LSTM layer, or single int for all layers
        output_dim: Number of time intervals in the time grid
        num_causes: Number of competing risks (default: 2)
        dropout: Dropout rate (default: 0.2)
        bidirectional: Whether to use bidirectional LSTM (default: True)
    """
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        lstm_hidden_dims: Union[int, List[int]],
        output_dim: int = 10,
        num_causes: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.num_causes = num_causes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        
        # Handle both single int and list of ints for hidden dimensions
        if isinstance(lstm_hidden_dims, int):
            # For backward compatibility, if single int provided, use it for all layers
            self.lstm_hidden_dims = [lstm_hidden_dims]
            self.lstm_num_layers = 1
        else:
            self.lstm_hidden_dims = lstm_hidden_dims
            self.lstm_num_layers = len(lstm_hidden_dims)
        
        # Build stacked LSTM layers
        self.lstm_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(self.lstm_hidden_dims):
            lstm_layer = nn.LSTM(
                input_size=prev_dim,
                hidden_size=hidden_dim,
                num_layers=1,  # Single layer per module for flexibility
                dropout=0,  # We'll add dropout between layers manually
                bidirectional=bidirectional,
                batch_first=True
            )
            self.lstm_layers.append(lstm_layer)
            
            # Next layer's input size is current layer's output size
            prev_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Calculate final LSTM output dimension
        final_hidden_dim = self.lstm_hidden_dims[-1]
        lstm_output_dim = final_hidden_dim * (2 if bidirectional else 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Separate output heads for each cause
        self.cause_heads = nn.ModuleList([
            nn.Linear(lstm_output_dim, output_dim) for _ in range(num_causes)
        ])
    
    def forward(self, x):
        """
        Forward pass through the stacked LSTM DeepHit network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            List of output tensors for each cause, each of shape (batch_size, output_dim)
        """
        # Pass through each LSTM layer sequentially
        current_input = x
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            # LSTM forward pass
            lstm_out, (hidden, cell) = lstm_layer(current_input)
            
            # Apply dropout between layers (except after the last layer)
            if i < len(self.lstm_layers) - 1 and self.dropout_rate > 0:
                lstm_out = self.dropout(lstm_out)
            
            # Use output as input for next layer
            current_input = lstm_out
        
        # Use the last timestep output from the final LSTM layer
        if self.bidirectional:
            # For bidirectional LSTM, concatenate forward and backward hidden states
            forward_hidden = hidden[0]  # Forward direction
            backward_hidden = hidden[1]  # Backward direction
            last_output = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # For unidirectional LSTM, use the hidden state
            last_output = hidden[0]
        
        # Apply final dropout
        last_output = self.dropout(last_output)
        
        # Generate outputs for each cause
        outputs = []
        for cause_head in self.cause_heads:
            cause_output = cause_head(last_output)
            outputs.append(cause_output)
        
        # Concatenate outputs and reshape for pycox DeepHit compatibility
        # pycox expects shape: (batch_size, num_causes, num_time_intervals)
        concatenated_output = torch.cat(outputs, dim=1)  # Shape: (batch_size, num_time_intervals * num_causes)
        
        # Reshape to (batch_size, num_causes, num_time_intervals)
        batch_size = concatenated_output.shape[0]
        reshaped_output = concatenated_output.view(batch_size, self.num_causes, self.output_dim)
        
        return reshaped_output
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as a dictionary."""
        return {
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'lstm_hidden_dims': self.lstm_hidden_dims,
            'lstm_num_layers': self.lstm_num_layers,
            'output_dim': self.output_dim,
            'num_causes': self.num_causes,
            'dropout': self.dropout_rate,
            'bidirectional': self.bidirectional
        }


def create_network(
    model_type: str,
    input_dim: int,
    hidden_dims: List[int] = None,
    output_dim: int = 1,
    dropout: float = 0.2,
    activation: str = 'relu',
    batch_norm: bool = True,
    num_causes: int = 2,
    network_type: str = 'ann',
    sequence_length: int = None,
    lstm_hidden_dim: int = 64,
    lstm_num_layers: int = 2,
    bidirectional: bool = True
) -> nn.Module:
    """
    Factory function to create a neural network for survival analysis.
    
    Args:
        model_type: Type of model ('deepsurv' or 'deephit')
        input_dim: Number of input features (per timestep for LSTM)
        hidden_dims: List of hidden layer dimensions for MLP (default: [128, 64, 32])
        output_dim: Number of output features (default: 1 for DeepSurv, num_durations for DeepHit)
        dropout: Dropout rate (default: 0.2)
        activation: Activation function (default: 'relu')
        batch_norm: Whether to use batch normalization for MLP (default: True)
        num_causes: Number of competing risks for DeepHit (default: 2)
        network_type: Type of network architecture ('ann' or 'lstm', default: 'ann')
        sequence_length: Number of timesteps for LSTM (required if network_type='lstm')
        lstm_hidden_dim: Hidden dimension for LSTM layers (default: 64)
        lstm_num_layers: Number of LSTM layers (default: 2)
        bidirectional: Whether to use bidirectional LSTM (default: True)
        
    Returns:
        PyTorch neural network
    """
    if hidden_dims is None:
        hidden_dims = [128, 64, 32]
    
    # Validate LSTM parameters
    if network_type.lower() == 'lstm' and sequence_length is None:
        raise ValueError("sequence_length must be specified when network_type='lstm'")
    
    # Create network based on network type and model type
    if network_type.lower() == 'ann':
        # Traditional MLP-based networks
        if model_type.lower() == 'deepsurv':
            return MLP(
                in_features=input_dim,
                hidden_dims=hidden_dims,
                out_features=output_dim,
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm
            )
        elif model_type.lower() == 'deephit':
            return CNNMLP(
                in_features=input_dim,
                hidden_dims=hidden_dims,
                out_features=output_dim,
                num_causes=num_causes,
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm
            )
        else:
            raise ValueError(f"Unsupported model type for ANN: {model_type}")
    
    elif network_type.lower() == 'lstm':
        # LSTM-based networks
        # Handle both single hidden_dim and list of hidden_dims
        if isinstance(lstm_hidden_dim, int) and lstm_num_layers > 1:
            # If single hidden_dim provided but multiple layers requested, replicate it
            lstm_hidden_dims = [lstm_hidden_dim] * lstm_num_layers
        elif isinstance(lstm_hidden_dim, int):
            # Single layer case
            lstm_hidden_dims = lstm_hidden_dim
        else:
            # List of hidden dimensions provided
            lstm_hidden_dims = lstm_hidden_dim
        
        if model_type.lower() == 'deepsurv':
            return LSTMSurvival(
                input_dim=input_dim,
                sequence_length=sequence_length,
                lstm_hidden_dims=lstm_hidden_dims,
                output_dim=output_dim,
                dropout=dropout,
                bidirectional=bidirectional,
                activation=None  # DeepSurv typically doesn't use output activation
            )
        elif model_type.lower() == 'deephit':
            return LSTMDeepHit(
                input_dim=input_dim,
                sequence_length=sequence_length,
                lstm_hidden_dims=lstm_hidden_dims,
                output_dim=output_dim,
                num_causes=num_causes,
                dropout=dropout,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unsupported model type for LSTM: {model_type}")
    
    else:
        raise ValueError(f"Unsupported network type: {network_type}. Supported types are 'ann' and 'lstm'")