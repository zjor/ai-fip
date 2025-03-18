from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker, ModelProto
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.output_activation(x)
        return x

    def save(self, filename: str) -> None:
        torch.save(self.state_dict(), filename)

    def load(self, filename: str) -> None:
        self.load_state_dict(torch.load(filename))
        self.eval()

    def get_params(self) -> dict[str, Any]:
        params = {}
        for name, param in self.named_parameters():
            params[name] = param.detach().numpy()
        return params


class ModelTrainer:
    def __init__(self, input_dim=20, hidden_dim=10, output_dim=1, export_path: str = "simple_model.onnx"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = SimpleModel(input_dim, hidden_dim, output_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.export_path = export_path

    def generate_data(self, n_samples=1000):
        """Generate synthetic data for training and testing"""
        X, y = make_classification(n_samples=n_samples, n_features=self.input_dim,
                                   n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to PyTorch tensors
        self.X_train_tensor = torch.FloatTensor(X_train)
        self.y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
        self.X_test_tensor = torch.FloatTensor(X_test)
        self.y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))

        # Store numpy versions for ONNX testing
        self.X_test_np = X_test

        print(f"Data generated: {len(self.X_train_tensor)} training samples, {len(self.X_test_tensor)} test samples")

    def train_model(self, num_epochs=100, batch_size=32):
        """Train the PyTorch model"""
        num_batches = len(self.X_train_tensor) // batch_size

        print("Starting training...")
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0

            # Create random batches
            indices = torch.randperm(len(self.X_train_tensor))

            for i in range(num_batches):
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                batch_X = self.X_train_tensor[batch_indices]
                batch_y = self.y_train_tensor[batch_indices]

                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / num_batches:.4f}')

        print("Training completed")

    def evaluate_model(self):
        """Evaluate the trained model"""
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test_tensor)
            test_loss = self.criterion(y_pred, self.y_test_tensor).item()
            y_pred_class = (y_pred > 0.5).float()
            accuracy = (y_pred_class == self.y_test_tensor).sum().item() / len(self.y_test_tensor)

        print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
        return test_loss, accuracy

    def export_to_onnx(self):
        """Export PyTorch model to ONNX format"""
        # Example input for tracing
        dummy_input = torch.randn(1, self.input_dim)

        # Export the model
        torch.onnx.export(
            self.model,  # PyTorch model
            dummy_input,  # Example input
            self.export_path,  # Output file path
            export_params=True,  # Store the trained weights
            opset_version=11,  # ONNX version
            do_constant_folding=True,  # Optimize constant folding
            input_names=['input'],  # Input names
            output_names=['output'],  # Output names
            dynamic_axes={'input': {0: 'batch_size'},  # Variable batch size
                          'output': {0: 'batch_size'}}
        )

        print(f"Model exported to {self.export_path}")

    def verify_onnx_model(self):
        """Verify the exported ONNX model structure"""
        onnx_model = onnx.load(self.export_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure is valid!")

    def compare_outputs(self, num_samples=5):
        """Compare outputs between PyTorch and ONNX models"""
        # Create an ONNX Runtime session
        ort_session = ort.InferenceSession(self.export_path)

        # Prepare input for ONNX model (needs to be a dictionary)
        ort_inputs = {ort_session.get_inputs()[0].name: self.X_test_np[:num_samples].astype(np.float32)}

        # Run the ONNX model
        ort_outputs = ort_session.run(None, ort_inputs)

        # Get predictions from PyTorch model for the same inputs
        with torch.no_grad():
            pytorch_inputs = torch.FloatTensor(self.X_test_np[:num_samples])
            pytorch_outputs = self.model(pytorch_inputs).numpy()

        # Compare the results
        max_diff = np.abs(pytorch_outputs - ort_outputs[0]).max()

        print("\nComparing outputs:")
        print("PyTorch output:", pytorch_outputs[:3].flatten())
        print("ONNX output:", ort_outputs[0][:3].flatten())
        print(f"Maximum difference: {max_diff:.6f}")

        return max_diff


def main():
    # Initialize trainer
    trainer = ModelTrainer(input_dim=20, hidden_dim=10, output_dim=1)

    # Execute workflow
    trainer.generate_data()
    trainer.train_model(num_epochs=100, batch_size=32)
    trainer.evaluate_model()

    trainer.model.save("simple_model.pth")

    # trainer.export_to_onnx()
    # trainer.verify_onnx_model()
    # trainer.compare_outputs()

    print("\nTo visualize the ONNX model structure, you can use the netron package:")
    print("pip install netron")
    print("import netron")
    print(f"netron.start('{trainer.export_path}')")


def create_simple_network(torch_params: dict[str, Any], input_dim=20, hidden_dim=10, output_dim=1,
                          model_path="simple_model_scratch.onnx") -> ModelProto:
    """
    Create a simple neural network in ONNX format (input -> linear -> relu -> linear -> sigmoid)

    Args:
        torch_params:
        input_dim: Number of input features
        hidden_dim: Number of neurons in hidden layer
        output_dim: Number of output neurons
        model_path: Path to save the ONNX model
    """

    # Extract weights and biases from params
    layer1_weight = torch_params['layer1.weight']  # Shape: [hidden_dim, input_dim]
    layer1_bias = torch_params['layer1.bias']  # Shape: [hidden_dim]
    layer2_weight = torch_params['layer2.weight']  # Shape: [output_dim, hidden_dim]
    layer2_bias = torch_params['layer2.bias']  # Shape: [output_dim]

    # PyTorch linear layer weight is transposed compared to what ONNX MatMul expects
    layer1_weight_transposed = layer1_weight.transpose()  # Shape: [input_dim, hidden_dim]
    layer2_weight_transposed = layer2_weight.transpose()  # Shape: [hidden_dim, output_dim]

    # Create input and output tensors with dynamic batch size
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch_size', input_dim])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch_size', output_dim])

    # Define the model's weights and biases with random initialization
    # First layer weights: [input_dim, hidden_dim]
    W1_name = 'W1'

    W1 = helper.make_tensor(
        W1_name,
        TensorProto.FLOAT,
        list(layer1_weight_transposed.shape),
        layer1_weight_transposed.flatten().tolist())

    # First layer bias: [hidden_dim]
    B1_name = 'B1'
    B1 = helper.make_tensor(
        B1_name,
        TensorProto.FLOAT,
        list(layer1_bias.shape),
        layer1_bias.flatten().tolist()
    )

    # Second layer weights
    W2_name = 'W2'
    W2 = helper.make_tensor(
        W2_name,
        TensorProto.FLOAT,
        list(layer2_weight_transposed.shape),
        layer2_weight_transposed.flatten().tolist()
    )

    # Second layer bias
    B2_name = 'B2'
    B2 = helper.make_tensor(
        B2_name,
        TensorProto.FLOAT,
        list(layer2_bias.shape),
        layer2_bias.flatten().tolist()
    )

    # Create initializer nodes for weights and biases
    initializers = [W1, B1, W2, B2]

    # Create the graph nodes
    # First linear layer: input -> linear operation
    node1 = helper.make_node(
        'MatMul',
        inputs=['input', W1_name],
        outputs=['linear1_output'],
        name='linear1_matmul'
    )

    # Add bias to first linear layer
    node2 = helper.make_node(
        'Add',
        inputs=['linear1_output', B1_name],
        outputs=['linear1_bias_output'],
        name='linear1_add_bias'
    )

    # ReLU activation
    node3 = helper.make_node(
        'Relu',
        inputs=['linear1_bias_output'],
        outputs=['relu_output'],
        name='relu'
    )

    # Second linear layer
    node4 = helper.make_node(
        'MatMul',
        inputs=['relu_output', W2_name],
        outputs=['linear2_output'],
        name='linear2_matmul'
    )

    # Add bias to second linear layer
    node5 = helper.make_node(
        'Add',
        inputs=['linear2_output', B2_name],
        outputs=['linear2_bias_output'],
        name='linear2_add_bias'
    )

    # Sigmoid activation for output
    node6 = helper.make_node(
        'Sigmoid',
        inputs=['linear2_bias_output'],
        outputs=['output'],
        name='sigmoid'
    )

    # Create the graph and define the model
    graph = helper.make_graph(
        nodes=[node1, node2, node3, node4, node5, node6],
        name='simple_network',
        inputs=[X],
        outputs=[Y],
        initializer=initializers
    )

    # Create the model with metadata
    model = helper.make_model(
        graph,
        producer_name='onnx_manual',
        opset_imports=[helper.make_opsetid("", 11)]
    )

    # Check the model for correctness
    checker.check_model(model)

    # Save the model
    onnx.save(model, model_path)
    print(f"Model saved to {model_path}")

    return model


if __name__ == "__main__":
    """
    How to run
        python -m fip_env.sandbox.pytorch_to_onnx

    """
    # main()
    model = SimpleModel(input_dim=20, hidden_dim=10, output_dim=1)
    model.load("simple_model.pth")
    params = model.get_params()

    print("PyTorch model parameters extracted:")
    for name, param in params.items():
        print(f"{name}: shape {param.shape}")

    onnx_filename = "simple_model_scratch.onnx"
    create_simple_network(params)

    trainer = ModelTrainer(export_path=onnx_filename)
    trainer.generate_data()
    trainer.model.load("simple_model.pth")

    trainer.verify_onnx_model()
    trainer.compare_outputs()


