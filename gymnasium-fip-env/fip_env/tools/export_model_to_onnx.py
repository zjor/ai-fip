import sys
from typing import Tuple

import numpy as np
import torch as th
import torch.onnx
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

import onnx
from onnx import helper, TensorProto, numpy_helper


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return self.policy(observation, deterministic=True)[0]


def export_model_to_onnx(model: PPO, filename: str):
    model_to_export = OnnxableSB3Policy(model.policy)
    model_to_export.eval()

    dummy_input = th.rand(1, *model.observation_space.shape)

    torch.onnx.export(
        model_to_export,  # Model to export
        dummy_input,  # Example input
        f"{filename}.onnx",  # Output file path
        export_params=True,  # Export model parameters (weights)
        opset_version=11,  # ONNX opset version (e.g., 11)
        do_constant_folding=True,  # optimization
        input_names=['observation'],  # the model's input names
        output_names=['actions'],  # the model's output names
        dynamic_axes={'observation': {0: 'batch_size'},  # variable length axes
                      'actions': {0: 'batch_size'}}
    )
    print(f"Exported to {filename}.onnx")


def build_onnx_manually(model: PPO, filename: str):
    policy_layer0_weight = model.policy.mlp_extractor.policy_net[0].weight.detach().numpy()
    policy_layer0_bias = model.policy.mlp_extractor.policy_net[0].bias.detach().numpy()
    policy_layer2_weight = model.policy.mlp_extractor.policy_net[2].weight.detach().numpy()
    policy_layer2_bias = model.policy.mlp_extractor.policy_net[2].bias.detach().numpy()
    action_net_weight = model.policy.action_net.weight.detach().numpy()
    action_net_bias = model.policy.action_net.bias.detach().numpy()

    # Define the ONNX model
    # Input: observation with 4 features
    X = helper.make_tensor_value_info('observation', TensorProto.FLOAT, [None, 4])
    # Output: action
    Y = helper.make_tensor_value_info('action', TensorProto.FLOAT, [None, 1])

    # Define the nodes (operations)
    # First linear layer: y = x * W^T + b
    policy_layer0_weight_initializer = numpy_helper.from_array(policy_layer0_weight.astype(np.float32),
                                                               name='policy_layer0_weight')
    policy_layer0_bias_initializer = numpy_helper.from_array(policy_layer0_bias.astype(np.float32),
                                                             name='policy_layer0_bias')
    policy_layer0 = helper.make_node(
        'Gemm',
        inputs=['observation', 'policy_layer0_weight', 'policy_layer0_bias'],
        outputs=['policy_layer0_output'],
        name='policy_layer0',
        alpha=1.0,
        beta=1.0,
        transB=1
    )

    # Tanh activation after first layer
    tanh1 = helper.make_node(
        'Tanh',
        inputs=['policy_layer0_output'],
        outputs=['tanh1_output'],
        name='tanh1'
    )

    # Second linear layer: y = x * W^T + b
    policy_layer2_weight_initializer = numpy_helper.from_array(policy_layer2_weight.astype(np.float32),
                                                               name='policy_layer2_weight')
    policy_layer2_bias_initializer = numpy_helper.from_array(policy_layer2_bias.astype(np.float32),
                                                             name='policy_layer2_bias')
    policy_layer2 = helper.make_node(
        'Gemm',
        inputs=['tanh1_output', 'policy_layer2_weight', 'policy_layer2_bias'],
        outputs=['policy_layer2_output'],
        name='policy_layer2',
        alpha=1.0,
        beta=1.0,
        transB=1
    )

    # Tanh activation after second layer
    tanh2 = helper.make_node(
        'Tanh',
        inputs=['policy_layer2_output'],
        outputs=['tanh2_output'],
        name='tanh2'
    )

    # Action network (final layer): y = x * W^T + b
    action_net_weight_initializer = numpy_helper.from_array(action_net_weight.astype(np.float32),
                                                            name='action_net_weight')
    action_net_bias_initializer = numpy_helper.from_array(action_net_bias.astype(np.float32), name='action_net_bias')
    action_net = helper.make_node(
        'Gemm',
        inputs=['tanh2_output', 'action_net_weight', 'action_net_bias'],
        outputs=['action'],
        name='action_net',
        alpha=1.0,
        beta=1.0,
        transB=1
    )

    # Create the graph
    graph = helper.make_graph(
        nodes=[policy_layer0, tanh1, policy_layer2, tanh2, action_net],
        name='PPO_Policy',
        inputs=[X],
        outputs=[Y],
        initializer=[
            policy_layer0_weight_initializer,
            policy_layer0_bias_initializer,
            policy_layer2_weight_initializer,
            policy_layer2_bias_initializer,
            action_net_weight_initializer,
            action_net_bias_initializer
        ]
    )

    # Create the model
    model_def = helper.make_model(graph, producer_name='sb3_ppo_exporter')
    model_def.opset_import[0].version = 11
    onnx.checker.check_model(model_def)
    print("ONNX model checking passed")
    onnx.save(model_def, filename)


class PolicyWithScaling(torch.nn.Module):
    def __init__(self, policy, action_low, action_high):
        super().__init__()
        self.policy = policy
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

    def forward(self, x):
        # Get features and latent representation
        features = self.policy.extract_features(x)
        latent_pi, _ = self.policy.mlp_extractor(features)

        # Get mean actions
        mean_actions = self.policy.action_net(latent_pi)

        normalized_actions = torch.tanh(mean_actions)

        scaled_actions = self.action_low + (0.5 * (normalized_actions + 1.0) * (self.action_high - self.action_low))

        return scaled_actions


def export_scaled_onnx(model: PPO, filename: str):
    action_low = model.action_space.low
    action_high = model.action_space.high

    print("Action space low:", action_low)
    print("Action space high:", action_high)

    scaled_model = PolicyWithScaling(model.policy, action_low, action_high)
    scaled_model.eval()

    dummy_input = th.rand(1, *model.observation_space.shape)
    torch.onnx.export(
        scaled_model,
        dummy_input,
        f"{filename}.onnx",
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["scaled_action"],
        dynamic_axes={"input": {0: "batch_size"}, "scaled_action": {0: "batch_size"}}
    )


if __name__ == "__main__":
    """
    How to run
        python -m fip_env.tools.export_model_to_onnx

    """

    filename = "fip_solver"
    model = PPO.load(f"{filename}.pth")

    export_scaled_onnx(model, filename)

    # build_onnx_manually(model, f"{filename}.onnx")

    # export_model_to_onnx(model, filename)

    x = np.random.rand(1, 4).astype(np.float32)
    print(f"Input: {x}")

    with torch.no_grad():
        torch_input = torch.FloatTensor(x)
        torch_action, _ = model.predict(torch_input, deterministic=True)

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(f"{filename}.onnx")
    print(ort_session.get_inputs()[0].name, ort_session.get_inputs()[0].shape, ort_session.get_inputs()[0].type)

    ort_inputs = {ort_session.get_inputs()[0].name: x}
    actions = ort_session.run(None, ort_inputs)

    print("PyTorch output:", torch_action)
    print("ONNX output: ", actions[0])
