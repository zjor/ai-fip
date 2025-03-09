import numpy as np
import torch as th
import torch.onnx
from stable_baselines3 import PPO
from torch import nn


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class ActorCriticWrapper(nn.Module):
    def __init__(self, policy):
        super(ActorCriticWrapper, self).__init__()
        self.policy = policy

    def forward(self, x):
        action_dist = self.policy.get_distribution(x)
        action = action_dist.get_actions()
        value = self.policy.predict_values(x)
        return action, value


def export_model_to_onnx(model: PPO, filename: str):
    wrapped_model = ActorCriticWrapper(model.policy)
    wrapped_model.eval()

    dummy_input = th.rand(1, *model.observation_space.shape)

    torch.onnx.export(
        wrapped_model,
        dummy_input, f"{filename}.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['action_mean', 'value'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # Allow dynamic batch size
            'action_mean': {0: 'batch_size'},  # Allow dynamic batch size
            'value': {0: 'batch_size'}  # Allow dynamic batch size
        }
    )
    print(f"Exported to {filename}.onnx")


def main(filename: str = "fip_solver"):
    model = PPO.load(f"{filename}.pth")
    actor = model.policy.mlp_extractor.policy_net
    actor.eval()

    batch_size = 64
    x = torch.randn(batch_size, 4, requires_grad=True)
    torch_out = actor(x)

    torch.onnx.export(actor,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      f"{filename}.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    print("Exported model to ONNX")


def load_and_check_model(filename: str = "fip_solver"):
    import onnx
    onnx_model = onnx.load(f"{filename}.onnx")
    onnx.checker.check_model(onnx_model)


def test_onnx_output(filename: str = "fip_solver"):
    import time
    import onnxruntime

    model = PPO.load(f"{filename}.pth")
    actor = model.policy.mlp_extractor.policy_net
    actor.eval()

    batch_size = 64
    x = torch.randn(batch_size, 4, requires_grad=True)
    torch_out = actor(x)

    ort_session = onnxruntime.InferenceSession(f"{filename}.onnx", providers=["CPUExecutionProvider"])

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Torch output", to_numpy(torch_out)[0])
    print("ONNX output", ort_outs[0][0])

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # checking time performance
    start = time.time()
    actor(x)  # torch run
    end = time.time()
    print(f"Inference of Pytorch model used {end - start} seconds")

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    start = time.time()
    ort_session.run(None, ort_inputs)  # onnx run
    end = time.time()
    print(f"Inference of ONNX model used {end - start} seconds")


if __name__ == "__main__":
    """
    How to run
        python -m fip_env.tools.export_model_to_onnx

    """

    # main()
    # load_and_check_model()
    # test_onnx_output()
    filename = "fip_solver"
    model = PPO.load(f"{filename}.pth")
    export_model_to_onnx(model, filename)

    x = th.randn(1, 4, requires_grad=True)
    print(x)
    action, states = model.predict(to_numpy(x), deterministic=True)
    print(action, states)

    import onnxruntime
    ort_session = onnxruntime.InferenceSession(f"{filename}.onnx", providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print("Torch output", ort_outs)

