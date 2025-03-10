from pathlib import Path

import onnx
import onnxoptimizer
import onnxsim
import torch
from onnxruntime.tools import symbolic_shape_infer


def get_model(name: str, *args, **kwargs):
    import importlib
    module = importlib.import_module(f'models.{name}')
    return module.get_model(*args, **kwargs)


def export_model(model, path: Path, infer_shapes=True, infer_shapes_symbolic=True, optimize=True, simplify=True):
    # Export Torch model with dummy input
    input_dummy = torch.randn((16, 1024, 6), dtype=torch.float32),
    torch.onnx.export(
        model,
        input_dummy,
        path,
        opset_version=13,
        # ONNX nodes contain docstrings mapping back to Torch calls
        verbose=True,
        # Statically provide weights
        keep_initializers_as_inputs=False,
    )

    # Load back from ONNX file
    onnx_model = onnx.load(path)
    # Static shape inference
    if infer_shapes:
        onnx_model = onnx.shape_inference.infer_shapes(
            onnx_model, strict_mode=True)
    # Symbolic shape inference
    if infer_shapes_symbolic:
        onnx_model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(
            onnx_model, int_max=2 ** 31 - 1, auto_merge=True, guess_output_rank=True, verbose=True)
    # Optimize graph
    if optimize:
        onnx_model = onnxoptimizer.optimize(onnx_model)
    # Simplify graph
    if simplify:
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'Simplified model could not be validated'

    # Write summary as text file
    with open(path.with_suffix('.graph.txt'), 'w') as f:
        f.write(onnx.helper.printable_graph(onnx_model.graph))
    # Write model to file
    onnx.save(onnx_model, path)


def cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('-o', '--output', type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(args.name).with_suffix('.onnx')

    model = get_model(args.name, num_class=6)
    export_model(model.eval(), args.output)


if __name__ == '__main__':
    cli()
