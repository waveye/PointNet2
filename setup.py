from setuptools import find_packages, setup

setup(
    name="pointnet2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>2.2.0",
        "onnx==1.17.0",
        "onnxruntime==1.21.0",
        "onnxoptimizer==0.3.13",
        "onnxsim==0.4.36",
    ],
)
