# Supplementary material for ASPEN: Breaking Operator Barriers for Efficient Parallelization of Deep Neural Networks

This is supplementary material for NeuralIPS 2023 submission "ASPEN: Breaking Operator Barriers for Efficient Parallelization of Deep Neural Networks".

The included files and code provides three simple examples and evaluations of the ASPEN inference system:

1. Executing ResNet-50 Inference using ASPEN.
2. Executing batched muti-DNN co-inference of ResNet-50 and VGG-16 using ASPEN.
3. Migrating and executing a custom DNN from PyTorch to ASPEN.

The detailed instruction for each example is included in each directory, in the "instructions.txt" file.

ASPEN code must be compiled and run on x86 CPU with AVX2 support, using Ubuntu or other similar Linux distributions.

ASPEN has a dependency on OpenMP. The examples of this supplementary material have dependencies on PyTorch, TorchVision, and GCC.

To install requirements:
  ```install
  sudo apt install gcc python3 python3-pip
  pip3 install torch torchvision
  ```
> We plan to release the ASPEN source code, but to keep anonymity we included a pre-compiled library of ASPEN for this supplementary material.
