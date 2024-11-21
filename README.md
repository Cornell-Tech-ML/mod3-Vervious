# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


# Task 3.4

Fast ops vs GPU ops

<img src="images/output.png" width="50%">

```Timing summary
Size: 64
    fast: 0.00329
    gpu: 0.00599
Size: 128
    fast: 0.01575
    gpu: 0.01293
Size: 256
    fast: 0.10087
    gpu: 0.04873
Size: 512
    fast: 1.11760
    gpu: 0.18876
Size: 1024
    fast: 9.06136
    gpu: 0.87195
```

# Task 3.5

## Split dataset (GPU)

## Simple dataset (GPU)

## Xor dataset (GPU)