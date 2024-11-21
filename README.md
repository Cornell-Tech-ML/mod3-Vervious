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
```
Epoch  0  loss  7.464073762413637 correct 36 | avgtime 4.068
Epoch  10  loss  5.536555825528271 correct 43 | avgtime 1.865
Epoch  20  loss  4.774814833569234 correct 47 | avgtime 1.787
Epoch  30  loss  3.8176758058738 correct 44 | avgtime 1.719
Epoch  40  loss  2.9218034894494864 correct 46 | avgtime 1.686
Epoch  50  loss  2.475222529670729 correct 49 | avgtime 1.677
Epoch  60  loss  3.245775696642008 correct 46 | avgtime 1.661
Epoch  70  loss  1.6787209530132712 correct 46 | avgtime 1.648
Epoch  80  loss  2.276273825738415 correct 47 | avgtime 1.648
Epoch  90  loss  1.0007776050356716 correct 48 | avgtime 1.641
Epoch  100  loss  1.6410498466692882 correct 48 | avgtime 1.634
Epoch  110  loss  0.9508725542146957 correct 49 | avgtime 1.632
Epoch  120  loss  0.5797632004892914 correct 49 | avgtime 1.629
Epoch  130  loss  1.5882832943870908 correct 49 | avgtime 1.626
Epoch  140  loss  0.8321662589123356 correct 49 | avgtime 1.622
Epoch  150  loss  1.368811896760549 correct 49 | avgtime 1.629
Epoch  160  loss  1.400348545784629 correct 50 | avgtime 1.625
Epoch  170  loss  2.588119824015272 correct 49 | avgtime 1.622
Epoch  180  loss  1.1637917898002055 correct 49 | avgtime 1.623
Epoch  190  loss  0.7531781036566642 correct 49 | avgtime 1.621
Epoch  200  loss  1.4269170157060298 correct 49 | avgtime 1.618
Epoch  210  loss  0.6253528259471341 correct 49 | avgtime 1.617
Epoch  220  loss  2.132025693356082 correct 48 | avgtime 1.617
Epoch  230  loss  0.5295008517777808 correct 50 | avgtime 1.614
Epoch  240  loss  0.7495446833219394 correct 49 | avgtime 1.612
Epoch  250  loss  0.5772737508157826 correct 49 | avgtime 1.613
Epoch  260  loss  0.3232378682476648 correct 49 | avgtime 1.612
Epoch  270  loss  1.9320733232214835 correct 49 | avgtime 1.610
Epoch  280  loss  0.4368446426728211 correct 50 | avgtime 1.614
Epoch  290  loss  2.1570271089545066 correct 49 | avgtime 1.613
Epoch  300  loss  1.217703041063809 correct 49 | avgtime 1.611
Epoch  310  loss  0.10507894503625358 correct 50 | avgtime 1.610
Epoch  320  loss  0.41581565072377646 correct 50 | avgtime 1.610
Epoch  330  loss  0.15122259431158358 correct 49 | avgtime 1.609
Epoch  340  loss  0.7119654758750691 correct 49 | avgtime 1.608
Epoch  350  loss  0.05698788457019405 correct 49 | avgtime 1.609
Epoch  360  loss  0.32745715360620054 correct 49 | avgtime 1.607
Epoch  370  loss  0.2726870050517636 correct 49 | avgtime 1.606
Epoch  380  loss  0.4501957616896136 correct 50 | avgtime 1.606
Epoch  390  loss  0.14714890522417204 correct 49 | avgtime 1.606
Epoch  400  loss  0.7234296888670858 correct 49 | avgtime 1.605
Epoch  410  loss  0.3594594629588038 correct 48 | avgtime 1.606
Epoch  420  loss  1.3954968352491783 correct 49 | avgtime 1.607
Epoch  430  loss  1.1948079665303868 correct 49 | avgtime 1.606
Epoch  440  loss  1.1415222506845761 correct 49 | avgtime 1.605
Epoch  450  loss  0.30690778494167087 correct 49 | avgtime 1.606
Epoch  460  loss  0.18615760295839998 correct 49 | avgtime 1.605
Epoch  470  loss  1.330428429328573 correct 49 | avgtime 1.605
Epoch  480  loss  1.0865034255977042 correct 49 | avgtime 1.604
Epoch  490  loss  1.295505819559996 correct 49 | avgtime 1.605
```

## Simple dataset (GPU)

## Xor dataset (GPU)