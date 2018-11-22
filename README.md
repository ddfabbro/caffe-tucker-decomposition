# caffe-tucker-decomposition
Caffe implementation of Tucker tensor decomposition for convolutional layers, as described on [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](https://arxiv.org/abs/1511.06530).

For Caffe implementation of CP tensor decomposition for convolutional layers, refer to [caffe-cp-decomposition](https://github.com/ddfabbro/cnn-cpd).

For PyTorch implementation of both CP and Tucker decomposition, refer to [PyTorch Tensor Decompositions](https://github.com/jacobgil/pytorch-tensor-decompositions).

## Requirements

- `pycaffe`, which should include:
    - `numpy`
    - `scipy`
    - `protobuf`
- `scikit-tensor` (Does not support Python 3)

## Usage

Clone this respository and import `cnn_tucker` to a python script.

The model decomposition is implemented in `cnn_tucker.decompose_model(model_def_path, model_weights_path, layer_ranks)`, where:
- `model_def_path`: path to the `.prototxt` file.
- `model_weights_path`: path to the `.caffemodel` file.
- `layer_ranks`: dictionary of convolutional layers to be decomposed with its corresponding ranks.

The ranks can be chosen arbitrarily, but estimation of ranks via VBMF is provided as a utility function `cnn_tucker.util.estimate_ranks(weights)`, where:
- `weights`: tensor (`ndarray`) containing the weights of a convolutional layer.

Example of how these are used can be found in `cnn_tucker_example.py`.

## Fast getting started example: Tucker decomposition of VGG-16

You can easily get started with Tucker decomposition of layers **conv4_1**, **conv4_2** and **conv4_3** of VGG-16 by following these 3 steps:

1. Clone respository

```
git clone https://github.com/ddfabbro/caffe-tucker-decomposition.git
```

2. Add it to `PYTHONPATH`

```
export PYTHONPATH=$PYTHONPATH:$(pwd)/caffe-tucker-decomposition
```

3. Run Tucker decomposition. **NOTE:** VGG-16 caffe model will be downloaded.

```
python caffe-tucker-decomposition/cnn_tucker_example.py
```

Aside from downloading VGG-16 caffe model, this example should take less than 1 minute.

## Limitations

- Convolutional layer paramaters that are non-uniform (e.g `kernel_h`, `kernel_w`, `pad_h`, `pad_w`, `stride_h` and `stride_w` ) are not supported. However, you can easily modify the code to your needs.
- Multi branch networks are not supported.

## Additional resources

- [Accelerating deep neural networks with tensor decompositions](https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning).
- [Python implementation](https://github.com/CasvandenBogaard/VBMF) of Variational Bayes Matrix Factorization.
