# caffe-tucker-decomposition
Caffe implementation of Tucker tensor decomposition for convolutional layers. Based on [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](https://arxiv.org/abs/1511.06530)

Example of usage can be found in `cnn_tucker_example.py`

For Caffe implementation of CP tensor decomposition for convolutional layers refer to [caffe-cp-decomposition](https://github.com/ddfabbro/cnn-cpd)

For PyTorch implementation of both CP and Tucker decomposition, refer to [PyTorch Tensor Decompositions](https://github.com/jacobgil/pytorch-tensor-decompositions)

## Requirements

- `pycaffe`
- `scikit-tensor`

## Limitations

- Convolutional layer paramaters that are non-uniform (e.g `kernel_h`, `kernel_w`, `pad_h`, `pad_w`, `stride_h` and `stride_w` ) are not supported. However, you can easily modify the code to your needs.
- Multi branch networks are not supported

## Additional resources

- [Accelerating deep neural networks with tensor decompositions](https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning)
- [Python implementation](https://github.com/CasvandenBogaard/VBMF) of Variational Bayes Matrix Factorization
