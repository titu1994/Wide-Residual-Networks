# Wide Residual Networks in Keras

Implementation of Wide Residual Networks from the paper <a href="http://arxiv.org/pdf/1605.07146v1.pdf">Wide Residual Networks</a> in Keras.

## Usage

It can be used by importing the wide_residial_network script and using the create_wide_residual_network() method.
There are several parameters which can be changed to increase the depth or width of the network.

Note that the number of layers can be calculated by the formula : `nb_layers = 4 + 6 * N` <br>
Therefore N can be computed as : `N = (nb_layers - 4) / 6`

`import wide_residial_network as wrn`<br>
`ip = Input(shape=(3, 32, 32)) # For CIFAR 10`

`wrn_28_10 = wrn.create_wide_residual_network(ip, nb_classes=10, N=4, k=10, dropout=0.0, verbose=1)`

`model = Model(ip, wrn_28_10)`

## Testing
### WRN-16-8
The WRN-16-8 model has been tested on the CIFAR 10 dataset. It achieves a score of 93.68% after 100 epochs. It is not as high as the accuracy posted in the paper (95.19%), however the score may improve with further training. 

Training was done by using the Adam optimizer instead of SGD+Momentum for faster convergence. The history of training/validation accuracy and loss is not available for the first 30 epochs due to an overwriting of the files. However the history of the last 70 epochs has been shown in the figure below. The script and weights for this model are also provided.

<img src="https://raw.githubusercontent.com/titu1994/Wide-Residual-Networks/master/plots/Validation curves.png" height=100% width=100%>

### WRN-28-8
The WRN-28-10 model could not be used due to GPU memory constraints, hence WRN-28-8 model was used instead with a batch size of 64. Each epoch requires roughly 886 seconds, and therefore this was only run for 100 epochs. It achieves a score of 95.08 %, less than the best score of 95.83 % obtained by the WRN-28-10 network.

The Adadelta optimizer was used instead of SGD+Momentum for faster convergence. The history of training/validation accuracy and loss is shown as below. The script and weights for this model are also provided.

<img src="https://github.com/titu1994/Wide-Residual-Networks/blob/master/plots/WRN_28_8%20validation%20curves.png?raw=true" height=100% width=100%>

## Models
The below model is the WRN-28-8 model.

<img src="https://raw.githubusercontent.com/titu1994/Wide-Residual-Networks/master/plots/WRN-28-8.png" height=100% width=100%>
