# Conv_LSTM(-RNN-GRU)
## References
Conv LSTM/GRU/RNN. See [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214). 

## download Moving MNIST
See this [link](http://www.cs.toronto.edu/~nitish/unsupervised_video/), or [download](http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy) directly.
```
cd ./data
wget http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
```

## Split datasets

```python
import numpy as np

n_train = 6000
n_val = 1000
n_test = 3000
n_total = n_train + n_val + n_test

data = np.load('./data/mnist_test_seq.npy')
indices = range(n_total)
np.random.shuffle(indices)

train = data[:, :n_train]
val = data[:, n_train:n_train+n_val]
test = data[:, n_train+n_val:n_total]

np.save('./data/mmnist_train.npy', train)
np.save('./data/mmnist_val.npy', val)
np.save('./data/mmnist_test.npy', test)
```

## Run
```
python main.py --hparams=MMNIST_CONV_LSTM
```

