import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File('eye_data.h5', 'r') as h5f:
    train_x = h5f['train/x_small'][:]
    train_y = h5f['train/y'][:]

    validation_x = h5f['validation/x_small'][:]
    validation_y = h5f['validation/y'][:]

    test_x = h5f['test/x_small'][:]
    test_y = h5f['test/y'][:]


def angular_error(X, y):
    """Calculate angular error (via cosine similarity)."""

    def pitchyaw_to_vector(pitchyaws):
        """Convert given pitch and yaw angles to unit gaze vectors."""
        n = pitchyaws.shape[0]
        sin = np.sin(pitchyaws)
        cos = np.cos(pitchyaws)
        out = np.empty((n, 3))
        out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
        out[:, 1] = sin[:, 0]
        out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
        return out

    a = pitchyaw_to_vector(y)
    b = pitchyaw_to_vector(X)

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * (180.0 / np.pi)


def predict_and_calculate_mean_error(nn, x, y):
    """Calculate mean error of neural network predictions on given data."""
    n, _, _ = x.shape
    predictions = nn.predict(x.reshape(n, -1)).reshape(-1, 2)
    labels = y.reshape(-1, 2)
    errors = angular_error(predictions, labels)
    return np.mean(errors)


def predict_and_visualize(nn, x, y):
    """Visualize errors of neural network on given data."""

    nr, nc = 1, 12
    n = nr * nc
    fig = plt.figure(figsize=(12, 2.))
    predictions = nn.predict(x[:n, :].reshape(n, -1))
    for i, (image, label, prediction) in enumerate(zip(x[:n], y[:n], predictions)):
        plt.subplot(nr, nc, i + 1)
        plt.imshow(image, cmap='gray')
        error = angular_error(prediction.reshape(1, 2), label.reshape(1, 2))
        plt.title('%.1f' % error, color='g' if error < 7.0 else 'r')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout(pad=0.0)
    plt.show()


def ReLU(x):
    """Computes the Rectified Linear Unit function."""
    x[x <= 0] = 0
    return x


def ReLU_(x):
    """Computes the derivative of the ReLU function."""
    return (x > 0).astype(np.float32)


""" The ReLU and its derivatives look like this"""

x = np.linspace(-2., 2., num=400)
relu = ReLU(x)
relu_prime = ReLU_(relu)

plt.figure(figsize=(8, 6))
plt.plot(x, relu, label="ReLU")
plt.plot(x, relu_prime, label="ReLU prime")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([-2, 2])
plt.ylim([-1, 2])
plt.legend(prop={'size': 16})
plt.show()


def MSE(Y, YH):
    """Compute elementwise mean square error between two matrices."""
    residual = Y - YH
    return np.mean(np.square(residual))


class NNRegressor:

    def __init__(self, n_outputs, n_features, n_hidden_units=30,
                 l2_reg=0.0, epochs=500, learning_rate=0.01,
                 batch_size=10, random_seed=None):

        if random_seed:
            np.random.seed(random_seed)
        self.n_outputs = n_outputs
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.w1, self.w2 = self._init_weights()
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def _init_weights(self):
        # Truncated normal for weights initialization
        w1 = np.random.normal(0.0, 0.01,
                              size=self.n_hidden_units * (self.n_features + 1))
        w1 = np.clip(w1, -0.01, 0.01)
        w1 = w1.reshape(self.n_hidden_units, self.n_features + 1)
        w1[:, 0] = 1e-5  # Constant bias initialization
        w2 = np.random.normal(0.0, 0.01,
                              size=self.n_outputs * (self.n_hidden_units + 1))
        w2 = np.clip(w2, -0.01, 0.01)
        w2 = w2.reshape(self.n_outputs, self.n_hidden_units + 1)
        w2[:, 0] = 1e-5  # Constant bias initialization
        return w1, w2

    def _add_bias_unit(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        return X_new

    def _error(self, y, output):
        return MSE(y, output)

    def _backprop_step(self, X, y):
        net_hidden, act_hidden, net_out = self._forward(X)
        grad1, grad2 = self._backward(X, net_hidden, act_hidden, net_out, y)

        # regularize
        grad1[:, 1:] += 2 * self.l2_reg * self.w1[:, 1:]
        grad2[:, 1:] += 2 * self.l2_reg * self.w2[:, 1:]

        error = self._error(y, net_out)
        return error, grad1, grad2

    def predict(self, X):
        net_hidden, act_hidden, net_out = self._forward(X)
        return net_out

    def fit(self, X_train, y_train, X_test, y_test):
        training_errors = []
        testing_errors = []

        self.error_ = []
        for i in range(self.epochs):
            n_batches = int(X_train.shape[0] / self.batch_size)
            X_mb = np.array_split(X_train, n_batches)
            y_mb = np.array_split(y_train, n_batches)

            epoch_errors = []

            for Xi, yi in zip(X_mb, y_mb):
                batch_size = Xi.shape[0]

                # update weights
                error, grad1, grad2 = self._backprop_step(Xi.reshape(batch_size, -1), yi)
                epoch_errors.append(error)
                self.w1 -= (self.learning_rate * grad1)
                self.w2 -= (self.learning_rate * grad2)
            mean_epoch_errors = np.mean(epoch_errors)
            self.error_.append(mean_epoch_errors)

            # Evaluate errors and visualize progress
            if i % 5 == 0:
                batch_train_error = predict_and_calculate_mean_error(self, Xi, yi)
                training_errors.append([i + 1, batch_train_error])
                if i % 10 == 0:
                    mean_test_error = predict_and_calculate_mean_error(self, X_test, y_test)
                    testing_errors.append([i + 1, mean_test_error])
                    print('Epoch %d> mean test error: %f degrees' % (i + 1, mean_test_error))
                    predict_and_visualize(self, X_test, y_test)

        # Now plot a graph of error progression
        training_errors = np.asarray(training_errors)
        testing_errors = np.asarray(testing_errors)
        plt.plot(training_errors[:, 0], training_errors[:, 1], 'g-*', label='train')
        plt.plot(testing_errors[:, 0], testing_errors[:, 1], 'b-*', label='test')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Angular Gaze Error')

        return self


def nn_forward_pass(self, X):
    """Perform a forward pass of input data through this neural network.

    Note: The output of every step of this forward pass must be cached to be
          used for the backward-pass. The backward-pass updates the neural
          network weight and bias parameters.

    Params:
        X: input data of shape (N x F)

    Neural Network Weights:
        self.w1: weight parameters of shape (H x F+1)
        self.w2: weight parameters of shape (O x H+1)

    Legend:
        N: Number of input data entries
        F: Number of features
        H: Number of neurons in hidden layer
        O: Number of output values
    """
    ## First step: ReLU(X * W1)
    # Adjust input data to be of shape (N x F+1)
    net_input_padded = self._add_bias_unit(X)

    # Calculate hidden layer output of shape (N x H)
    net_hidden = net_input_padded.dot(self.w1.T)

    # Calculate hidden layer activations of shape (N x H)
    act_hidden = ReLU(net_hidden)

    ## Second step: X * W2
    # Adjust activations to be of shape (N x H+1)
    act_hidden_padded = self._add_bias_unit(act_hidden)

    # Calculate neural network output of shape (N x O)
    net_out = act_hidden_padded.dot(self.w2.T)

    return net_hidden, act_hidden, net_out


NNRegressor._forward = nn_forward_pass


def nn_gradient_calculations(self, net_input, net_hidden, act_hidden, net_out, y):
    """Calculate gradients for a backward pass through this neural network.

    Params:
        net_input: input data of shape (N x F)
        net_hidden: output of hidden layer (N x H)
        act_hidden: activations of hidden layer (N x H)
        net_out: output of neural network (N x O)
        y: ground-truth labels (N x O)

    Neural Network Weights:
        self.w1: weight parameters of shape (H x F+1)
        self.w2: weight parameters of shape (O x H+1)

    Legend:
        N: Number of input data entries
        F: Number of features
        H: Number of neurons in hidden layer
        O: Number of output values
    """

    # Calculate error residual (N x O)
    de_do = 2 * (net_out - y)

    # ---#

    # Calculate derivative of output w.r.t w2 (N x H+1)
    do_dw2 = self._add_bias_unit(act_hidden)

    # Calculate gradient w.r.t self.w2 (O x H+1)
    de_dw2 = de_do.T.dot(do_dw2)

    # ---#

    # Calculate derivative of output w.r.t hidden layer activations (O x H+1)
    # Remember: o = a * w2
    do_da = self.w2

    # Calculate derivative of hidden layer activations w.r.t hidden layer output (N x H+1)
    # Remember: a = ReLU(h) but with bias-padding
    da_dh = ReLU_(self._add_bias_unit(net_hidden))

    # Calculate derivative of hidden layer output w.r.t w1 (N x F+1)
    # Remember: h = x * w1
    dh_dw1 = self._add_bias_unit(net_input)

    # Calculate gradient w.r.t self.w1 (H x F+1)
    de_dw1 = (de_do.dot(do_da) * da_dh).T.dot(dh_dw1)[1:, :]

    return de_dw1, de_dw2


NNRegressor._backward = nn_gradient_calculations

# A neural network should be trained until the training and test
# errors plateau, that is, they do not improve any more.
epochs = 201

# Having more neurons in a network allows for more complex
# mappings to be learned between input data and expected outputs.
# However, defining the function to be too complex can lead to
# overfitting, that is, any function can be learned to memorize
# training data.
n_hidden_units = 64

# Lower batch sizes can cause noisy training error progression,
# but sometimes lead to better generalization (less overfitting
# to training data)
batch_size = 24

# A higher learning rate makes training faster, but can cause
# overfitting
learning_rate = 0.0005

# Increase to reduce over-fitting effects
l2_regularization_coefficient = 0.0001

N_FEATURES = len(train_x[0, :].flatten())
N_OUTPUTS = train_y.shape[1]

nn = NNRegressor(n_outputs=N_OUTPUTS,
                 n_features=N_FEATURES,
                 n_hidden_units=n_hidden_units,
                 l2_reg=l2_regularization_coefficient,
                 epochs=epochs,
                 learning_rate=learning_rate,
                 batch_size=batch_size,
                 random_seed=42)

nn.fit(train_x, train_y, test_x, test_y)
