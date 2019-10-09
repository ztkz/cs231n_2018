import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  batch_size = X.shape[0]
  for i in range(batch_size):
    logits = np.dot(X[i], W)
    logits -= logits.max()
    logits_exp = np.exp(logits)
    logits_exp_sum = logits_exp.sum()
    loss -= logits[y[i]] - np.log(logits_exp_sum)
    dlogits = logits_exp / logits_exp_sum
    dlogits[y[i]] -= 1
    dW += X[i][:, np.newaxis] * dlogits[np.newaxis, :]
  loss /= batch_size
  dW /= batch_size
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  batch_size = X.shape[0]
  logits = np.matmul(X, W)
  logits -= np.max(logits, axis=1, keepdims=True)
  logits_exp = np.exp(logits)
  logits_exp_sum = logits_exp.sum(axis=1)
  loss -= (logits[np.arange(batch_size), y] - np.log(logits_exp_sum)).sum()
  dlogits = logits_exp / logits_exp_sum[:, np.newaxis]
  dlogits[np.arange(batch_size), y] -= 1
  dW += np.matmul(X.T, dlogits)
  loss /= batch_size
  dW /= batch_size
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

