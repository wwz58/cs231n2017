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
  N, D = X.shape
  _, C = W.shape
  for i in range(N):
    score = X[i].dot(W)
    scores = score - np.max(score)
    p = np.exp(scores) / np.sum(np.exp(scores))
    dscore = p 
    loss -= np.log(p[y[i]])
    dscore = p
    dscore[y[i]] -= 1
    dW += X[i].reshape(-1,1).dot(dscore.reshape(1,-1))
  loss /= N
  loss += reg*np.sum(W**2)/2
  dW /= N
  dW += reg*W
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
  N, D = X.shape
  _, C = W.shape
  scores = X.dot(W)
  scores -= np.max(scores)
  p = np.exp(scores)/np.sum(np.exp(scores),axis=1,keepdims=True)
  loss = np.sum(-np.log(p[range(N),y]))/N+ reg*0.5*np.sum(W**2)
  dscores = p 
  dscores[range(N),y] -= 1
  dW = X.T.dot(dscores)/N
  dW += reg*W   
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

            