# models.py

import numpy as np

def param_free(cum_g=0,cum_var=1,h=1):
    """
    This function implement the descent in https://proceedings.mlr.press/v125/mhammedi20a/mhammedi20a.pdf 
    and ensure a parameter-free regret - Assumption 1.
    Args:
      - cum_g = cumulative gradients
      - cum_var = cumulative squared gradients
      - h = hints (bound G on gradients)
    """
    return - cum_g * (2 * cum_var + h * np.abs(cum_g))/(2 * (cum_var + h*np.abs(cum_g))**2 * np.sqrt(cum_var)) * np.exp(cum_g**2/(2*cum_var + 2 * h * np.abs(cum_g)))



class ChainingTree :

  """Chaining Tree model class"""

  def __init__(self, root=[[0,1]], depth=1, dim=1):
    #self.T = T
    self.depth = depth
    self.root = root
    self.dim = dim
    self.delta = [root[d][1]-root[d][0] for d in range(self.dim)]
    self.nodes = dict() 
    self.nodes[(0,)] = {'theta':0., 'visits':1, 'hint':1, 'cum_grad':0., 'cum_var':1.}

  def path(self, x):
    """
    Returns the nodes on each level containing data x.

    Args:
        x (array-like): Data point.

    Returns:
        list: Sequence of nodes along the path.
    """

    path = [(0,)]

    for k in range(1,self.depth):
      node = tuple([int((x[d] - self.root[d][0])* 2 ** k / self.delta[d]) % 2 for d in range(self.dim)])
      p = path[-1]+node
      if p not in self.nodes.keys():
        self.nodes[p] = {'theta':0., 'visits':1, 'hint':1, 'cum_grad':0., 'cum_var':1}
      path.append(p)

    return path

  def predict(self,x):
    """
    Returns the prediction of the model given the data x.

    Args:
        x (array-like): Data point.

    Returns:
        tuple: (list of nodes along the path, predicted value)
    """

    path = self.path(x)
    thetas = [self.nodes[node]['theta'] for node in path]
    return path, sum(thetas)

  def optimize(self, g, path, eta=1.0):
    """
    Optimizes nodes in the given path using gradient updates.

    Args:
        g (float or np.ndarray): Gradient value.
        path (list of tuples): List of nodes to be updated.
        eta (float, optional): Learning rate (default is 1.0).
    """
    for node in path:
        # Ensure node exists in the dictionary before updating
        self.nodes.setdefault(node, {'theta': 0., 'visits': 0, 'hint': 1, 'cum_grad': 0., 'cum_var': 1})

        # Update visit count and statistics
        self.nodes[node]['visits'] += 1
        self.nodes[node]['hint'] = max(self.nodes[node]['hint'], np.abs(g))
        self.nodes[node]['cum_grad'] += g
        self.nodes[node]['cum_var'] += g**2

        # Param-Free optimization update
        self.nodes[node]['theta'] = param_free(
            self.nodes[node]['cum_grad'],
            self.nodes[node]['cum_var'],
            self.nodes[node]['hint']
        )

  def train(self, X_train, y_train, loss='absolute', display=1000, custom_loss=None):
    """
    Train the ChainingTree model.

    Args:
        X_train (array-like): Covariates of shape (T, d) where d is the dimension.
        y_train (array-like): Targets of shape (T,).
        loss (str): Loss function ('absolute', 'squared', or 'custom').
        display (int): Frequency of logging progress.
        custom_loss (callable, optional): A function that takes (y_pred, y_true) and returns (loss_value, gradient).
    
    Returns:
        tuple: (sequence of loss values, max target magnitude B, max gradient magnitude G).
    """
    seq_loss, B, G = [], 0, 0

    # Define loss functions and their corresponding gradients
    loss_functions = {
        'absolute': lambda y_hat, y: (np.abs(y_hat - y), np.sign(y_hat - y)),
        'squared': lambda y_hat, y: ((y_hat - y) ** 2, 2 * (y_hat - y))
    }

    if loss == 'custom' and custom_loss is None:
        raise ValueError("If 'custom' loss is selected, a valid custom_loss function must be provided.")

    for t, (x, y_true) in enumerate(zip(X_train, y_train)):
        if t == 0:
            print('Start training Chaining Tree')

        if t % display == 0:
            print(f'\t Time: {t}')

        path, y_hat = self.predict(x)
        B = max(B, np.abs(y_true))

        # Compute loss and gradient
        if loss in loss_functions:
            loss_value, g = loss_functions[loss](y_hat, y_true)
        elif loss == 'custom':
            loss_value, g = custom_loss(y_hat, y_true)
        else:
            raise ValueError(f"Invalid loss function '{loss}'. Use 'absolute', 'squared', or provide a custom loss.")

        seq_loss.append(loss_value)
        G = max(G, np.abs(g))

        # Update model parameters
        self.optimize(g, path)

    return seq_loss, B, G


class Locally_Adaptive_Reg :
  
  """Locally Adaptive Online Regression model class"""

  def __init__(self, root=[[0,1]], depth_core=1, depth_chaining=1, dim=1):
    #self.T = T
    self.root = root
    self.dim = dim
    self.delta =[root[d][1] - root[d][0] for d in range(self.dim)]
    self.depth = depth_core
    self.depth_chaining = depth_chaining
    self.number_nodes = int((2**(self.dim*self.depth) - 1)/(2**self.dim - 1))
    self.nodes = dict()
    self.nodes[(0,)] = {'pred':ChainingTree(root=root, depth=depth_chaining, dim=dim), 'w':1./self.number_nodes, 'g':0., 'eta_inv_2':0., 'reg':0., 'cum_reg':0}
    self.sleep_tilde_g = 0.

  def path(self, x):
    """
    Returns the nodes on each level containing data x.

    Args:
        x (array-like): Data point.

    Returns:
        list: Sequence of nodes along the path.
    """

    path = [(0,)]

    for k in range(1,self.depth):
      m = [int((x[d] - self.root[d][0])* 2 ** k / self.delta[d]) for d in range(self.dim)]
      subroot = [[self.root[d][0]+ m[d] * 2 ** (-k) * self.delta[d], self.root[d][0]+ (m[d]+1) * 2 ** (-k) *self.delta[d]] for d in range(self.dim)]
      node = tuple([m[d] % 2 for d in range(self.dim)])
      p = path[-1]+node
      if p not in self.nodes.keys(): # set new node
        self.nodes[p] = {'pred':ChainingTree(root=subroot, depth=self.depth_chaining, dim=self.dim), 'w':1./self.number_nodes, 'g':0., 'eta_inv_2':0., 'reg':0., 'cum_reg':0}
      path.append(p)

    return path

  def predict(self,x):
    """
    Returns the prediction of the model given the data x.

    Args:
        x (array-like): Data point.

    Returns:
        tuple: (list of nodes along the path, predicted value)
    """

    path = self.path(x)
    preds, w_sleep = np.array([self.nodes[node]['pred'].predict(x)[1] for node in path]),np.array([self.nodes[node]['w'] for node in path])
    w_sleep /= np.sum(w_sleep)

    return path, np.sum(w_sleep * preds)

  def train(self, X_train, y_train, loss='absolute', display=1000, custom_loss=None):
    """
    Train the Locally_Adaptive_Reg model.

    Args:
        X_train (array-like): Covariates of shape (T, d), where T is the number of data points and d is the dimension.
        y_train (array-like): Target values of shape (T,).
        loss (str): Loss function ('absolute', 'squared', or 'custom').
        display (int): Frequency of logging progress.
        custom_loss (callable, optional): Function (y_pred, y_true) -> (loss_value, gradient) for custom losses.

    Returns:
        tuple: (sequence of loss values, max target magnitude B, max gradient magnitude G).
    """

    seq_loss, W, B, G, tilde_G = [], [], 0, 0, 0

    # Define loss functions and their gradients
    loss_functions = {
        'absolute': lambda y_hat, y: (np.abs(y_hat - y), np.sign(y_hat - y)),
        'squared': lambda y_hat, y: ((y_hat - y) ** 2, 2 * (y_hat - y))
    }

    if loss == 'custom' and custom_loss is None:
        raise ValueError("If 'custom' loss is selected, a valid custom_loss function must be provided.")

    for t, (x, y_true) in enumerate(zip(X_train, y_train)):
        if t == 0:
            print('Start training Locally Adaptive Online Reg')
        if t % display == 0:
            print(f'\t Time: {t}')

        # Make prediction
        path, y_hat = self.predict(x)
        B = max(B, np.abs(y_true))

        # Compute loss and gradient
        if loss in loss_functions:
            loss_value, g = loss_functions[loss](y_hat, y_true)
        elif loss == 'custom':
            loss_value, g = custom_loss(y_hat, y_true)
        else:
            raise ValueError(f"Invalid loss function '{loss}'. Use 'absolute', 'squared', or provide a custom loss.")

        seq_loss.append(loss_value)

        # Discover gradients for core tree
        active_weights, active_grads = [], []
        for node in path:
            node_data = self.nodes[node]
            y_hat_n = node_data['pred'].predict(x)[1]
            tilde_g = g * y_hat_n  # Gradient scaling

            node_data['g'] = tilde_g
            active_weights.append(node_data['w'])
            active_grads.append(tilde_g)

            tilde_G = max(tilde_G, tilde_g)

        # Sleeping nodes update
        self.sleep_tilde_g = g * y_hat
        tilde_G = max(self.sleep_tilde_g, tilde_G)

        # Compute weighted gradient
        wg = np.dot(active_weights, active_grads)  # Active nodes
        sleeping_nodes = set(self.nodes.keys()) - set(path)
        sleeping_weights = [self.nodes[node]['w'] for node in sleeping_nodes]
        wg += np.sum(sleeping_weights) * self.sleep_tilde_g  # Sleeping nodes

        # Compute regret and update weights
        R_aux, nodes_to_opt = [], []
        for node in self.nodes.keys():
            node_data = self.nodes[node]
            if node in path:
                r = wg - node_data['g']
                node_data['eta_inv_2'] += 2.2 * r ** 2
                if node_data['eta_inv_2'] > 0:
                    nodes_to_opt.append(node)
                    node_data['reg'] = r - r**2 / np.sqrt(node_data['eta_inv_2'])
                    node_data['cum_reg'] += node_data['reg']
                    R_aux.append(-np.log(node_data['eta_inv_2']) / 2 + np.log(1 / self.number_nodes) + node_data['reg'] / np.sqrt(node_data['eta_inv_2']))
            else:
                node_data['cum_reg'] += node_data['reg']

        if R_aux:
            max_R_aux = np.max(R_aux)
            exp_R = np.exp(R_aux - max_R_aux)
            normalization_factor = np.sum(exp_R)
            for node in nodes_to_opt:
                self.nodes[node]['w'] = len(nodes_to_opt) / self.number_nodes * exp_R[nodes_to_opt.index(node)] / normalization_factor

        # Update chaining tree
        for n in path:
            fn = self.nodes[n]['pred']
            path_n, y_hat_n = fn.predict(x)

            gn = np.sign(y_hat_n - y_true) if loss == 'absolute' else 2 * (y_hat_n - y_true)

            # Fast learning of the root (online mean) with square loss
            if loss == 'squared':
                fn.nodes[(0,)]['visits'] += 1
                visits = fn.nodes[(0,)]['visits']
                fn.nodes[(0,)]['theta'] = (visits - 1) / visits * fn.nodes[(0,)]['theta'] + y_true / visits

            G = max(G, gn)
            learn_root = int(loss != 'absolute')  # 1 if 'squared', 0 if 'absolute'
            fn.optimize(gn, path_n[learn_root:])

    return seq_loss, B, max(tilde_G, G)
