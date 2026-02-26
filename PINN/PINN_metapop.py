# -*- coding: utf-8 -*-
# PINN for Metapopulation SIR+D with unknown movement (piecewise-constant)
import os, time, math
import numpy as np
import tensorflow as tf
from scipy.special import jacobi


# -----------------------
# Utilities
# -----------------------
def make_bin_gating(t_f, B, lb, ub):
    """
    Build a one-hot gating matrix G_f for piecewise-constant time bins on [lb,ub].
    t_f : (M+1,1) time grid used for residuals
    B   : number of time bins
    returns:
      G_f   : (M+1, B) one-hot row that selects the bin for each t_f[k]
      edges : (B+1,) bin edges
    """
    lb_ = float(lb); ub_ = float(ub)
    edges = np.linspace(lb_, ub_, B+1)
    tt = t_f.reshape(-1)
    idx = np.searchsorted(edges, tt, side='right') - 1
    idx = np.clip(idx, 0, B-1)
    G = np.zeros((len(tt), B), dtype=np.float64)
    G[np.arange(len(tt)), idx] = 1.0
    return G, edges

def offdiag_mask(P):
    m = np.ones((P,P), dtype=np.float64)
    np.fill_diagonal(m, 0.0)
    return m

# -----------------------
# PINN class
# -----------------------
class PhysicsInformedNN_Metapop:
    """
    Metapopulation fractional SIR+D PINN with unknown movement M(t).
    - Compartments per patch: [S,I,R,D]; D never moves.
    - Movement M(t) is piecewise-constant in B bins, learned from data.
    - Beta(t) is learned (scalar, shared across patches).
    - Kappa_1..4 shared across patches (like your original).
    """

    def __init__(self, P, N_vec,
                 t_f, t_train,
                 I_train, R_train, D_train,
                 U0, lb, ub,
                 layers, layers_Beta,
                 B_bins=1,
                 lambda_M=1e-4,
                 seed=1234):

        """
        Args
        ----
        P        : number of patches
        N_vec    : (P,) or (1,P) array of populations per patch (scaled same as data)
        t_f      : (M+1,1) residual grid (uniform)
        t_train  : (T,1) observation times
        I_train  : (T,P) Infectious series per patch
        R_train  : (T,P) Recovered series per patch
        D_train  : (T,P) Death series per patch
        U0       : [S0, I0, R0, D0], each (1,P)
        lb, ub   : scalars for time normalization in the nets (like original)
        layers   : e.g. [1] + 5*[20] + [4*P]
        layers_Beta : e.g. [1] + 5*[20] + [1]
        B_bins   : number of time bins for piecewise-constant M(t)
        lambda_M : L1 regularization on movement rates (keeps rates modest)
        """

        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.P = P
        self.N_vec = np.asarray(N_vec, dtype=np.float64).reshape(1, P)
        self.B_bins = int(B_bins)
        self.lambda_M = float(lambda_M)

        # Data
        self.t_f = np.asarray(t_f, dtype=np.float64)        # (M+1,1)
        self.t_train = np.asarray(t_train, dtype=np.float64)  # (T,1)
        self.I_train = np.asarray(I_train, dtype=np.float64)  # (T,P)
        self.R_train = np.asarray(R_train, dtype=np.float64)  # (T,P)
        self.D_train = np.asarray(D_train, dtype=np.float64)  # (T,P)

        self.S0 = np.asarray(U0[0], dtype=np.float64).reshape(1,P)
        self.I0 = np.asarray(U0[1], dtype=np.float64).reshape(1,P)
        self.R0 = np.asarray(U0[2], dtype=np.float64).reshape(1,P)
        self.D0 = np.asarray(U0[3], dtype=np.float64).reshape(1,P)

        # Time discretization for fractional derivative
        self.M = len(self.t_f) - 1
        self.tau = self.t_f[1,0] - self.t_f[0,0]

        # Bounds
        self.lb = lb
        self.ub = ub

        # TF session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))

        # Placeholders
        self.t_tf = tf.placeholder(tf.float64, shape=[None, 1])     # residual grid
        self.t_u  = tf.placeholder(tf.float64, shape=[None, 1])     # observed times
        self.I_u  = tf.placeholder(tf.float64, shape=[None, self.P])
        self.R_u  = tf.placeholder(tf.float64, shape=[None, self.P])
        self.D_u  = tf.placeholder(tf.float64, shape=[None, self.P])

        self.S0_u = tf.placeholder(tf.float64, shape=[None, self.P])  # (1,P)
        self.I0_u = tf.placeholder(tf.float64, shape=[None, self.P])
        self.R0_u = tf.placeholder(tf.float64, shape=[None, self.P])
        self.D0_u = tf.placeholder(tf.float64, shape=[None, self.P])

        # Parameters a (recovery) and b (death) — fixed as in your original
        self.a = tf.Variable(2.15e-2, dtype=tf.float64, trainable=False)
        self.b = tf.Variable(0.48e-2, dtype=tf.float64, trainable=False)

        # Population per patch (1,P)
        self.N_tf = tf.constant(self.N_vec, dtype=tf.float64)

        # Networks
        self.weights, self.biases = self.initialize_NN(layers)
        self.weights_Beta, self.biases_Beta = self.initialize_NN(layers_Beta)

        # Fractional orders (shared across patches)
        self.Kappa1_COEF = tf.Variable(tf.zeros([poly_order,1], dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.Kappa2_COEF = tf.Variable(tf.zeros([poly_order,1], dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.Kappa3_COEF = tf.Variable(tf.zeros([poly_order,1], dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.Kappa4_COEF = tf.Variable(tf.zeros([poly_order,1], dtype=tf.float64), dtype=tf.float64, trainable=True)

        self.Kappa_pred1 = self.net_Kappa_plot(self.Kappa1_COEF)
        self.Kappa_pred2 = self.net_Kappa_plot(self.Kappa2_COEF)
        self.Kappa_pred3 = self.net_Kappa_plot(self.Kappa3_COEF)
        self.Kappa_pred4 = self.net_Kappa_plot(self.Kappa4_COEF)

        # Movement parameterization: piecewise-constant in B bins, nonnegative, zero diag
        mask_np = offdiag_mask(self.P)
        self.mask_offdiag = tf.constant(mask_np, dtype=tf.float64)
        self.M_raw = tf.Variable(tf.random_normal([self.B_bins, self.P, self.P],
                                                  stddev=1e-3, dtype=tf.float64), trainable=True)
        self.M_bins = tf.nn.softplus(self.M_raw) * self.mask_offdiag   # (B,P,P)

        # Build gating for the residual grid
        G_f_np, self.bin_edges = make_bin_gating(self.t_f, self.B_bins, float(lb), float(ub))
        self.G_f = tf.constant(G_f_np, dtype=tf.float64)               # (M+1,B)
        # Per-time movement matrices on residual grid: (M+1, P, P)
        self.M_seq_f = tf.tensordot(self.G_f, self.M_bins, axes=[[1],[0]])

        # Predictions on observed times
        self.S_pred, self.I_pred, self.R_pred, self.D_pred = self.net_u(self.t_u)  # (T,P) each
        self.BetaI = self.net_Beta(self.t_u)                                       # (T,1)

        self.S0_pred = self.S_pred[0:1, :]   # (1,P)
        self.I0_pred = self.I_pred[0:1, :]
        self.R0_pred = self.R_pred[0:1, :]
        self.D0_pred = self.D_pred[0:1, :]

        # Residuals on t_f grid (fractional dynamic + movement)
        self.f_S, self.f_I, self.f_R, self.f_D, self.f_con = self.net_f(self.t_tf)

        # Loss terms
        self.lossU0 = tf.reduce_mean(tf.square(self.I0_u - self.I0_pred)) \
                    + tf.reduce_mean(tf.square(self.R0_u - self.R0_pred)) \
                    + tf.reduce_mean(tf.square(self.D0_u - self.D0_pred))
        self.lossU = 8.0*tf.reduce_mean(tf.square(self.I_pred - self.I_u)) \
                   + 1.0*tf.reduce_mean(tf.square(self.R_pred - self.R_u)) \
                   + 60.0*tf.reduce_mean(tf.square(self.D_pred - self.D_u))
        self.lossF = tf.reduce_mean(tf.square(self.f_S)) \
                   + tf.reduce_mean(tf.square(self.f_I)) \
                   + tf.reduce_mean(tf.square(self.f_R)) \
                   + tf.reduce_mean(tf.square(self.f_D)) \
                   + tf.reduce_mean(tf.square(self.f_con))
        # small regularization on movement magnitudes
        self.lossM = self.lambda_M * tf.reduce_mean(self.M_bins)

        self.loss = 1.0*self.lossU0 + 5.0*self.lossU + self.lossF + self.lossM

        # Optimizers
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.loss, method='L-BFGS-B',
            options={'maxiter': 50000, 'maxfun': 50000, 'maxcor': 50,
                     'maxls': 50, 'ftol': 1.0 * np.finfo(float).eps})

        # Init
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver(var_list=tf.global_variables())

    # ---------------- NN blocks ----------------
    def initialize_NN(self, layers):
        weights, biases = [], []
        for l in range(0, len(layers)-1):
            W = self.xavier_init([layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim, out_dim = size
        std = np.sqrt(2.0/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=std, dtype=tf.float64), dtype=tf.float64)

    def neural_net(self, t, weights, biases):
        # map t to [-1,1] internally (like your original)
        H = 2.0*(t-self.lb)/(self.ub-self.lb) - 1.0
        for l in range(0, len(weights)-1):
            H = tf.tanh(tf.add(tf.matmul(H, weights[l]), biases[l]))
        Y = tf.add(tf.matmul(H, weights[-1]), biases[-1])
        return Y

    def net_u(self, t):
        """
        Output all patches at once. Final layer size must be 4*P.
        Returns S,I,R,D each of shape (len(t), P).
        """
        Y = self.neural_net(t, self.weights, self.biases)   # (T, 4P)
        Y = tf.reshape(Y, (-1, self.P, 4))                  # (T, P, 4)
        S = Y[:, :, 0]; I = Y[:, :, 1]; R = Y[:, :, 2]; D = Y[:, :, 3]
        return S, I, R, D

    def net_Beta(self, t):
        BetaI = self.neural_net(t, self.weights_Beta, self.biases_Beta)
        # bound to [0,1] like original
        return (tf.sigmoid(BetaI) - 0.0) * 1.0

    # Kappa (shared)
    def net_Kappa_plot(self, COEF):
        polys = tf.constant(np.transpose(Jacobi_polys_plots[:poly_order, :]), dtype=tf.float64)
        Kappa = tf.matmul(polys, COEF)
        return 0.2 + 0.8*tf.sigmoid(Kappa)

    def net_Kappa(self, COEF):
        polys = tf.constant(np.transpose(Jacobi_polys[:poly_order, :]), dtype=tf.float64)
        Kappa = tf.matmul(polys, COEF)
        return 0.2 + 0.8*tf.sigmoid(Kappa)

    # Fractional L1 coefficients (vectorized across columns)
    def FDM1(self, Kappa):
        m = self.M
        Tau = self.tau
        kappa_vec = tf.reshape(Kappa, [m+1,1])  # (m+1,1)
        kappa_mat = tf.tile(kappa_vec, [1, m-1])

        idx = np.tril_indices(m+1, k=-1)
        Temp1 = np.zeros([m+1, m+1]); Temp1[idx] = idx[0]-idx[1]; Temp1 = np.tril(Temp1, k=-2)
        Temp2 = -np.eye(m+1);         Temp2[idx] = idx[0]-idx[1]-1; Temp2 = np.tril(Temp2, k=-2)
        Temp3 = -2*np.eye(m+1);       Temp3[idx] = idx[0]-idx[1]-2; Temp3 = np.tril(Temp3, k=-2)

        A = np.concatenate((np.zeros((1,m)), np.eye(m)), axis=0); A = A[:,0:m-1]

        Temp1 = tf.constant(Temp1[:,0:m-1], dtype=tf.float64)
        Temp2 = tf.constant(Temp2[:,0:m-1], dtype=tf.float64)
        Temp3 = tf.constant(Temp3[:,0:m-1], dtype=tf.float64)
        A     = tf.constant(A, dtype=tf.float64)

        Temp = tf.pow(Temp1, 1.0-kappa_mat) - 2*tf.pow(Temp2, 1.0-kappa_mat) + tf.pow(Temp3, 1.0-kappa_mat) + A

        L_Temp1 = tf.pow(tf.reshape(tf.constant(np.arange(m),   dtype=tf.float64), [m,1]), 1.0-kappa_vec[1:m+1, 0:1])
        L_Temp2 = tf.pow(tf.reshape(tf.constant(np.arange(m)+1, dtype=tf.float64), [m,1]), 1.0-kappa_vec[1:m+1, 0:1])
        L_Temp  = tf.concat((tf.zeros((1,1), dtype=tf.float64), L_Temp1 - L_Temp2), axis=0)
        R_Temp  = tf.concat((tf.zeros((m,1), dtype=tf.float64), tf.ones((1,1), dtype=tf.float64)), axis=0)

        coeff_mat = tf.concat((L_Temp, Temp, R_Temp), axis=1)

        c = tf.tile(tf.math.divide(tf.pow(Tau, -kappa_vec), tf.exp(tf.lgamma(2-kappa_vec))),
                    multiples=tf.constant([1, m+1], dtype=tf.int32))
        coeff_mat = tf.multiply(c, coeff_mat)
        return coeff_mat  # (m+1, m+1)

    def net_f(self, t):
        """
        Residuals on the residual grid t_f (must call with t = self.t_tf).
        Vectorized across patches; includes movement M_seq_f.
        """
        # States on residual grid
        S, I, R, D = self.net_u(t)  # (m+1,P) each

        # Beta on residual grid (broadcast to (m+1,P))
        r = self.net_Beta(t)        # (m+1,1)
        rP = r * tf.ones([tf.shape(S)[0], self.P], dtype=tf.float64)

        # Fractional derivatives (vectorized columnwise)
        K1 = self.net_Kappa(self.Kappa1_COEF)
        K2 = self.net_Kappa(self.Kappa2_COEF)
        K3 = self.net_Kappa(self.Kappa3_COEF)
        K4 = self.net_Kappa(self.Kappa4_COEF)

        D1 = self.FDM1(K1); D2 = self.FDM1(K2); D3 = self.FDM1(K3); D4 = self.FDM1(K4)
        S_t = tf.matmul(D1, S)
        I_t = tf.matmul(D2, I)
        R_t = tf.matmul(D3, R)
        D_t = tf.matmul(D4, D)

        # fractional scaling (as in your original)
        Tsc = tf.constant(7.0, dtype=tf.float64)
        S_t = tf.pow(Tsc, K1-1.0)*S_t/tf.exp(tf.lgamma(1.0+K1))
        I_t = tf.pow(Tsc, K2-1.0)*I_t/tf.exp(tf.lgamma(1.0+K2))
        R_t = tf.pow(Tsc, K3-1.0)*R_t/tf.exp(tf.lgamma(1.0+K3))
        D_t = tf.pow(Tsc, K4-1.0)*D_t/tf.exp(tf.lgamma(1.0+K4))

        # Movement terms (per time): inflow - outflow, for S/I/R
        # self.M_seq_f: (m+1,P,P)
        row_sums = tf.reduce_sum(self.M_seq_f, axis=2)  # (m+1,P)

        # Inflows: X (t,P) with M(t) (t,P,P) -> (t,P)
        infl_S = tf.einsum('tp,tpq->tq', S, self.M_seq_f)
        infl_I = tf.einsum('tp,tpq->tq', I, self.M_seq_f)
        infl_R = tf.einsum('tp,tpq->tq', R, self.M_seq_f)
        out_S  = S * row_sums
        out_I  = I * row_sums
        out_R  = R * row_sums

        move_S = infl_S - out_S
        move_I = infl_I - out_I
        move_R = infl_R - out_R

        # Incidence (per patch); use living population N_live = S+I+R (SIR+D)
        eps = 1e-12
        N_live = S + I + R
        inc = rP * S * I / (N_live + eps)   # (m+1,P)

        # Residuals (==0 ideally)
        f_S = S_t + inc - move_S
        f_I = I_t - inc + (self.a + self.b)*I - move_I
        f_R = R_t - self.a*I - move_R
        f_D = D_t - self.b*I

        # Mass conservation per patch over time
        ones_t = tf.ones([tf.shape(S)[0], 1], dtype=tf.float64)
        N_broadcast = ones_t * self.N_tf        # (m+1,P)
        f_con = S + I + R + D - N_broadcast

        return f_S, f_I, f_R, f_D, f_con

    # ---------------- Training / Predict / Movement ----------------
    def callback(self, loss, lossU0, lossU, lossF):
        print('Loss: %.3e, U0: %.3e, U: %.3e, F: %.3e' % (loss, lossU0, lossU, lossF))

    def train(self, nIter=10000, use_lbfgs=True):
        tf_dict = {
            self.t_u: self.t_train,
            self.t_tf: self.t_f,
            self.I_u: self.I_train,
            self.R_u: self.R_train,
            self.D_u: self.D_train,
            self.S0_u: self.S0,
            self.I0_u: self.I0,
            self.R0_u: self.R0,
            self.D0_u: self.D0
        }
        t0 = time.time()
        for it in range(nIter+1):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 100 == 0:
                loss_v, l0, lu, lf = self.sess.run([self.loss, self.lossU0, self.lossU, self.lossF], tf_dict)
                print(f"It {it:5d} | loss {loss_v:.3e} | U0 {l0:.3e} | U {lu:.3e} | F {lf:.3e} | {time.time()-t0:.1f}s")
                t0 = time.time()
        if use_lbfgs:
            self.optimizer.minimize(self.sess, feed_dict=tf_dict,
                                    fetches=[self.loss, self.lossU0, self.lossU, self.lossF],
                                    loss_callback=self.callback)

    def predict(self, t_star):
        tf_dict = {self.t_u: t_star}
        S = self.sess.run(self.S_pred, tf_dict)
        I = self.sess.run(self.I_pred, tf_dict)
        R = self.sess.run(self.R_pred, tf_dict)
        D = self.sess.run(self.D_pred, tf_dict)
        Beta = self.sess.run(self.BetaI, tf_dict)
        K1 = self.sess.run(self.Kappa_pred1, {self.t_u: t_star})
        K2 = self.sess.run(self.Kappa_pred2, {self.t_u: t_star})
        K3 = self.sess.run(self.Kappa_pred3, {self.t_u: t_star})
        K4 = self.sess.run(self.Kappa_pred4, {self.t_u: t_star})
        return S, I, R, D, K1, K2, K3, K4, Beta

    def movement_matrices(self):
        """Return learned movement matrices per bin as a numpy array (B,P,P)."""
        return self.sess.run(self.M_bins)
