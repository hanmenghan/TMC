import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


# loss function
def KL(alpha, c):
    beta = tf.ones((1, c))
    S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
    S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
    ln_B = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
    ln_B_uni = tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True) - tf.math.lgamma(S_beta)
    dg0 = tf.math.digamma(S_alpha)
    dg1 = tf.math.digamma(alpha)
    kl = tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + ln_B + ln_B_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    E = alpha - 1
    label = tf.one_hot(p, depth=c)
    A = tf.reduce_sum(label * (tf.math.digamma(S) - tf.math.digamma(alpha)), axis=1, keepdims=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return A + B


class TMC(Model):
    def __init__(self, classes, views, classifier_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param lambda_epochs: KL divergence annealing epoch during training
        """
        super(TMC, self).__init__()
        self.views = views
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.Classifiers = [Classifier(classifier_dims[i], self.classes) for i in range(self.views)]

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = tf.reduce_sum(alpha[v], axis=1, keepdims=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / tf.broadcast_to(S[v], E[v].shape)
                u[v] = self.classes / S[v]

            # b^0 @ b^(0+1)
            bb = tf.matmul(tf.reshape(b[0], [-1, self.classes, 1]), tf.reshape(b[1], [-1, 1, self.classes]))
            # b^0 * u^1
            uv1_expand = tf.broadcast_to(u[1], b[0].shape)
            bu = tf.multiply(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = tf.broadcast_to(u[0], b[0].shape)
            ub = tf.multiply(b[1], uv_expand)
            # calculate C
            bb_sum = tf.reduce_sum(bb, axis=(1, 2))
            bb_diag = tf.linalg.diag_part(bb)
            C = bb_sum - tf.reduce_sum(bb_diag, axis=-1)

            # calculate b^a
            b_a = (tf.multiply(b[0], b[1]) + bu + ub) / tf.broadcast_to(tf.reshape(1 - C, [-1, 1]), b[0].shape)
            # calculate u^a
            u_a = tf.multiply(u[0], u[1]) / tf.broadcast_to(tf.reshape(1 - C, [-1, 1]), u[0].shape)

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = tf.multiply(b_a, tf.broadcast_to(S_a, b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    def call(self, X, y, global_step):
        # step one
        evidence = self.infer(X)
        loss = 0
        alpha = dict()
        for v_num in range(len(X)):
            # step two
            alpha[v_num] = evidence[v_num] + 1
            # step three
            loss += ce_loss(y, alpha[v_num], self.classes, global_step, self.lambda_epochs)
        # step four
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)
        loss = tf.reduce_mean(loss)
        return evidence, evidence_a, loss

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input[v_num])
        return evidence


class Classifier(Model):
    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = []
        for i in range(self.num_layers - 1):
            self.fc.append(Dense(classifier_dims[i + 1]))
        self.fc.append(Dense(classes))

    def call(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return tf.keras.activations.softplus(h)
