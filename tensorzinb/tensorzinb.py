import warnings
import time

# import tensorflow as tf
import contextlib
import os
import numpy as np
from keras.models import Model
from keras.layers import Lambda, Input, Dense, RepeatVector, Reshape, Add
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras import backend as K
from scipy.special import gammaln
import statsmodels.api as sm


from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.backend import get_session
from tensorflow.python.framework.ops import disable_eager_execution


def import_tensorflow():
    # Filter tensorflow version warnings
    import os

    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
    import warnings

    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=Warning)
    import tensorflow as tf

    tf.get_logger().setLevel("INFO")
    tf.autograph.set_verbosity(0)
    import logging

    tf.get_logger().setLevel(logging.ERROR)
    return tf


tf = import_tensorflow()

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()


class ZINBLogLik:
    def __init__(
        self, pi=None, log_theta=None, nb_only=False, scope="zinb/", zero_threshold=1e-8
    ):
        self.pi = pi
        self.zero_threshold = zero_threshold
        self.scope = scope
        self.log_theta = log_theta
        self.nb_only = nb_only
        self.y = None
        self.llf = None

    def loss(self, y_true, y_pred):
        with tf.name_scope(self.scope):
            y = tf.cast(y_true, tf.float32)
            # mu is already in log
            mu = tf.cast(y_pred, tf.float32)
            log_theta = self.log_theta
            theta = tf.math.exp(log_theta)

            t1 = tf.math.lgamma(y + theta)
            t2 = -tf.math.lgamma(theta)
            t3 = theta * log_theta
            t4 = y * mu
            ty = tf.reduce_logsumexp(tf.stack([log_theta, mu], axis=0), axis=0)
            t5 = -(theta + y) * ty

            if self.nb_only:
                result = -(t1 + t2 + t3 + t4 + t5)
            else:
                log_q0 = -tf.nn.softplus(-self.pi)
                # log_q1 = -tf.nn.softplus(self.pi) = -tf.nn.softplus(-self.pi) - self.pi
                log_q1 = log_q0 - self.pi

                nb_case = -(t1 + t2 + t3 + t4 + t5 + log_q1)

                p1 = theta * (log_theta - ty) + log_q1
                zero_case = -tf.reduce_logsumexp(tf.stack([log_q0, p1], axis=0), axis=0)
                result = tf.where(tf.less(y, self.zero_threshold), zero_case, nb_case)
            self.llf = tf.reduce_mean(result, axis=0)
            self.y = y
            result = tf.reduce_sum(self.llf)

        return result


class PredictionCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.weights = []

    def on_epoch_end(self, epoch, logs={}):
        ws = (
            np.concatenate([np.array(w.flatten()) for w in self.model.get_weights()])
        ).flatten()
        self.weights.append(ws)


class ReduceLROnPlateauSkip(ReduceLROnPlateau):
    def __init__(
        self,
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_lr=0,
        num_epoch_skip=3,
        **kwargs,
    ):
        self.num_epoch_skip = num_epoch_skip

        super(ReduceLROnPlateauSkip, self).__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr,
            **kwargs,
        )

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.num_epoch_skip:
            return
        super(ReduceLROnPlateauSkip, self).on_epoch_end(epoch, logs)


class TensorZINB:
    def __init__(
        self,
        endog,
        exog,
        exog_c=None,
        exog_infl=None,
        exog_infl_c=None,
        same_dispersion=False,
        nb_only=False,
        **kwargs,
    ):
        self.endog = endog
        if len(endog.shape) == 1:
            self.num_sample = len(endog)
            self.num_out = 1
            self.endog = endog.reshape((-1, 1))
        else:
            self.num_sample, self.num_out = np.shape(endog)

        df_model = 0
        df_model_c = 0

        self.exog = exog
        df_model = np.linalg.matrix_rank(exog)
        if len(exog.shape) == 1:
            self.k_exog = 1
            self.exog = exog.reshape((-1, 1))
        else:
            self.k_exog = exog.shape[1]

        self.exog_c = exog_c
        if exog_c is None:
            self.k_exog_c = 0
            self._no_exog_c = True
            self.exog_c = np.zeros((self.num_sample, self.k_exog_c), dtype=np.float64)
        else:
            self.k_exog_c = exog_c.shape[1]
            self._no_exog_c = False
            if self.k_exog_c > 0:
                df_model_c = df_model_c + np.linalg.matrix_rank(exog_c)

        self.nb_only = nb_only
        if exog_infl is None and exog_infl_c is None:
            self.nb_only = True

        self.exog_infl = exog_infl
        if exog_infl is None or self.nb_only:
            self.k_exog_infl = 0
            self._no_exog_infl = True
            self.exog_infl = np.ones(
                (self.num_sample, self.k_exog_infl), dtype=np.float64
            )
        else:
            self.k_exog_infl = exog_infl.shape[1]
            self._no_exog_infl = False
            if self.k_exog_infl > 0:
                df_model = df_model + np.linalg.matrix_rank(exog_infl)

        self.exog_infl_c = exog_infl_c
        if exog_infl_c is None or self.nb_only:
            self.k_exog_infl_c = 0
            self._no_exog_infl_c = True
            self.exog_infl_c = np.ones(
                (self.num_sample, self.k_exog_infl_c), dtype=np.float64
            )
        else:
            self.k_exog_infl_c = exog_infl_c.shape[1]
            self._no_exog_infl_c = False
            if self.k_exog_infl_c > 0:
                df_model_c = df_model_c + np.linalg.matrix_rank(exog_infl_c)

        df_model_each = df_model + df_model_c
        df_model = df_model * self.num_out + df_model_c

        self.same_dispersion = same_dispersion
        if same_dispersion:
            self.k_disperson = 1
            df_model = df_model + 1
        else:
            self.k_disperson = self.num_out
            df_model = df_model + self.num_out
        df_model_each = df_model_each + 1
        self.df_model = df_model
        self.df_model_each = df_model_each

        self.loglike_method = "nb2"

    def fit(
        self,
        init_weights={},
        init_method="poi",
        device_type="CPU",
        device_name=None,
        return_history=False,
        epochs=5000,
        learning_rate=0.01,
        num_epoch_skip=3,
        is_early_stop=True,
        is_reduce_lr=True,
        min_delta_early_stop=0.05,
        patience_early_stop=50,
        factor_reduce_lr=0.8,
        patience_reduce_lr=10,
        min_lr=0.001,
        reset_keras_session=False,
        **kwargs,
    ):
        if device_name is None:
            devices = tf.config.list_logical_devices(device_type)
            device_name = devices[0].name

        num_sample = self.num_sample
        num_out = self.num_out
        num_feat = self.k_exog
        num_feat_infl = self.k_exog_infl
        num_feat_c = self.k_exog_c
        num_feat_infl_c = self.k_exog_infl_c
        num_dispersion = self.k_disperson

        # initiate weights
        if len(init_weights) == 0:
            if init_method == "poi":
                init_weights = self._poisson_init()
            elif init_method == "nb":
                init_weights = self._nb_init()

        # use distinct names to retrieve weights from layers
        weight_keys = ["x_mu", "z_mu", "x_pi", "z_pi", "theta"]

        if reset_keras_session:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                reset_keras()
                K.clear_session()

        with tf.device(device_name):
            disable_eager_execution()

            inputs = Input(shape=(num_feat,))
            inputs_infl = Input(shape=(num_feat_infl,))
            inputs_c = Input(shape=(num_feat_c,))
            inputs_infl_c = Input(shape=(num_feat_infl_c,))
            inputs_theta = Input(shape=(1,))

            if 'x_mu' in init_weights:
                x = Dense(num_out, use_bias=False, name='x_mu', weights=[init_weights['x_mu']])(inputs)
            else:
                x = Dense(num_out, use_bias=False, name='x_mu')(inputs)
            if num_feat_c>0:
                if 'z_mu' in init_weights:
                    x_c = Dense(1, use_bias=False, name='z_mu', weights=[init_weights['z_mu']])(inputs_c)
                else:
                    x_c = Dense(1, use_bias=False, name='z_mu')(inputs_c)
                predictions = Add()([x,x_c])
            else:
                predictions = x

            if self.nb_only:
                pi = None
            else:
                if 'x_pi' in init_weights:
                    x_infl = Dense(num_out, use_bias=False, name='x_pi', weights=[init_weights['x_pi']])(inputs_infl)
                else:
                    x_infl = Dense(num_out, use_bias=False, name='x_pi')(inputs_infl)

                if num_feat_infl_c>0:
                    if 'z_pi' in init_weights:
                        x_infl_c = Dense(1, use_bias=False, name='z_pi', weights=[init_weights['z_pi']])(inputs_infl_c)
                    else:
                        x_infl_c = Dense(1, use_bias=False, name='z_pi')(inputs_infl_c)
                    pi = Add()([x_infl, x_infl_c])
                else:
                    pi = x_infl

            if 'theta' in init_weights:
                theta0 = Dense(num_dispersion, use_bias=False, name='theta', weights=[init_weights['theta']])(inputs_theta)
            else:
                theta0 = Dense(num_dispersion, use_bias=False, name='theta')(inputs_theta)
            if self.same_dispersion:
                theta = Reshape((num_out,))(RepeatVector(num_out)(theta0))
            else:
                theta = theta0

            zinb = ZINBLogLik(pi, theta, nb_only=self.nb_only)

            if self.nb_only:
                output = Lambda(lambda x: x[0])([predictions, theta])
            else:
                output = Lambda(lambda x: x[0])([predictions, pi, theta])

            model = Model(
                inputs=[inputs, inputs_c, inputs_infl, inputs_infl_c, inputs_theta],
                outputs=[output],
            )

            # with contextlib.redirect_stdout(open(os.devnull, "w")):
            #     model_summary = model.summary()

            opt = RMSprop(learning_rate=learning_rate)
            model.compile(loss=zinb.loss, optimizer=opt)

            callbacks = []
            if is_early_stop:
                early_stop = EarlyStopping(
                    monitor="loss",
                    min_delta=min_delta_early_stop / num_sample,
                    patience=patience_early_stop,
                    mode="min",
                )
                callbacks.append(early_stop)

            if is_reduce_lr:
                reduce_lr = ReduceLROnPlateauSkip(
                    monitor="loss",
                    factor=factor_reduce_lr,
                    patience=patience_reduce_lr,
                    min_lr=min_lr,
                    num_epoch_skip=num_epoch_skip,
                )
                callbacks.append(reduce_lr)

            if return_history:
                # this get all weights over epoch
                get_weights = PredictionCallback()
                callbacks.append(get_weights)

            # TODO: FIX this. code randomly crashes on apple silicon M1/M2 with 
            # error `Incompatible shapes`. code usually runs fine after second try.
            # similar to this issue https://developer.apple.com/forums/thread/701985
            for i in range(10):
                try:
                    start_time = time.time()
                    losses = model.fit(
                        [
                            self.exog,
                            self.exog_c,
                            self.exog_infl,
                            self.exog_infl_c,
                            np.ones((num_sample, 1)),
                        ],
                        [self.endog],
                        callbacks=callbacks,
                        batch_size=num_sample,
                        epochs=epochs,
                        verbose=0,
                    )
                    cpu_time = time.time() - start_time
                    break
                except Exception as e:
                    print(model.summary())
                    print('--------------')
                    print(num_sample)
                    print('--------------')
                    print(np.ones((num_sample, 1)))
                    print('--------------')
                    print(inputs_theta)
                    print('--------------')
                    print(theta)
                    print('--------------')
                    print(e)
                    continue

            # retrieve LL
            get_llfs = K.function(
                [inputs, inputs_c, inputs_infl, inputs_infl_c, inputs_theta, zinb.y],
                [zinb.llf],
            )
            llft = get_llfs(
                [
                    self.exog,
                    self.exog_c,
                    self.exog_infl,
                    self.exog_infl_c,
                    np.ones((num_sample, 1)),
                    self.endog,
                ]
            )[0]

            llfs = -(llft * num_sample + np.sum(gammaln(self.endog + 1), axis=0))
            aics = -2 * (llfs - self.df_model_each)

            llf = np.sum(llfs)
            aic = -2 * (llf - self.df_model)

            # get weights
            weights = model.get_weights()

            names_t = [
                (weight.name).split("/")[0]
                for layer in model.layers
                for weight in layer.weights
            ]
            # tf layer weight has subscript in names
            weight_names = []
            for name in names_t:
                matched = name
                for nt in weight_keys:
                    if nt in name:
                        matched = nt
                        break
                weight_names.append(matched)

            weights_dict = dict(zip(weight_names, weights))

            res = {
                "llf_total": llf,
                "llfs": llfs,
                "aic_total": aic,
                "aics": aics,
                "df_model_total": self.df_model,
                "df": self.df_model_each,
                "weights": weights_dict,
                "cpu_time": cpu_time,
                "num_sample": num_sample,
                "epochs": len(losses.history["loss"]),
            }

            if return_history:
                res["loss_history"] = losses.history["loss"]
                res["weights_history"] = np.array(get_weights.weights)

        return res

    # https://github.com/statsmodels/statsmodels/blob/main/statsmodels/discrete/discrete_model.py#L3691
    def _estimate_dispersion(self, mu, resid, df_resid=None, loglike_method="nb2"):
        if df_resid is None:
            df_resid = resid.shape[0]
        if loglike_method == "nb2":
            a = ((resid**2 / mu - 1) / mu).sum() / df_resid
        else:  # self.loglike_method == 'nb1':
            a = (resid**2 / mu - 1).sum() / df_resid
        return a

    def _compute_pi_init(self, nz_prob, p_nonzero, infl_prob_max=0.99):
        ww = 1 - min(nz_prob / p_nonzero, infl_prob_max)
        return -np.log(1 / ww - 1)

    def _poisson_init_each(
        self,
        Y,
        estimate_infl=True,
        eps=1e-10,
        maxiter=100,
        theta_lb=0.05,
        intercept_var_th=1e-3,
        infl_prob_max=0.99,
    ):
        find_poi_sol = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                poi_mod = sm.Poisson(Y, self.exog).fit(
                    maxiter=maxiter, disp=False, warn_convergence=False
                )
                if np.isnan(poi_mod.params).any():
                    find_poi_sol = False
                else:
                    mu = poi_mod.predict()
                    a = self._estimate_dispersion(
                        mu, poi_mod.resid, df_resid=poi_mod.df_resid
                    )
                    theta = 1 / max(a, theta_lb)
                    x_mu = np.reshape(poi_mod.params, (self.k_exog, 1))
            except:
                find_poi_sol = False

        if not find_poi_sol:
            vs = np.std(self.exog, axis=0)
            # find intercept index
            min_idx = np.argmin(vs)
            if vs[min_idx] < intercept_var_th:
                x_mu = np.zeros((self.k_exog, 1))
                mu = np.mean(Y)
                x_mu[min_idx] = np.log(mu) / np.mean(self.exog[:, min_idx])
                resid = Y - mu
                a = self._estimate_dispersion(mu, resid)
                if np.isnan(a) or np.isinf(a):
                    a = theta_lb
                theta = 1 / max(a, theta_lb)
            else:
                x_mu = None
                return {}

        weights = {"x_mu": x_mu, "theta": np.array([np.log(theta)]).reshape((-1, 1))}

        if not self._no_exog_infl and estimate_infl:
            pred = np.maximum(mu, 10 * eps)
            p_nonzero = 1 - np.mean(np.power(theta / (theta + pred + eps), theta))

            # find intercept index
            vs = np.std(self.exog_infl, axis=0)
            min_idx = np.argmin(vs)
            if vs[min_idx] < intercept_var_th:
                nz_prob = np.mean(Y > 0)
                x_pi = np.zeros((self.k_exog_infl, 1))
                fv = np.mean(self.exog_infl[:, min_idx])
                w_pi = self._compute_pi_init(
                    nz_prob, p_nonzero, infl_prob_max=infl_prob_max
                )
                x_pi[min_idx] = w_pi / fv
                weights["x_pi"] = x_pi

        return weights

    def _poisson_init(
        self,
        eps=1e-10,
        maxiter=100,
        theta_lb=0.05,
        intercept_var_th=1e-3,
        infl_prob_max=0.99,
    ):
        x_mu = []
        x_pi = []
        theta = []
        return_x_pi = True
        return_theta = True
        for i in range(self.num_out):
            w = self._poisson_init_each(
                self.endog[:, i],
                eps=eps,
                maxiter=maxiter,
                theta_lb=theta_lb,
                intercept_var_th=intercept_var_th,
                infl_prob_max=infl_prob_max,
            )

            if len(w) == 0:
                return {}

            if "x_mu" in w:
                x_mu.append(w["x_mu"])
            else:
                return {}

            if "x_pi" in w:
                x_pi.append(w["x_pi"])
            else:
                return_x_pi = False

            if "theta" in w:
                theta.append(w["theta"])
            else:
                return_theta = False
        weights = {"x_mu": np.concatenate(x_mu, axis=1)}
        if return_x_pi:
            weights["x_pi"] = np.concatenate(x_pi, axis=1)
        if return_theta:
            t = np.concatenate(theta, axis=1)
            if self.same_dispersion:
                weights["theta"] = np.array(np.mean(t)).reshape((-1, 1))
            else:
                weights["theta"] = t
        return weights

    def _nb_init(self, infl_prob_max=0.99, intercept_var_th=1e-3):
        nb_mod = TensorZINB(
            self.endog,
            self.exog,
            exog_c=self.exog_c,
            same_dispersion=self.same_dispersion,
            nb_only=True,
        )
        nb_res = nb_mod.fit(init_method="poi")
        weights = nb_res["weights"]

        if self._no_exog_infl:
            return weights
        # find intercept index
        vs = np.std(self.exog_infl, axis=0)
        min_idx = np.argmin(vs)
        # do not compute logit weight if there is no intercept
        if vs[min_idx] < intercept_var_th:
            x_pi = np.zeros((self.k_exog_infl, self.num_out))
            fv = np.mean(self.exog_infl[:, min_idx])
            if self.same_dispersion:
                theta = np.exp(
                    np.array(list(weights["theta"].flatten()) * self.num_out)
                )
            else:
                theta = np.exp(weights["theta"].flatten())

            mu_c = 0
            if self.k_exog_c > 0 and "z_mu" in weights:
                mu_c = np.dot(self.exog_c, weights["z_mu"])

            for i in range(self.num_out):
                mu = np.dot(self.exog, weights["x_mu"][:, i]) + mu_c
                mu = np.exp(mu)
                p_nonzero = 1 - np.mean(np.power(theta[i] / (theta[i] + mu), theta[i]))
                nz_prob = np.mean(self.endog[:, i] > 0)
                w_pi = self._compute_pi_init(
                    nz_prob, p_nonzero, infl_prob_max=infl_prob_max
                )
                x_pi[min_idx, i] = w_pi / fv

            weights["x_pi"] = x_pi

            if self.k_exog_infl_c > 0:
                weights["z_pi"] = np.zeros((self.k_exog_infl_c, 1))

        return weights
