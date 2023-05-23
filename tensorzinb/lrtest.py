import pandas as pd
import numpy as np
from patsy import dmatrices
from scipy import stats
from .tensorzinb import TensorZINB


class LRTest:
    def __init__(
        self,
        df_data,
        df_feature,
        conditions,
        nb_features,
        nb_features_c=None,
        infl_features=None,
        infl_features_c=None,
        add_intercept=True,
        nb_only=False,
        same_dispersion=False,
    ):
        self.df_data = df_data
        self.df_feature = df_feature
        self.conditions = list(conditions)
        self.nb_features = list(set(nb_features) - set(conditions))
        self.nb_features_c = nb_features_c
        if nb_features_c is None or len(nb_features_c) == 0:
            self._no_nb_c = True
        else:
            self._no_nb_c = False

        if infl_features is not None and len(infl_features) > 0:
            self.infl_features = list(set(infl_features) - set(conditions))
            self._no_infl = False
        else:
            self.infl_features = None
            self._no_infl = True

        self.infl_features_c = infl_features_c
        if infl_features_c is None or len(infl_features_c) == 0:
            self._no_infl_c = True
        else:
            self._no_infl_c = False

        self.nb_only = nb_only
        if self._no_infl_c and self._no_infl:
            self.nb_only = True

        self.add_intercept = add_intercept
        self.same_dispersion = same_dispersion

        self._gen_feature_dfs()

        self.res0 = None
        self.res1 = None
        self.df_result = None

    def _get_feature_idx_map(self, Xs):
        df_maps = []
        for X in Xs:
            df_f = pd.DataFrame(X.columns, columns=["feature"])
            df_f["idx"] = range(len(df_f))
            df_maps.append(df_f)
        dft = pd.merge(df_maps[1], df_maps[0], on="feature")
        idx_map = [dft.idx_x.values, dft.idx_y.values]
        return idx_map

    def _gen_feature_df(self, df_feature, features, add_intercept=True):
        if add_intercept:
            formula = "{} ~ {}".format("1", " + ".join(features))
        else:
            formula = "{} ~ {} - 1".format("1", " + ".join(features))

        _, predictors = dmatrices(formula, df_feature, return_type="dataframe")
        return predictors

    def _gen_feature_dfs(self):
        fs = [self.nb_features, self.nb_features + self.conditions]
        Xs = []
        dfs = []
        for feature in fs:
            df = self._gen_feature_df(
                self.df_feature, feature, add_intercept=self.add_intercept
            )
            dfs.append(df)
            Xs.append(df.values)
        self.Xs = Xs
        self.X_idx_map = self._get_feature_idx_map(dfs)

        if self._no_nb_c:
            self.X_c = None
        else:
            self.X_c = self._gen_feature_df(
                self.df_feature, self.nb_features_c, add_intercept=False
            ).values

        if not self.nb_only and not self._no_infl:
            fs = [self.infl_features, self.infl_features + self.conditions]
            X_infls = []
            dfs = []
            for feature in fs:
                df = self._gen_feature_df(
                    self.df_feature, feature, add_intercept=self.add_intercept
                )
                dfs.append(df)
                X_infls.append(df.values)
            self.X_infls = X_infls
            self.X_infl_idx_map = self._get_feature_idx_map(dfs)
        else:
            self.X_infls = [None, None]
            self.X_infl_idx_map = None

        if self._no_infl_c:
            self.X_infl_c = None
        else:
            self.X_infl_c = self._gen_feature_df(
                self.df_feature, self.infl_features_c, add_intercept=False
            ).values

    def run(self, learning_rate=0.008, epochs=5000):

        zinb0 = TensorZINB(
            self.df_data.values,
            self.Xs[0],
            exog_c=self.X_c,
            exog_infl=self.X_infls[0],
            exog_infl_c=self.X_infl_c,
            same_dispersion=self.same_dispersion,
            nb_only=self.nb_only,
        )

        res0 = zinb0.fit(learning_rate=learning_rate, epochs=epochs)
        self.res0 = res0

        zinb1 = TensorZINB(
            self.df_data.values,
            self.Xs[1],
            exog_c=self.X_c,
            exog_infl=self.X_infls[1],
            exog_infl_c=self.X_infl_c,
            same_dispersion=self.same_dispersion,
            nb_only=self.nb_only,
        )

        weights = res0["weights"]
        if "x_mu" in weights:
            x_mu = np.zeros((zinb1.k_exog, zinb1.num_out))
            x_mu[self.X_idx_map[0], :] = weights["x_mu"][self.X_idx_map[1], :]
            weights["x_mu"] = x_mu

        if "x_pi" in weights:
            x_pi = np.zeros((zinb1.k_exog_infl, zinb1.num_out))
            x_pi[self.X_infl_idx_map[0], :] = weights["x_pi"][self.X_infl_idx_map[1], :]
            weights["x_pi"] = x_pi

        res1 = zinb1.fit(init_weights=weights, learning_rate=learning_rate, epochs=epochs)
        self.res1 = res1

        dfr = pd.DataFrame(self.df_data.columns, columns=["test"])
        dfr["llf0"] = res0["llfs"]
        dfr["aic0"] = res0["aics"]
        dfr["df0"] = res0["df"]
        dfr["cpu_time0"] = res0["cpu_time"] / len(dfr)
        dfr["llf1"] = res1["llfs"]
        dfr["aic1"] = res1["aics"]
        dfr["df1"] = res1["df"]
        dfr["cpu_time1"] = res1["cpu_time"] / len(dfr)

        dfr["llfd"] = dfr["llf1"] - dfr["llf0"]
        dfr["aicd"] = dfr["aic1"] - dfr["aic0"]
        dfd = dfr["df1"] - dfr["df0"]
        dfr["pvalue"] = 1 - stats.chi2.cdf(2 * dfr["llfd"], dfd)

        self.df_result = dfr

        return dfr
