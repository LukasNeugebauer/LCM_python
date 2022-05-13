"""
Python implementation of Sam Gershman's LCM toolbox. https://github.com/sjgershm/LCM
Pretty much pure numpy.
Written to understand what he's doing, but maybe it's of help to someone.

January 2020, Lukas Neugebauer (l.neugebauer@uke.de)
"""

import numpy as np
from random import choices
from scipy.stats import norm


class ParticleFilter:
    """
    Implement the particle filter algorithm for binary features.
    This class is roughly equivalent to "LCM_infer" in the LCM toolbox,
    i.e. it implements the actual algorithm.
    """

    def __init__(self, features, opts={}):
        """
        Expected input:
            features -  (TxD) array of features per trials
                        where first column is US
                        Can be anything that can be easily
                        converted to np.array
            opts -      dictionary of options. Possible fields (defaults):
                        alpha (0.1): parameter for probability of new latent causes
                        stickiness: tendency to stick to current cause
                        beta_prior ([1, 1]): parameters of prior on likelihood of features
                        max_cause (10):  maximum number of causes
                        n_particles (100): default: 100, number of particles
        """
        self.opts = opts
        if not isinstance(features, np.ndarray):
            try:
                features = np.array(features)
            except Exception:
                raise RuntimeError(
                    "Please give features as np.array or"
                    + "something that can be converted to one."
                )
        self.features = features
        # infer number of trials and features
        self.n_trials, self.n_features = features.shape
        # initialize arrays
        self.init_arrays()

    @property
    def opts(self):
        """
        Wrapper around all options. Just for convenience to get
        all options at once
        """
        return {
            "alpha": self.alpha,
            "stickiness": self.stickiness,
            "beta_prior": self.beta_prior,
            "max_cause": self.max_cause,
            "n_particles": self.n_particles,
        }

    @opts.setter
    def opts(self, opts):
        """
        Handle (partial) default options
        Defaults are the same as in Gershman's MATLAB version
        """
        defaults = {
            "alpha": 0,  # parameter for probability of new latent causes
            "stickiness": 0,  # tendency to stick to the current cause assignment
            "beta_prior": [1, 1],  # prior on likelihood of feature
            "max_cause": 10,  # maximum number of causes
            "n_particles": 100,  # number of particles
        }
        for key in defaults.keys():
            val = opts[key] if key in opts.keys() else defaults[key]
            setattr(self, key, val)

    def init_arrays(self):
        """
        initialize arrays for computation
        Also used to bring them back to initial states for fitting.
        """
        # feature-cause-cooccurence for features being 1 and 0 respectively
        self.fcc1, self.fcc0 = [
            np.zeros((self.n_particles, self.max_cause, self.n_features), dtype=np.int)
            for _ in range(2)
        ]
        # counts of causes per particle
        self.cause_count = np.zeros((self.n_particles, self.max_cause), dtype=np.int)
        # posterior over causes per trial, summarizing particles
        self.posterior = np.c_[
            np.ones((self.n_trials, 1)), np.zeros((self.n_trials, self.max_cause - 1))
        ]
        # US predictions based on posterior
        self.pred_US = np.zeros((self.n_trials))
        # currect cause:
        self.current_cause = np.ones((self.n_particles), dtype=np.int)

    def sample(self):
        """
        Implement the algorithm, return results
        """
        posterior = np.zeros(self.max_cause)
        posterior[0] = 1
        posterior0 = np.c_[
            np.ones((self.n_particles, 1)),
            np.zeros((self.n_particles, self.max_cause - 1)),
        ]
        for t in range(self.n_trials):
            # initiate likelihood as cooccurence of causes with features being 1
            likelihood = self.fcc1.copy()
            # replace with cause-feature cooccurence of features being 0 for these features
            feat0 = np.where(self.features[t, :] == 0)
            likelihood[:, :, feat0] = self.fcc0[:, :, feat0]
            # likelihood of features given previous cause assignments
            likelihood = (
                (likelihood + self.beta_prior[0])
                / (self.cause_count + sum(self.beta_prior))
            )[:, :, np.newaxis]
            if self.alpha > 0:
                # prior for old cause mainly depends on cause counts
                # i.e. more frequent cause in the past is more likely a priori
                prior = self.cause_count.astype(np.float).copy()
                # stickiness makes current cause more likely
                if self.stickiness > 0:
                    prior[
                        range(self.n_particles), self.current_cause - 1
                    ] += self.stickiness
                # probability of new cause is proportional to alpha
                # As per Gershman's implementation, the first unused cause per particle
                # is the new cause. Not entirely sure if that's intended or a bug.
                for i in range(prior.shape[0]):
                    idx = np.where(prior[i, :] == 0)[0]
                    if idx.shape[0] > 0:
                        prior[i, idx[0]] = self.alpha
                # compute posterior of product of prior and likelihood
                posterior = prior * likelihood[:, :, 1:].prod(axis=2)
                # Normalize it without likelihood of US
                posterior0 = posterior / posterior.sum(axis=1, keepdims=True)
                # include likelihood of US
                posterior *= likelihood[:, :, 0]
                posterior /= posterior.sum(axis=1, keepdims=True)
                # marginalize, i.e. probability of different causes for present trial
                posterior = posterior.mean(axis=0)
                self.posterior[t, :] = posterior
            # compute probabity of US given cause assigments per cause and particle
            p_US = (self.fcc1[:, :, 0] + self.beta_prior[0]) / (
                self.cause_count + sum(self.beta_prior)
            )
            # weigh by probability of causes per particle
            self.pred_US[t] = (
                posterior0.flatten().dot(p_US.flatten()) / self.n_particles
            )
            # remember which features were 1 or 0 in this trial
            feat0 = np.where(self.features[t, :] == 0)[0]
            feat1 = np.where(self.features[t, :] == 1)[0]
            # sample new particles according to posterior probability
            # take MAP if only 1 particle
            if self.n_particles == 1:
                self.current_cause = posterior.argmax() + 1
            else:
                self.current_cause = (
                    np.array(
                        choices(
                            np.arange(self.max_cause),
                            posterior,
                            k=self.current_cause.size,
                        )
                    )
                    + 1
                )
            # update cause counts
            self.cause_count[np.arange(self.n_particles), self.current_cause - 1] += 1
            # update cause-feature-cooccurences.
            self.fcc1[
                np.arange(self.n_particles)[:, np.newaxis],
                self.current_cause[:, np.newaxis] - 1,
                feat1,
            ] += 1
            self.fcc0[
                np.arange(self.n_particles)[:, np.newaxis],
                self.current_cause[:, np.newaxis] - 1,
                feat0,
            ] += 1

        # delete unused causes
        self.posterior = self.posterior[:, self.posterior.sum(axis=0) != 0]


class LCM_gridsearch:
    """
    Use grid search to tune the alpha parameter of the Chinese restaurant prior
    This could be relatively easy adapted to also tune the beta prior on p(features|cause)
    Unfortunately this would take a while especially for independent priors per feature/cause
    """

    def __init__(self, CR, features, n_alpha=50, range_alpha=[0, 10], opts={}):
        """
        CR - vector, conditioned response of subject
        features - matrix, first col is US, rest are CS. MUST be binary.
        n_alpha - grid resolution for alpha
        range_alpha - min and max for alpha
        opts - potential options for Particle_filter. Alpha will be overwritten.
        """
        self.CR = CR
        self.features = features
        self.alpha = np.linspace(*range_alpha, n_alpha)
        self.opts = opts

    def loop_alpha(self):
        """
        Loops over alpha values, computes predicted CR and log likelihood
        """
        results = []
        latents = []
        opts = self.opts
        for a in self.alpha:
            opts["alpha"] = a
            p = Particle_filter(self.features, opts)
            p.sample()
            p_US = p.pred_US
            beta, pred = self.linreg(self.CR, p_US)
            ll = self.ll(self.CR, pred)
            results.append(
                {
                    "alpha": a,
                    "beta": beta,
                    "pred": pred,
                    "ll": ll,
                    "p_US": p_US,
                    "posterior": p.posterior,
                }
            )
        self.results = results

    def inference(self):
        """
        Make sense of the looped outout,
        i.e. compute the same quantities as Gershman:
            * expected value of alpha
            * posterior probability over alpha (assuming uniform prior on alpha)
            * logBF, full alpha range vs. alpha = 0;
        """
        # no samples - no inference
        if not hasattr(self, "results"):
            self.loop_alpha()
        # logsumexp, i.e. sum of likelihoods on log scale without leaving log scale
        LSE = self.logsumexp(self.like)
        # normalize posterior
        out = {}
        out["P"] = np.exp(self.like - LSE)
        out["alpha"] = self.alpha.dot(out["P"])
        # BF is average likelihood divided by likelihood for alpha.
        # this is the same on log scale.
        out["logBF"] = LSE - np.log(self.alpha.size) - self.like[0]
        return out

    @property
    def like(self):
        """
        return likelihoods for all alphas
        """
        return np.array([l["ll"] for l in self.results])[:, np.newaxis]

    @staticmethod
    def linreg(obs, p_US):
        """
        Compute the regression of observed responses onto p( US )
        where obs is the vector of observed responses and p_US is
        the vector of model implied US probabilities.

        returns (beta, predicted response)
        """
        obs = obs[:, np.newaxis] if obs.ndim == 1 else obs
        p_US = p_US[:, np.newaxis] if p_US.ndim == 1 else p_US
        # least square regression parameters by projection
        beta = np.linalg.inv(p_US.T @ p_US) * (p_US.T @ obs)
        # predict conditioned response based on these coefficients
        pred = p_US * beta
        return beta, pred

    @staticmethod
    def ll(obs, pred):
        """
        Compute log likelihood,
        assuming gaussian errors.
        """
        # compute SD of error of prediction
        sd = np.sqrt(np.power(obs - pred, 2).mean())
        # compute log likelihood
        ll = norm.logpdf(obs, pred, sd).sum()
        return ll

    @staticmethod
    def logsumexp(p, dim=0):
        """
        compute logsumexp for some vector of probabilities
        along first axis, i.e. rows by default
        is the same as log( sum( X ) ), except X is already in the log scale
        """
        p_max = p.max(axis=dim, keepdims=True)
        p -= p_max
        s = p_max + np.log(np.exp(p).sum(axis=dim))
        return s
