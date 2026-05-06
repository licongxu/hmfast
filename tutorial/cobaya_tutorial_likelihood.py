"""
Gaussian tSZ D_ell likelihood (signal-only) for Cobaya tutorial.

Loads binned D_ell from a text file (ell, D_ell) and covariance from .npy or .txt.
Compatible with tszpower tszpower_cobaya_likelihood_masked.tSZ_PS_Likelihood_SignalOnly
but adds NumPy covariance loading.
"""

from __future__ import annotations

import os
import time

import numpy as np
from cobaya.likelihood import Likelihood


class tSZ_PS_Likelihood_FullSkyNPY(Likelihood):
    """
    Gaussian chi^2 for signal-only binned :math:`10^{12} D_\ell` (same convention as the
    scatter ``synthetic_data`` files and ``hmfast_cobaya_theory_fullsky`` theory output).

    data_file: two columns (ell_eff, D_binned), header lines starting with # allowed.
    cov_file:  NxN as .npy (saved with numpy.save) or text (loadtxt).
    """

    data_directory: str = "."
    data_file: str = "Dl_binned_fullsky_signal_only_real0.txt"
    cov_file: str = "covmat_Dl_binned_fullsky_signal_only.npy"
    # Like tszsbi: log chi2 / loglike every N evaluations (1 = every MCMC step; use 0 to disable).
    log_every: int = 1

    _n_eval: int = 0

    def initialize(self):
        data_path = os.path.join(self.data_directory, self.data_file)
        D = np.loadtxt(data_path)
        self.ell_data = D[:, 0]
        self.Dl_obs = D[:, 1]
        n = len(self.ell_data)

        cov_path = os.path.join(self.data_directory, self.cov_file)
        if cov_path.endswith(".npy"):
            C = np.load(cov_path)
        else:
            C = np.loadtxt(cov_path)
        C = np.atleast_2d(C)
        if C.shape != (n, n):
            raise ValueError(
                f"tSZ_PS_Likelihood_FullSkyNPY: cov shape {C.shape} != ({n},{n})"
            )

        self.covmat = C
        self.inv_covmat = np.linalg.inv(self.covmat)
        sign, logdet = np.linalg.slogdet(self.covmat)
        if sign <= 0:
            raise ValueError("Covariance not positive definite.")
        self.logdet_covmat = logdet

        self.log.info(
            "tSZ_PS_Likelihood_FullSkyNPY: loaded %d bins from %s; cov from %s",
            n,
            self.data_file,
            self.cov_file,
        )
        super().initialize()

    def get_requirements(self):
        return {"Cl_sz": {}}

    def logp(self, **params_values):
        # Wall time for likelihood only (get_Cl_sz + chi2); theory time is logged in HMFastTSZFullSky.calculate.
        t0 = time.perf_counter()
        theory_dict = self.provider.get_Cl_sz()
        Dl_th = np.asarray(theory_dict["1h"], dtype=float)
        if Dl_th.shape != self.Dl_obs.shape:
            raise ValueError(
                f"Theory D_l shape {Dl_th.shape} != data {self.Dl_obs.shape}"
            )

        resid = self.Dl_obs - Dl_th
        chi2 = float(resid @ self.inv_covmat @ resid)
        loglike = -0.5 * chi2
        dt_like = time.perf_counter() - t0

        self._n_eval += 1
        if self.log_every and (self._n_eval % self.log_every == 0):
            # mean(Dl_th) tracks whether the theory vector actually changes with parameters
            self.log.info(
                "tSZ_PS_Likelihood_FullSkyNPY: chi2 = %.2f, loglike = %.2f, mean(Dl_th) = %.6g, likelihood_eval = %.4f s",
                chi2,
                loglike,
                float(np.mean(Dl_th)),
                dt_like,
            )

        return loglike
