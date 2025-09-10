
# -*- coding: utf-8 -*-
"""
Critical Line Algorithm (CLA) con covarianza estimada vía CÓPULA GAUSSIANA.

Contiene:
- Estimador GaussianCopulaCovariance (Spearman → Pearson bajo cópula gaussiana).
- Optimizador estilo CLA (conjunto activo) para Markowitz con cotas l ≤ w ≤ u.
- Funciones de utilidad y un demo reproducible.

Autor: ChatGPT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict

def _make_psd_from_corr(R: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Asegura PSD de una correlación vía limpieza espectral y re-escalado a diag=1.
    """
    vals, vecs = np.linalg.eigh(R)
    vals_clipped = np.clip(vals, a_min=0.0, a_max=None)
    R_psd = (vecs @ np.diag(vals_clipped) @ vecs.T)
    d = np.sqrt(np.diag(R_psd))
    d[d < eps] = 1.0
    R_psd = R_psd / np.outer(d, d)
    R_psd = 0.5 * (R_psd + R_psd.T)
    np.fill_diagonal(R_psd, 1.0)
    return R_psd

class GaussianCopulaCovariance:
    """
    Estimación de Σ vía cópula gaussiana:
      1) ρ_Spearman ← corr_spearman(X)
      2) ρ_Pearson ≈ 2 sin( (π/6) ρ_Spearman )
      3) Σ = D ρ_Pearson D, con D=diag(std_muestral)
    """
    def __init__(self, ensure_psd: bool = True):
        self.ensure_psd = ensure_psd
        self.R_ = None
        self.sigma_ = None
        self.Sigma_ = None

    @staticmethod
    def spearman_to_gaussian_pearson(rho_s: np.ndarray) -> np.ndarray:
        return 2.0 * np.sin(np.pi * rho_s / 6.0)

    def fit(self, returns: pd.DataFrame) -> "GaussianCopulaCovariance":
        rho_s = returns.corr(method="spearman").values
        R = self.spearman_to_gaussian_pearson(rho_s)
        if self.ensure_psd:
            R = _make_psd_from_corr(R)
        sigma = returns.std(ddof=1).values
        D = np.diag(sigma)
        Sigma = D @ R @ D
        self.R_ = R
        self.sigma_ = sigma
        self.Sigma_ = Sigma
        return self

    def get_covariance(self) -> np.ndarray:
        if self.Sigma_ is None:
            raise ValueError("Llama .fit primero.")
        return self.Sigma_

    def get_correlation(self) -> np.ndarray:
        if self.R_ is None:
            raise ValueError("Llama .fit primero.")
        return self.R_

    def get_sigmas(self) -> np.ndarray:
        if self.sigma_ is None:
            raise ValueError("Llama .fit primero.")
        return self.sigma_

class CLAActiveSet:
    """
    Optimizador estilo CLA (conjunto activo) para:
      min  w' Σ w
      s.a. 1' w = 1, μ' w = r*, l ≤ w ≤ u

    Procedimiento:
      - Resolver KKT en el subespacio LIBRE; si algún w viola l/u → fijarlo y repetir.
      - Recorriendo r* en una grilla [r_min, r_max] se traza la frontera eficiente.
    """
    def __init__(self, mu: np.ndarray, Sigma: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                 tol: float = 1e-10, max_iter: int = 10_000):
        self.mu = np.asarray(mu).reshape(-1)
        self.Sigma = np.asarray(Sigma)
        self.lower = np.asarray(lower).reshape(-1)
        self.upper = np.asarray(upper).reshape(-1)
        self.N = self.mu.shape[0]
        self.tol = tol
        self.max_iter = max_iter
        assert self.Sigma.shape == (self.N, self.N)
        assert np.all(self.upper >= self.lower)
        self.ones = np.ones(self.N)

    @staticmethod
    def _solve_kkt(Sigma_ff, ones_f, mu_f, rhs1, rhs2, tol):
        k = Sigma_ff.shape[0]
        KKT = np.zeros((k + 2, k + 2))
        KKT[:k, :k] = Sigma_ff
        KKT[:k, k] = ones_f
        KKT[:k, k + 1] = mu_f
        KKT[k, :k] = ones_f
        KKT[k + 1, :k] = mu_f
        rhs_vec = np.zeros(k + 2)
        rhs_vec[k] = rhs1
        rhs_vec[k + 1] = rhs2
        try:
            sol = np.linalg.solve(KKT, rhs_vec)
        except np.linalg.LinAlgError:
            sol = np.linalg.pinv(KKT, rcond=tol) @ rhs_vec
        return sol[:k], sol[k], sol[k+1]

    def _bounded_equality_qp(self, r_target: float) -> np.ndarray:
        N = self.N
        free_idx = list(range(N))
        fixed_mask = np.zeros(N, dtype=bool)
        w = np.zeros(N)
        rhs1, rhs2 = 1.0, r_target

        for _ in range(self.max_iter):
            Sigma_ff = self.Sigma[np.ix_(free_idx, free_idx)]
            ones_f = self.ones[free_idx]
            mu_f = self.mu[free_idx]
            sum_fixed = w[fixed_mask].sum()
            ret_fixed = (w[fixed_mask] * self.mu[fixed_mask]).sum()
            rhs1_eff = rhs1 - sum_fixed
            rhs2_eff = rhs2 - ret_fixed
            if len(free_idx) == 0:
                w = np.clip(w, self.lower, self.upper)
                s = w.sum()
                if s != 0:
                    w = w / s
                return w
            w_f, _, _ = self._solve_kkt(Sigma_ff, ones_f, mu_f, rhs1_eff, rhs2_eff, self.tol)
            w[:] = 0.0
            w[fixed_mask] = w[fixed_mask]
            for il, ig in enumerate(free_idx):
                w[ig] = w_f[il]

            viol_low = [(i, self.lower[i] - w[i]) for i in free_idx if w[i] < self.lower[i] - self.tol]
            viol_up = [(i, w[i] - self.upper[i]) for i in free_idx if w[i] > self.upper[i] + self.tol]
            if not viol_low and not viol_up:
                w = np.clip(w, self.lower, self.upper)
                s = w.sum()
                if abs(s - 1.0) > 1e-10 and len(free_idx) > 0:
                    scale = (1.0 - w.sum()) / (w[free_idx].sum() + 1e-16)
                    w[free_idx] *= (1.0 + scale)
                return w

            cand = []
            if viol_low:
                j, mag = sorted(viol_low, key=lambda x: -abs(x[1]))[0]
                cand.append((j, "low", abs(mag)))
            if viol_up:
                j, mag = sorted(viol_up, key=lambda x: -abs(x[1]))[0]
                cand.append((j, "up", abs(mag)))
            j, side, _ = sorted(cand, key=lambda x: -x[2])[0]
            w[j] = self.lower[j] if side == "low" else self.upper[j]
            fixed_mask[j] = True
            if j in free_idx:
                free_idx.remove(j)

        w = np.clip(w, self.lower, self.upper)
        s = w.sum()
        if s != 0:
            w = w / s
        return w

    def _feasible_return_range(self) -> Tuple[float, float]:
        order_min = np.argsort(self.mu)
        order_max = order_min[::-1]
        def fill(order):
            remaining, r_acc = 1.0, 0.0
            for i in order:
                w_i = min(self.upper[i], remaining)
                w_i = max(self.lower[i], min(w_i, remaining))
                if w_i < self.lower[i]:
                    w_i = max(0.0, min(self.lower[i], remaining))
                r_acc += w_i * self.mu[i]
                remaining -= w_i
                if remaining <= 1e-12:
                    break
            return r_acc
        return fill(order_min), fill(order_max)

    def efficient_frontier(self, num_points: int = 40) -> Dict[str, np.ndarray]:
        r_min, r_max = self._feasible_return_range()
        rt = np.linspace(r_min + 1e-8, r_max - 1e-8, num_points)
        W, risks = [], []
        for r_target in rt:
            w = self._bounded_equality_qp(r_target)
            W.append(w)
            risks.append(np.sqrt(w @ self.Sigma @ w))
        return {"returns": rt, "risks": np.array(risks), "weights": np.vstack(W)}

def random_spd_correlation(N: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(N, N))
    M = B @ B.T + N * 1e-3 * np.eye(N)
    d = np.sqrt(np.diag(M))
    R = M / np.outer(d, d)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R

def demo(seed: int = 11, T: int = 1200, N: int = 8, num_points: int = 40):
    rng = np.random.default_rng(seed)
    R_true = random_spd_correlation(N, seed)
    sig_true = rng.uniform(0.10, 0.30, size=N)
    Sigma_true = np.diag(sig_true) @ R_true @ np.diag(sig_true)
    mu_true = rng.uniform(0.05, 0.15, size=N)
    scale = 1.0/250.0
    mean_vec = mu_true * scale
    try:
        L = np.linalg.cholesky(Sigma_true * scale)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(Sigma_true * scale)
        vals = np.clip(vals, 1e-12, None)
        L = vecs @ np.diag(np.sqrt(vals))
    Z = rng.normal(size=(T, N))
    X = Z @ L.T + mean_vec
    df = pd.DataFrame(X, columns=[f"A{i+1}" for i in range(N)])
    Sigma_hat = GaussianCopulaCovariance(True).fit(df).get_covariance()
    mu_hat = df.mean().values
    lower = np.zeros(N)
    upper = np.full(N, 0.6)
    cla = CLAActiveSet(mu_hat, Sigma_hat, lower, upper)
    frontier = cla.efficient_frontier(num_points)
    plt.figure(figsize=(6,4))
    plt.plot(frontier["risks"], frontier["returns"], marker="o", linestyle="-")
    plt.xlabel("Riesgo (σ)"); plt.ylabel("Retorno (μ)")
    plt.title("Frontera eficiente (CLA + cópula gaussiana)")
    plt.grid(True); plt.show()
    return frontier

if __name__ == "__main__":
    demo()
