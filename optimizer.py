import numpy as np

class SPSA:
    def __init__(self, theta0, a=0.1, c=0.1, alpha=0.602, gamma=0.101, A=10, bounds=None):
        self.theta = np.array(theta0, dtype=float)
        self.a, self.c, self.alpha, self.gamma, self.A = a, c, alpha, gamma, A
        self.k = 0
        self.bounds = bounds  # (lower, upper) or None

    def step(self, eval_fn):
        self.k += 1
        ak = self.a / (self.k + self.A) ** self.alpha
        ck = self.c / (self.k) ** self.gamma

        # Random perturbation (±1)
        delta = 2 * (np.random.rand(*self.theta.shape) > 0.5) - 1  

        try:
            loss_plus = eval_fn(self.theta + ck * delta)
            loss_minus = eval_fn(self.theta - ck * delta)
            if not np.isfinite(loss_plus) or not np.isfinite(loss_minus):
                raise ValueError("Non-finite loss encountered.")
        except Exception as e:
            print(f"[SPSA] Eval failed: {e}")
            return self.theta, None, None  # safe exit

        # Gradient estimate (element-wise division)
        ghat = (loss_plus - loss_minus) / (2.0 * ck * delta)

        # Update
        self.theta -= ak * ghat

        # Optional: bounds
        if self.bounds:
            lower, upper = self.bounds
            self.theta = np.clip(self.theta, lower, upper)

        return self.theta, None, None
