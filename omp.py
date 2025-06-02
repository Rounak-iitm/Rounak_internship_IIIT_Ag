import numpy as np

def simple_omp(Phi, y, sparsity):
    n_atoms = Phi.shape[1]
    x_hat = np.zeros(n_atoms)
    residual = y.copy()
    chosen = []

    for _ in range(sparsity):
        # Find the column most correlated with the residual
        idx = np.argmax(np.abs(Phi.T @ residual))
        if idx in chosen:
            break  # Avoid picking the same atom again
        chosen.append(idx)
        # Solve least squares for selected columns
        Phi_sel = Phi[:, chosen]
        x_sel, *_ = np.linalg.lstsq(Phi_sel, y, rcond=None)
        # Update the estimate and residual
        x_hat[chosen] = x_sel
        residual = y - Phi_sel @ x_sel
    return x_hat

# Example usage
Phi = np.random.randn(10, 20)
x_true = np.zeros(20)
x_true[[3, 7, 12]] = np.random.randn(3)
y = Phi @ x_true

x_recovered = simple_omp(Phi, y, sparsity=3)
print("Original sparse signal:\n", x_true)
print("\nRecovered sparse signal:\n", x_recovered)
