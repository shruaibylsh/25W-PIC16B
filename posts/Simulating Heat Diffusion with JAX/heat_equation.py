import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse

# part 1: matrix multiplication

# matrix vector multiplication
def advance_time_matvecmul(A, u, epsilon):
    """Advances the simulation by one timestep, via matrix-vector multiplication
    Args:
        A: The 2d finite difference matrix, N^2 x N^2. 
        u: N x N grid state at timestep k.
        epsilon: stability constant.

    Returns:
        N x N Grid state at timestep k+1.
    """
    N = u.shape[0]
    u = u + epsilon * (A @ u.flatten()).reshape((N, N))
    return u

# get_A(N) function
def get_A(N):
    """
    constructs the 2d finite difference matrix for heat diffusion simulation
    argument: N: grid size (N*N)
    return: A: the N^2 * N^2 finite difference matrix
    """
    
    n = N * N
    diagonals = [-4 * np.ones(n), np.ones(n-1), np.ones(n-1), np.ones(n-N), np.ones(n-N)]
    diagonals[1][(N-1)::N] = 0
    diagonals[2][(N-1)::N] = 0
    
    # Construct matrix A
    A = np.diag(diagonals[0])                 # Main diagonal (-4)
    A += np.diag(diagonals[1], 1)             # Right neighbor
    A += np.diag(diagonals[2], -1)            # Left neighbor
    A += np.diag(diagonals[3], N)             # Bottom neighbor
    A += np.diag(diagonals[4], -N)            # Top neighbor
    
    return A


# part 2: Simulation with Sparse Matrix in JAX

# create sparse matrix A in BCOO format
def get_sparse_A(N):
    """
    args:
        N: grid size (N*N)
    return:
        A_sp_matrix: matrix A in sparse (BCOO) format
    """
    n = N * N
    rows, cols, values = [], [], []

    # Main diagonal: -4 for each point
    for i in range(n):
        rows.append(i)
        cols.append(i)
        values.append(-4.0)

    # Right neighbors
    for i in range(n - 1):
        if (i + 1) % N != 0:  # Skip right boundary
            rows.append(i)
            cols.append(i + 1)
            values.append(1.0)

    # Left neighbors
    for i in range(1, n):
        if i % N != 0:  # Skip left boundary
            rows.append(i)
            cols.append(i - 1)
            values.append(1.0)

    # Bottom neighbors
    for i in range(n - N):
        rows.append(i)
        cols.append(i + N)
        values.append(1.0)

    # Top neighbors
    for i in range(N, n):
        rows.append(i)
        cols.append(i - N)
        values.append(1.0)

    # Convert to BCOO format
    indices = np.column_stack((rows, cols))
    A_sp_matrix = sparse.BCOO((np.array(values), indices), shape=(n, n))
    return A_sp_matrix


# git-ed version of advance_time_matvecmul
@jax.jit
def advance_time_matvecmul_sparse(A_sp, u_flat, epsilon):
    """
    args:
        A_sp: the sparse 2d finite difference matrix
        u_flat: the flattened NxN grid state at timestep k
        epsilon: stability constant
    returns:
        N x N Grid state at timestep k+1.
    """
    return u_flat + epsilon * (A_sp @ u_flat)



# part 3: Simulation with NumPy

def advance_time_numpy(u, epsilon):
    """
    Advances the solution by one timestep using vectorized array operations.
    
    Args:
        u: N x N grid state at timestep k.
        epsilon: Stability constant.
        
    Returns:
        N x N grid state at timestep k+1.
    """
    # Pad the grid with zeros to handle boundaries
    padded = np.pad(u, 1, mode='constant', constant_values=0)
    
    # Compute the finite differences using np.roll()
    u_new = u + epsilon * (
        np.roll(padded, 1, axis=0)[1:-1, 1:-1] +  # Top neighbor
        np.roll(padded, -1, axis=0)[1:-1, 1:-1] +  # Bottom neighbor
        np.roll(padded, 1, axis=1)[1:-1, 1:-1] +   # Left neighbor
        np.roll(padded, -1, axis=1)[1:-1, 1:-1] -  # Right neighbor
        4 * u                                       # Center
    )
    return u_new



# part 4: Simulation with JAX

@jax.jit  # JIT-compile this function for performance
def advance_time_jax(u, epsilon):
    """
    Advances the solution by one timestep using JAX and JIT compilation.
    
    Args:
        u: N x N grid state at timestep k (as a JAX array).
        epsilon: Stability constant.
        
    Returns:
        N x N grid state at timestep k+1.
    """
    # Pad the grid with zeros to handle boundaries
    padded = jnp.pad(u, 1, mode='constant', constant_values=0)
    
    # Compute the finite differences using jnp.roll()
    u_new = u + epsilon * (
        jnp.roll(padded, 1, axis=0)[1:-1, 1:-1] +  # Top neighbor
        jnp.roll(padded, -1, axis=0)[1:-1, 1:-1] +  # Bottom neighbor
        jnp.roll(padded, 1, axis=1)[1:-1, 1:-1] +   # Left neighbor
        jnp.roll(padded, -1, axis=1)[1:-1, 1:-1] -  # Right neighbor
        4 * u                                        # Center
    )
    return u_new