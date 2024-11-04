### Implement the full quantum circuit for the Jacobi solver


using LinearAlgebra
using Yao
using YaoPlots


## Finite difference - solving the viscous Burgers equation
function viscous_burgers_equation(n, x_i=0, x_f=1, M=50, t_i=0, t_f=1, P=15, ν=0.1, bc_i=0, bc_f=0)
    """ Linearise the viscous Burgers's equation using the BTCS finite difference method. """
    # Identify the dimension of the problem to solve
    N = 2^n + 1
    # Error fractions
    Δx = (x_f-x_i)/N
    Δt = (t_f-t_i)/M
    λ = ν*(Δt/(Δx^2))
    r = Δt/(2*Δx)
    # Create mesh
    x = range(x_i, x_f, length=N+1)
    # Initialise solution
    u = zeros(N+1, P+1)
    b = zeros(N-1)
    # Initial conditions
    for i=2:N
        u[i,1] = sin(2*π*x[i])
    end
    # Boundary conditions
    for i=1:P+1
        u[1,i] = bc_i
        u[N+1,i] = bc_f
    end
    # Discretised derivative matrix
    A = zeros(N-1, N-1)
    A[1,1] = 1+2*λ+r*u[2,1]
    for i=2:N-1
        A[i,i]=1+2*λ+r*(u[i+1,1]-u[i-1,1])
    end
    for i=1:N-2
        A[i+1,i] = -r
        A[i,i+1] = -r
    end
    A_inv = inv(A)
    # Solve resulting Ax=b problem numerically at different t points
    intermediate_b = []
    # Account for the linear and quadratic boundary conditions in the construction of the b vector
    for j=2:P+1
        # Construct RHS
        b[1] = r*(u[2,j-1]+1)*u[1,j-1]
        b[N-1] = r*(u[N,j-1]+1)*u[N+1,j-1]
        rhs = u[2:N,j-1]+b
        push!(intermediate_b, rhs)
        # Calculate true solution
        u[2:N,j] = A_inv*rhs
        u[N+1,j] = u[N+1,j-1]
    end
    # Extract final solution after P time steps and subsequent problem components
    return A, intermediate_b[end], u[:,end][2:end-1]
end

## State preparation - oracle access to amplitude encoding
function state_preparation(v)
    """ Assume access to an oracle that prepares the state |v>=U_v|0>. """
    U = hcat(v, nullspace(v'))
    return matblock(U)
end

## Block-encoding - oracle access to block-encoding protocol
function block_encoding(A)
    """ Assume access to and oracle that pepares the block-encoding of matrix A. """
    U = [A sqrt(Matrix(I,2,2)-A'A) ; sqrt(Matrix(I,2,2)-A'A) -A]
    return matblock(U)
end

## Quantum Jacobi method - implement Jacobi iteration using expansive LCU approach
function prepare_unitary(k, x0const, bconst, invDconst, Rconst)
    """ Construct the preparation unitary for the LCU circuit. The unitary multiplies each controlled-unitary in the LCU by a constant. This accounts for the normalisation constant from the state preparations and block-encodings. """
    vec = [bconst*Rconst^(i-1)*invDconst^i for i in 1:k]
    append!(vec, [Rconst^k*invDconst^k*x0const])
    vec = [sqrt(v)/sqrt(sum(vec)) for v in vec]
    return state_preparation(vec)
end
function intermediate_expansion_term(U_p, U_q, no_p, no_q)
    """ Multiply the block-encodings of p and q to obtain the block-encoding of an intermediate expansion term. """
    # Determine the total number of qubits required for the block-encoding of the expansion term
    n_p = nqubits(U_p)
    n_q = nqubits(U_q)
    n_term = 1 + no_p*n_p + no_q*n_q - no_p - no_q
    # Compose the block-encoding
    term = chain(n_term, put(1:n_p=>U_p))
    c = n_p
    while c < n_term
        push!(term, put([1,c+1,c+n_q-1]=>U_q))
        c += n_q-1
    end
    return term
end
function final_expansion_term(U_0, U_q, no_0, no_q)
    """ Multiply the block-encodings of x0 and q to obtain the block-encoding of the final expansion term. """
    # Determine the total number of qubits required for the block-encoding of the expansion term
    n_0 = nqubits(U_0)
    n_q = nqubits(U_q)
    n_term = 1 + no_0*n_0 + no_q*n_q - no_0 - no_q
    # Compose the block-encoding
    term = chain(n_term, put(1:n_0=>U_0))
    c = n_0
    while c < n_term
        push!(term, put([1,c+1,c+n_q-1]=>U_q))
        c += n_q-1
    end
    return term
end


## Example problem
# Set up the classical problem (calculating x_k at k=3)
A, b, x = viscous_burgers_equation(1)
D = diagm(diag(A))
R = A - D
x0 = [0.3, 0.8]

# Block-encode the data
Rconst = norm(R)
U_R = block_encoding(normalize(R))
invDconst = norm(inv(D))
U_invD = block_encoding(normalize(inv(D)))
bconst = norm(b)
U_b = state_preparation(normalize(b))
x0const = norm(x0)
U_0 = state_preparation(normalize(x0))

# Multiply the block-encodings of the individual expansion terms
U_p = chain(2, put(1=>U_b), put(1:2=>U_invD))
U_q = chain(3, put([1,2]=>U_R), put([1,3]=>U_invD))

# Identify true classical solution for x3=p-qp+q^2p-q^3x0
k = 3
p = inv(D)*b
q = inv(D)*R
x3 = p - q*p + q^2*p - q^3*x0 |> normalize

# Block-encode expansion terms to form the multiplication circuits
qp = intermediate_expansion_term(U_p, U_q, 1, 1)
q2p = intermediate_expansion_term(U_p, U_q, 1, 2)
q3x0 = final_expansion_term(U_0, U_q, 1, 3)
V3 = prepare_unitary(k, x0const, bconst, invDconst, Rconst)

# Compose block-encoded expansive LCU circuit
n_terms = k + 1
n_a = Int64(log2(n_terms))
n = n_a + nqubits(q3x0)
lcu = chain(n, put(1:n_a=>V3), control([-1,-2],n_a+1:n_a+nqubits(U_p)=>U_p), control([1,-2],n_a+1:n_a+nqubits(qp)=>-qp), control([-1,2],n_a+1:n_a+nqubits(q2p)=>q2p), control([1,2],n_a+1:n_a+nqubits(q3x0)=>-q3x0), put(1:n_a=>V3'))

# Extract quantum solution for x3 via projective measurements
est_x_state = zero_state(n) |> lcu
project = 0.5*(igate(1)+Z)
for j = 1:n
    if j in 1:n_a || j in n_a+2:n
        est_x_state |> put(n, j=>project)
    end
end
est_x3 = [j for j in statevec(est_x_state) if j != 0] |> normalize

# Compare normalised solutions
error3 = 1 - abs.(normalize(x3)' * normalize(est_x3))^2
