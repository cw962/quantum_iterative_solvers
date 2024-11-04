### Solve the Burgers equation using the Jacobi method and compare the error scaling between the classical and quantum Jacobi implementations


using LinearAlgebra
using Statistics
using Yao, YaoBlocks
using Plots


function state_preparation(v)
    """ Assume access to an oracle that prepares the state |v>=U_v|0>. """
    U = hcat(v, nullspace(v'))
    return matblock(U)
end

function iterative_jacobi(x0, b, R, D, true_x, n_iterations)
    """ Implement the quantum Jacobi method using an iterative LCU approach. Note that objects are assumed to be unitary (apart from U_xi when initialised with xi after each loop). """
    n = Int64(log2(size(D)[1]))
    n_tot = n + 2
    # Quantum objects
    U_b = state_preparation(b)
    U_R = matblock([R sqrt(Matrix(I,2^n,2^n)-R'R) ; sqrt(Matrix(I,2^n,2^n)-R'R) -R])
    rho = matblock(inv(D))
    # Jacobi iteration via iterative LCU loop
    project = matblock([1,0]*[1,0]')
    errors = Vector{Float64}()
    x_iterates = Vector{Vector{ComplexF64}}()
    push!(x_iterates, x0/2)
    k = 0
    while k < n_iterations
        xi = x_iterates[end]
        U_xi = state_preparation(2 * xi)
        lcu_block(U_xi) = chain(n_tot, put(1=>H), control(1,2:n_tot-1=>-U_xi), control(1,2:n_tot=>U_R), control(-1,2:n_tot-1=>U_b), put(2:n_tot-1=>rho), put(1=>H))
        est_x = statevec(zero_state(n_tot) |> lcu_block(U_xi) |> put(n_tot, 1=>project) |> put(n_tot, n_tot=>project))
        est_x = [j for j in est_x if j != 0]
        push!(x_iterates, est_x)
        error = 1 - abs.(normalize(true_x)' * normalize(est_x))^2
        push!(errors, error)
        k += 1
    end
    # Extract solution
    est_x = 2 * x_iterates[end]
    return est_x, errors
end

function classical_jacobi(x0, b, R, D, true_x, n_iterations)
    """ Implement the classical Jacobi iterative solver and caluclate the RMSE from each iteration. """
    x_iterates = Vector{Vector{ComplexF64}}()
    push!(x_iterates, x0)
    errors = Vector{Float64}()
    k = 0
    while k < n_iterations
        xi = x_iterates[end]
        x = inv(D) * (b - R*xi)
        push!(x_iterates, x)
        error = 1 - abs.(normalize(true_x)' * normalize(x))^2
        push!(errors, abs.(error))
        k += 1
    end
    est_x = x_iterates[end]
    return est_x, errors
end

function viscous_burgers_equation(N, M, n_iterations, x_i=0, x_f=1, t_i=0, t_f=0.5, ν=0.08)
    """ Solve the viscous Burgers's equation using the BTCS finite difference method and compare the error scaling across different Jacobi iterative solvers. """
    # Error fractions
    Δx = (x_f-x_i)/N
    Δt = (t_f-t_i)/M
    λ = ν*(Δt/(Δx^2))
    r = Δt/(2*Δx)
    # Create mesh
    t = collect(t_i:Δt:t_f)
    x = range(x_i, x_f, length=N+1)
    # Initialise solution
    u = zeros(N+1, M+1)
    b = zeros(N-1)
    # Initial conditions
    for i = 2:N
        u[i,1] = sin(2π*x[i])
    end
    # Dynamic boundary conditions
    for j = 1:M+1
        u[1,j] = -t[j]
        u[N+1,j] = t[j]
    end
    # Solve resulting Ax=b problem numerically at different t points
    classical_u = []
    classical_errors = []
    quantum_u = []
    quantum_errors = []
    # Time stepping loop
    for j = 2:M+1
        # Calculate discretised derivative matrix
        A = zeros(N-1, N-1)
        A[1,1] = 1+2*λ+r*u[2,j-1]
        for i = 2:N-1
            A[i,i] = 1+2*λ+r*(u[i+1,j-1]-u[i-1,j-1])
        end
        for i = 1:N-2
            A[i+1,i] = -λ
            A[i,i+1] = -λ
        end
        # Calculate b
        b[1] = λ*u[1,j-1] + r*u[1,j-1]*u[2,j-1]
        b[N-1] = λ*u[N+1,j-1] - r*u[N,j-1]*u[N+1,j-1]
        rhs = u[2:N,j-1]+b
        # True solution
        u[2:N,j] = inv(A)*rhs
        # Jacobi solution
        D = diagm(diag(A))
        R = A - D
        true_x = inv(A)*rhs
        # Classical Jacobi solution
        classical_sol, classical_errs = classical_jacobi(ones(N-1), rhs, R, D, true_x, n_iterations)
        classical_sol = vcat(-t[j], classical_sol, t[j])
        push!(classical_u, classical_sol)
        push!(classical_errors, classical_errs)
        # Quantum Jacobi solution
        quantum_sol, quantum_errs = iterative_jacobi(ones(N-1), rhs, R, D, true_x, n_iterations)
        quantum_sol = vcat(-t[j], quantum_sol, t[j])
        push!(quantum_u, quantum_sol)
        push!(quantum_errors, quantum_errs)
    end
    # Visualise results on 3D grid
    display(Plots.plot(t[2:M+1], x, u[:,2:M+1], st=:surface, camera=(40, 45), xlabel="\$t\$", ylabel="\$x\$", zlabel="\$u_{true}(x,t)\$", thickness_scaling = 1.3))
    display(Plots.plot(t[2:M+1], x, hcat(classical_u...)|>real, st=:surface, camera=(40, 45), xlabel="\$t\$", ylabel="\$x\$", zlabel="\$u_{classical}(x,t)\$", thickness_scaling = 1.3))
    display(Plots.plot(t[2:M+1], x, hcat(quantum_u...)|>real, st=:surface, camera=(40, 45), xlabel="\$t\$", ylabel="\$x\$", zlabel="\$u_{quantum}(x,t)\$", thickness_scaling = 1.3))
    # Visualise average Jacobi errors
    avg_classical_errors = mean(classical_errors)
    avg_quantum_errors = mean(quantum_errors)
    fig = Plots.plot(avg_classical_errors, xlabel="Iteration", ylabel="Average error", yaxis=:log, label="classical", thickness_scaling = 1.4)
    Plots.plot!(fig, avg_quantum_errors, label="quantum", linestyle=:dash)
    display(fig)
end


viscous_burgers_equation(65, 50, 40)
