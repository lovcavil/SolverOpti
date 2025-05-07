
using DifferentialEquations
using CSV, DataFrames
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end
function l1()
    # Parameters
    σ = 10.0
    ρ = 28.0
    β = 8/3

    params = (σ, ρ, β)

    # Initial conditions
    u0 = [1.0, 1.0, 1.0]

    # Time span
    tspan = (0.0, 50.0)

    prob = ODEProblem(lorenz!, u0, tspan, params)
    sol = solve(prob,reltol = 1e-18  ,abstol = 1e-18)



    # Create a DataFrame
    df = DataFrame(t = sol.t, x = getindex.(sol.u, 1), y = getindex.(sol.u, 2), z = getindex.(sol.u, 3))

    # Write to CSV
    CSV.write("lorenz_solution.csv", df)
end
function l2()
    # Parameters
    σ = 20.0
    ρ = 38.0
    β = 8/3

    params = (σ, ρ, β)

    # Initial conditions
    u0 = [1.0, 3.0, 5.0]

    # Time span
    tspan = (0.0, 50.0)

    prob = ODEProblem(lorenz!, u0, tspan, params)
    sol = solve(prob,reltol = 1e-18  ,abstol = 1e-18)


    # Create a DataFrame
    df = DataFrame(t = sol.t, x = getindex.(sol.u, 1), y = getindex.(sol.u, 2), z = getindex.(sol.u, 3))

    # Write to CSV
    CSV.write("lorenz_solution2.csv", df)
end
l2()