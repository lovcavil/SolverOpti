using SymbolicRegression
using Random
using MLJ:  fit!
rng = MersenneTwister(42)
x = 1 .+ rand(rng, 1000) * 9  # Sampling points in the range [1, 10]
y = @. 1 / (x^2 * sqrt(x^2 - 1))  # Values of the integrand
using SymbolicRegression: D

expression_spec = @template_spec(expressions=(f,)) do x
    D(f, 1)(x)
end
using MLJ

model = SRRegressor(
    binary_operators=(+, -, *, /),
    unary_operators=(sqrt,),
    maxsize=20,
    expression_spec=expression_spec,
)

X = (; x=x)
mach = machine(model, X, y)
fit!(mach)
r = report(mach)
best_expr = r.equations[r.best_idx]

println("Learned expression: ", best_expr)