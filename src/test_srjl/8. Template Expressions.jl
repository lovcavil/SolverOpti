using SymbolicRegression
using Random: rand, MersenneTwister
using MLJBase: machine, fit!, report
expression_spec = @template_spec(expressions=(f, g)) do x1, x2, x3
    f(x1, x2) + g(x2) - g(x3)
end
n = 100
rng = MersenneTwister(0)
x1 = 10rand(rng, n)
x2 = 10rand(rng, n)
x3 = 10rand(rng, n)
X = (; x1, x2, x3)
y = [
    2 * cos(x1[i] + 3.2) + x2[i]^2 - 0.8 * x3[i]^2
    for i in eachindex(x1)
]
model = SRRegressor(;
    binary_operators=(+, -, *, /),
    unary_operators=(cos,),
    niterations=500,
    maxsize=25,
    expression_spec=expression_spec,
)

mach = machine(model, X, y)
fit!(mach)
r = report(mach)
best_expr = r.equations[r.best_idx]

# Access individual parts of the template expression
println("f: ", get_contents(best_expr).f)
println("g: ", get_contents(best_expr).g)
best_expr(randn(3, 20))