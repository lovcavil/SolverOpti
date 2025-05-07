import SymbolicRegression: SRRegressor
import MLJ: machine, fit!, predict, report

X = 2randn(1000, 5)

y = @. 1/X[:, 1]+1+sin(X[:, 2])

my_inv(x) = 1/x

model = SRRegressor(
    binary_operators=[+, *],
    unary_operators=[my_inv,sin,cos],
)
mach = machine(model, X, y)
fit!(mach)
r = report(mach)
println(r.equations[r.best_idx])