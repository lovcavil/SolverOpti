using SymbolicRegression
using TensorBoardLogger
using MLJ
using MLJ:  fit!

logger = SRLogger(TBLogger("logs/sr_run"))

# Create and fit model with logger
model = SRRegressor(
    binary_operators=[+, -, *],
    maxsize=40,
    niterations=100,
    logger=logger
)

X = (a=rand(500), b=rand(500))
y = @. 2 * cos(X.a * 23.5) - X.b^2

mach = machine(model, X, y)
fit!(mach)
