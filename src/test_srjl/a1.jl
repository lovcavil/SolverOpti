import SymbolicRegression: Options, equation_search
using SymbolicUtils
X = randn(2, 100)
y = 2 * cos.(X[2, :]) + X[1, :] .^ 2 .- 2

options = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[cos, exp],
    populations=30
)

hall_of_fame = equation_search(
    X, y, niterations=100, options=options,
    parallelism=:multithreading
)
import SymbolicRegression: calculate_pareto_frontier

dominating = calculate_pareto_frontier(hall_of_fame)
trees = [member.tree for member in dominating]
tree = trees[end]
output = tree(X)
import SymbolicRegression: node_to_symbolic

eqn = node_to_symbolic(dominating[end].tree)
println(simplify(eqn))
import SymbolicRegression: compute_complexity, string_tree

#println("Complexity\tMSE\tEquation")

for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)

    #println("$(complexity)\t$(loss)\t$(string)")
end