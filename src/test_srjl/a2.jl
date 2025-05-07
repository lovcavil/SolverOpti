using SymbolicRegression: Options, Expression, Node

options = Options(;
    binary_operators=[+, -, *, /], unary_operators=[cos, exp, sin]
)
operators = options.operators
variable_names = ["x1", "x2", "x3"]
x1, x2, x3 = [Expression(Node(Float64; feature=i); operators, variable_names) for i=1:3]

tree = cos(x1 - 3.2 * x2) - x1 * x1
X = rand(Float32, 3, 100)

println(tree(X))
