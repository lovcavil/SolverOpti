using SymbolicRegression

# 参数设置
n = 500
coeff1 = rand(n) .* 2 .+ 1
coeff2 = rand(n) .* 10 .+ 10
X = hcat(coeff1, coeff2)  # n × 2

# 生成 y 曲线
x_vals = collect(range(0, 1; length=100))
function generate_y(c1, c2, x)
    @. c1 * cos(x * c2) - x^2
end
curves = [generate_y(c1, c2, x_vals) for (c1, c2) in zip(coeff1, coeff2)]
Y = reduce(hcat, curves)  # n × 100，✅ 正确
@show size(X), size(Y)     # (500, 2), (500, 100)

# 自定义损失函数
function my_loss(ytrue::Matrix{Float64}, ŷ::Vector{Float64})
    y_pred = [@. ŷ[i] * cos(x_vals * X[i,2]) - x_vals^2 for i in 1:length(ŷ)]
    Yp = reduce(hcat, y_pred)'
    return mean(abs.(Yp .- ytrue))
end

# 设置回归器
options = SymbolicRegression.Options(
    binary_operators=(+, -, *),
    unary_operators=(cos,),
    loss_function=my_loss,
    maxsize=30
)

hall_of_fame = EquationSearch(X, Y;
    options=options,
    niterations=50,
)
