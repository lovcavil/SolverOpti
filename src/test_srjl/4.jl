using SymbolicRegression
using Test
using Statistics
# —— 1. 构造数据 —— 
const n      = 500
const c1     = rand(n) .* 2 .+ 1
const c2     = rand(n) .* 10 .+ 10
const x_vals = collect(range(0,1;length=100))

# 把每条样本的曲线存成一个向量，放到一个长度-n 的 Vector 里
const curves = [
    @. c1[i]*cos(x_vals*c2[i]) - x_vals^2
    for i in 1:n
]  # Vector{Vector{Float64}}，长度 n，每个元素长度 100

# X 仍然保持 2×n
X = hcat(c1, c2)'    # 2×500

# —— 2. 在主进程和 worker 上定义自定义损失 —— 
defs = quote
  using SymbolicRegression
  using Statistics   # <- 加这一行
  const x_vals = $(x_vals)
  function my_curve_loss(
      ex::Expression,
      dataset::Dataset,
      options::Options
  )
      Xp     = dataset.X          # 2×n
      curves = dataset.y          # Vector{Vector}, length n

      n = length(curves)
      total = zero(eltype(curves[1]))

      for i in 1:n
          Y_true_i = curves[i]           
          m = length(Y_true_i)
          c1i, c2i = Xp[1,i], Xp[2,i]
          feat = vcat(
            fill(c1i, m)', 
            fill(c2i, m)', 
            x_vals'
          )
          pred, ok = eval_tree_array(ex, feat, options)
          if !ok
              return Inf
          end
          total += mean(abs.(pred .- Y_true_i))  # 现在 mean 已可用
      end

      return total / n
  end
end

if (@__MODULE__) != Core.Main
    Core.eval(Core.Main, defs)
    @eval Main: my_curve_loss
else
    eval(defs)
end

# —— 3. 配置并运行搜索 —— 
options = Options(
    binary_operators         = (+, -, *),
    unary_operators          = (cos,),
    loss_function_expression = my_curve_loss,
    maxsize                  = 30,
)

hof = equation_search(
    X,
    curves;                   # 传入 Vector{Vector} 作为 dataset.y
    options     = options,
    niterations  = 100,
    parallelism  = :multiprocessing,
    numprocs     = 2,
)

# —— 4. 打印最优表达式 —— 
best = hof.members[argmin(m->m.loss, hof.members[hof.exists])]
println("最佳表达式: ", best.tree)
