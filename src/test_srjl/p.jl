using Plots
const n      = 500
const c1     = rand(n) .* 2 .+ 1
const c2     = rand(n) .* 10 .+ 10
const x_vals = collect(range(0,1;length=100))

# 将每条样本的曲线存成一个向量，放到长度-n 的 Vector 里
const curves = [
    @. c1[i]*cos(x_vals*c2[i]) - x_vals^2
    for i in 1:n
]  # Vector{Vector{Float64}}，长度 n，每个元素长度 100
# Prepare a new plot
plot(title="First 5 Curves", xlabel="x", ylabel="y")

# Overlay the first five curves
for i in 1:5
    plot!(x_vals, curves[i], label="curve $i")
end

# Display
display(current())
