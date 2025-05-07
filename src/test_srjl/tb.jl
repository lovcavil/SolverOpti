# 检查TensorBoardLogger版本
using Pkg
Pkg.status("TensorBoardLogger")

# 检查Python环境
using PyCall
println(PyCall.python)
println(pyimport("tensorboard").__version__)