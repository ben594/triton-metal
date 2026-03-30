import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDAOptions
from triton.compiler.compiler import ASTSource
from triton.backends.metal.compiler import MetalBackend
from triton.backends.metal.msl_generator import MSLGenerator


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


context = ir.context()
ir.load_dialects(context)

target = GPUTarget("metal", 0, 32)  # dummy
options = CUDAOptions()  # dummy

src = ASTSource(
    fn=add_kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
    constexprs={"BLOCK_SIZE": 1024},
)

mod = src.make_ir(target, options, codegen_fns={}, module_map={}, context=context)

metadata = {}
mod = MetalBackend.make_ttir(mod, metadata, options)
MSLGenerator.generate(mod, metadata, options)
