# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg
from buddy.compiler.graph.transform import simply_fuse, apply_classic_fusion

def silu_pattern(x):
    sigmoid_x = torch.sigmoid(x)
    return torch.mul(x, sigmoid_x)

def foo(x):
    return silu_pattern(x)

x = torch.ones([4, 4], dtype=torch.float32)
# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
)

graphs = dynamo_compiler.importer(foo, x)
assert len(graphs) == 1
graph = graphs[0]
pattern_list = [apply_classic_fusion]
graphs[0].fuse_ops(pattern_list)

graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tensor.empty() : tensor<4x4xf32>
# CHECK: %{{.*}} = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<4x4xf32>) outs(%0 : tensor<4x4xf32>) {
# CHECK: ^bb0(%in: f32, %out: f32):
# CHECK: %{{.*}} = arith.negf %in : f32
# CHECK: %{{.*}} = math.exp %2 : f32
# CHECK: %{{.*}} = arith.constant 1.000000e+00 : f32
# CHECK: %{{.*}} = arith.addf %4, %3 : f32
# CHECK: %{{.*}} = arith.divf %in, %5 : f32
# CHECK: linalg.yield %6 : f32
# CHECK: } -> tensor<4x4xf32>
# CHECK: return %1 : tensor<4x4xf32>
# CHECK: }
# CHECK: }
