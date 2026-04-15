#pragma once

// Metal kernel arg layout
//
// Metal kernels have extra args appended in this order.
// Two pointer args come from amendFuncOp (Triton core), and four i32
// args are appended by Metal's FuncOpToLLVM:
//
//   numArgs - 6 : scratch ptr 1  (ptr, globalPtrTy from amendFuncOp)
//   numArgs - 5 : scratch ptr 2  (ptr, profilePtrTy from amendFuncOp)
//   numArgs - 4 : num_programs   (i32, air.threadgroups_per_grid)
//   numArgs - 3 : thread_idx     (i32, air.thread_position_in_grid)
//   numArgs - 2 : simdgroup_idx  (i32, air.simdgroup_index_in_threadgroup)
//   numArgs - 1 : threadgroup_idx (i32, air.threadgroup_position_in_grid)

namespace mlir::triton::metal {

// Total extra (non-user) args appended to Metal kernel
constexpr int kNumExtraArgs = 6;

// Offsets from END of arg list
constexpr int kThreadgroupIdxFromEnd = 1;
constexpr int kSimdgroupIdxFromEnd = 2;
constexpr int kThreadIdxFromEnd = 3;
constexpr int kNumProgramsFromEnd = 4;
constexpr int kScratchPtr2FromEnd = 5;
constexpr int kScratchPtr1FromEnd = 6;

// Metal-specific i32 value args appended by FuncOpToLLVM
constexpr int kNumI32ExtraArgs = 4;

} // namespace mlir::triton::metal
