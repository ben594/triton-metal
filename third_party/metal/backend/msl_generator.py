from typing import Any


class MSLValue:
    def __init__(self, name: str, val_type: str, val: Any):
        self.name = name
        self.val_type = val_type
        self.val = val


class MSLOp:
    def __init__(self, op_type: str, operands: list[MSLValue], results: list[MSLValue]):
        self.op_type = op_type
        self.operands = operands
        self.results = results

    def __repr__(self):
        operand_str = ", ".join([f"{var.val_type} {var.name}" for var in self.operands])
        result_str = ", ".join([f"{var.val_type} {var.name}" for var in self.results])
        return f"{self.op_type}({operand_str}) -> ({result_str})"


class MSLKernel:
    def __init__(self, name: str):
        self.name = name
        self.msl_ops: list[MSLOp] = []

    def init_from_module(self, mod):
        # walk module
        ops = []
        mod.walk(lambda op: ops.append(op))
        for i, op in enumerate(ops):
            msl_op = self.process_op(op)
            self.msl_ops.append(msl_op)

        return ops

    def process_op(self, op) -> MSLOp:
        name = op.get_name()
        method_name = f"process_{name}".replace(".", "_")
        handler = getattr(self, method_name, self.process_unknown_op)
        msl_op = handler(op)
        
        print("Processed op:", msl_op)

        return msl_op

    def process_arith_constant(self, op) -> MSLOp:
        op_type = "arith.constant"
        const_val = op.get_int_attr("value")
        assert op.get_num_results() == 1
        result_type = str(op.get_result(0).get_type())
        result_val = MSLValue(f"const_{const_val}", result_type, const_val)
        return MSLOp(op_type, [], [result_val])
    
    def process_tt_get_program_id(self, op) -> MSLOp:
        op_type = "tt.get_program_id"
        axis = op.get_int_attr("axis")
        assert op.get_num_results() == 1
        result_type = str(op.get_result(0).get_type())
        result_val = MSLValue(f"pid_axis_{axis}", result_type, None)
        return MSLOp(op_type, [], [result_val])

    def process_unknown_op(self, op) -> MSLOp:
        raise NotImplementedError(f"Processing for op {op.get_name()} is not implemented yet")


class MSLGenerator:
    def __init__(self):
        pass

    @staticmethod
    def generate(mod, metadata, options):
        print("Generating MSL code...")
        kernel = MSLKernel("my_kernel")
        ops = kernel.init_from_module(mod)
        return ops
