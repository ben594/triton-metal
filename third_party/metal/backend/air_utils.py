import re


def rewrite_ptrs(ir: str) -> str:
    # prev: getelementptr float, ptr addrspace(1) %0, i64 %9
    # new: getelementptr float, float addrspace(1)* %0, i64, %9
    ir = re.sub(
        r"(getelementptr(?:\s+inbounds)?\s+(<\d+\s+x\s+\w+>|\w+)),\s+ptr\s+addrspace\((\d+)\)",
        lambda m: f"{m.group(1)}, {m.group(2)} addrspace({m.group(3)})*",
        ir,
    )

    # prev: load float, ptr addrspace(1) %10
    # new: load float, float addrspace(1)* %10
    ir = re.sub(
        r"(load\s+(<\d+\s+x\s+\w+>|\w+)),\s+ptr\s+addrspace\((\d+)\)",
        lambda m: f"{m.group(1)}, {m.group(2)} addrspace({m.group(3)})*",
        ir,
    )

    # prev: store float %10, ptr addrspace(1) %15
    # new: store float %10, float addrspace(1)* %15
    ir = re.sub(
        r"(store\s+(<\d+\s+x\s+\w+>|\w+)\s+[^,\n]+),\s+ptr\s+addrspace\((\d+)\)",
        lambda m: f"{m.group(1)}, {m.group(2)} addrspace({m.group(3)})*",
        ir,
    )

    return ir


def generate_getelementptr_type_dict(ir: str) -> dict:
    getelementptr_type_dict: dict[
        str, tuple[str, str]
    ] = {}  # map ptr name (result of getelementptr) to ptr type and addr space
    for gep_match in re.finditer(
        r"(\s*)(%\w+)\s*=\s*getelementptr\b[^,\n]*,\s+(<\d+\s+x\s+\w+>|\w+)\s+addrspace\((\d+)\)\*",
        ir,
    ):
        # group(1): leading spaces
        # group(2): ptr var name, result of getelementptr
        # group(3): ptr type
        # group(4): addr space
        ptr_var_name = gep_match.group(2)
        ptr_type = gep_match.group(3)
        addrspace = gep_match.group(4)
        getelementptr_type_dict[ptr_var_name] = (ptr_type, addrspace)

    return getelementptr_type_dict


def insert_bitcast_for_vecs(ir: str, getelementptr_type_dict: dict, struct_type_dict: dict) -> str:
    breakpoint()
    # insert bitcast when loading vector from scalar ptr that is result of getelementptr
    # e.g.
    # %54 = getelementptr float, float addrspace(1)* %52, i64 %53
    # %62 = load <1 x float>, <1 x float> addrspace(1)* %54, align 4

    # get type of @global_smem
    global_smem_definition_match = re.search(
        r"@global_smem\s*=\s*\w+\s+addrspace\(\d+\)\s+global\s+(\[\d+\s+x\s+\w+\])", ir
    )
    global_smem_type = None
    if global_smem_definition_match:
        global_smem_type = global_smem_definition_match.group(1)
    else:
        raise RuntimeError("@global_smem definition not found")

    cast_idx = 0
    new_lines = []
    for line in ir.split("\n"):
        gep_match_global_smem = re.match(
            r"(\s*)(%\w+)\s*=\s*getelementptr\b[^,\n]*,\s+(<\d+\s+x\s+\w+>|\w+)\s+addrspace\((\d+)\)\*\s+([@%]\w+)(.*)",
            line,
        )
        if gep_match_global_smem:
            leading_spaces = gep_match_global_smem.group(1)
            ptr_var_name = gep_match_global_smem.group(2)
            ptr_type = gep_match_global_smem.group(3)
            addrspace = gep_match_global_smem.group(4)
            base_ptr = gep_match_global_smem.group(5)
            remaining = gep_match_global_smem.group(6)
            if int(addrspace) == 3:
                # breakpoint()
                assert base_ptr == "@global_smem"
                cast_result = f"%vec_cast_{cast_idx}"
                cast_idx += 1
                new_lines.append(
                    f"{leading_spaces}{cast_result} = bitcast {global_smem_type} addrspace({addrspace})* {base_ptr} to {ptr_type} addrspace({addrspace})*"
                )
                new_lines.append(
                    f"{leading_spaces}{ptr_var_name} = getelementptr inbounds {ptr_type}, {ptr_type} addrspace({addrspace})* {cast_result}{remaining}"
                )
                continue

        # search for load from scalar pointer that is the result of getelementptr or extractvalue
        load_match = re.match(
            r"(\s*)(%\w+)\s*=\s*load\s+(<\d+\s+x\s+(\w+)>),\s+<[^>]+>\s+addrspace\((\d+)\)\*\s+(%\w+)(.*)",
            line,
        )
        if load_match:
            leading_spaces = load_match.group(1)
            result_var = load_match.group(2)
            vec_type = load_match.group(3)
            scalar_elem = load_match.group(4)
            addrspace = load_match.group(5)
            ptr = load_match.group(6)
            remaining = load_match.group(7)
            if getelementptr_type_dict.get(ptr) == (scalar_elem, addrspace):
                cast_result = f"%vec_cast_{cast_idx}"
                cast_idx += 1
                new_lines.append(
                    f"{leading_spaces}{cast_result} = bitcast {scalar_elem} addrspace({addrspace})* {ptr} to {vec_type} addrspace({addrspace})*"
                )
                new_lines.append(
                    f"{leading_spaces}{result_var} = load {vec_type}, {vec_type} addrspace({addrspace})* {cast_result}{remaining}"
                )
                continue
            elif struct_type_dict.get(ptr) == f"{scalar_elem} addrspace({addrspace})*":
                cast_result = f"%vec_cast_{cast_idx}"
                cast_idx += 1
                new_lines.append(
                    f"{leading_spaces}{cast_result} = bitcast {scalar_elem} addrspace({addrspace})* {ptr} to {vec_type} addrspace({addrspace})*"
                )
                new_lines.append(
                    f"{leading_spaces}{result_var} = load {vec_type}, {vec_type} addrspace({addrspace})* {cast_result}{remaining}"
                )
                continue
            elif int(addrspace) == 3 and getelementptr_type_dict.get(ptr, (None, None))[0] != vec_type:
                # if loading from global smem (addrspace 3), then cast global smem ptr
                cast_result = f"%vec_cast_{cast_idx}"
                cast_idx += 1
                global_smem_ptr_type, global_smem_addrspace = getelementptr_type_dict[ptr]
                new_lines.append(
                    f"{leading_spaces}{cast_result} = bitcast {global_smem_ptr_type} addrspace({global_smem_addrspace})* {ptr} to {vec_type} addrspace({global_smem_addrspace})*"
                )
                new_lines.append(
                    f"{leading_spaces}load {vec_type}, {vec_type} addrspace({global_smem_addrspace})* {cast_result}{remaining}"
                )
                continue

        # search for store to scalar pointer that is the result of getelementptr
        store_match = re.match(
            r"(\s*)store\s+(<\d+\s+x\s+(\w+)>)\s+([^,]+),\s+<[^>]+>\s+addrspace\((\d+)\)\*\s+(%\w+)(.*)",
            line,
        )
        if store_match:
            leading_spaces = store_match.group(1)
            vec_type = store_match.group(2)
            scalar_type = store_match.group(3)
            val = store_match.group(4)
            addrspace = store_match.group(5)
            ptr = store_match.group(6)
            remaining = store_match.group(7)
            if getelementptr_type_dict.get(ptr) == (scalar_type, addrspace):
                cast_result = f"%vec_cast_{cast_idx}"
                cast_idx += 1
                new_lines.append(
                    f"{leading_spaces}{cast_result} = bitcast {scalar_type} addrspace({addrspace})* {ptr} to {vec_type} addrspace({addrspace})*"
                )
                new_lines.append(
                    f"{leading_spaces}store {vec_type} {val}, {vec_type} addrspace({addrspace})* {cast_result}{remaining}"
                )
                continue
            elif int(addrspace) == 3 and getelementptr_type_dict.get(ptr, (None, None))[0] != vec_type:
                # if storing into global smem (addrspace 3), then cast global smem ptr
                cast_result = f"%vec_cast_{cast_idx}"
                cast_idx += 1
                global_smem_ptr_type, global_smem_addrspace = getelementptr_type_dict[ptr]
                new_lines.append(
                    f"{leading_spaces}{cast_result} = bitcast {global_smem_ptr_type} addrspace({global_smem_addrspace})* {ptr} to {vec_type} addrspace({global_smem_addrspace})*"
                )
                new_lines.append(
                    f"{leading_spaces}store {vec_type} {val}, {vec_type} addrspace({global_smem_addrspace})* {cast_result}{remaining}"
                )
                continue

        new_lines.append(line)
    ir = "\n".join(new_lines)
    return ir


def get_func_args(s: str) -> list:
    """Get arg list, handling nested parentheses"""
    args, depth, cur = [], 0, []
    for ch in s:
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            cur.append(ch)
    if cur:
        args.append("".join(cur).strip())
    return args


def modify_func_signature(ir: str, ptr_type_dict: dict) -> str:
    func_start_idx = ir.index("define void @")
    open_paren_idx = ir.index("(", func_start_idx) + 1
    # find closing ")"
    depth, i = 1, open_paren_idx
    while i < len(ir) and depth > 0:
        if ir[i] == "(":
            depth += 1
        elif ir[i] == ")":
            depth -= 1
        i += 1
    paren_close = i - 1

    func_args_str = ir[open_paren_idx:paren_close]
    args = get_func_args(func_args_str)
    new_args = []
    for arg in args:
        m = re.match(r"(ptr addrspace\((\d+)\))(.*?)(%\w+)$", arg.strip())
        if m:
            addrspace = m.group(2)
            attrs = m.group(3)
            var_name = m.group(4)
            typed = ptr_type_dict.get(var_name, f"i8 addrspace({addrspace})*")
            new_args.append(f"{typed}{attrs}{var_name}")
        else:
            new_args.append(arg)
    ir = ir[:open_paren_idx] + ", ".join(new_args) + ir[paren_close:]
    return ir


def rewrite_metadata(ir: str) -> str:
    # rewrite metadata func ptr reference
    # e.g. ptr @funcname → void (arg types)* @funcname
    func_match = re.search(r"define void @(\w+)\(", ir)
    if func_match:
        func_name = func_match.group(1)
        func_open_str = "define void @" + func_name + "("
        func_signature_start_idx = ir.index(func_open_str) + len(func_open_str)
        depth, j = 1, func_signature_start_idx
        while j < len(ir) and depth > 0:
            if ir[j] == "(":
                depth += 1
            elif ir[j] == ")":
                depth -= 1
            j += 1
        func_signature_close_idx = j - 1

        func_args = get_func_args(ir[func_signature_start_idx:func_signature_close_idx])
        arg_types = []
        for arg in func_args:
            # extract type: "type addrspace(N)*" or just "type"
            m = re.match(r"(<\d+\s+x\s+\w+>|\w+)(\s+addrspace\(\d+\)\*)?", arg.strip())
            arg_types.append(m.group(0) if m else arg.strip())

        func_type = f"void ({', '.join(arg_types)})*"
        ir = re.sub(
            r"\bptr @" + re.escape(func_name) + r"\b",
            func_type + " @" + func_name,
            ir,
        )

    return ir


def convert_opaque_ptrs_insertvalue(ir: str, getelementptr_type_dict: dict) -> tuple[str, dict]:
    new_lines = []

    # contains types for results of insertvalue
    struct_type_dict = {}
    for line in ir.split("\n"):
        insertvalue_match = re.match(
            r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*insertvalue\s+(?P<agg_type>\{[^}]+\})\s+(?P<struct_vals>\S+),\s*(?P<elem_type>ptr\s+addrspace\((?P<addrspace>\d+)\))\s+(?P<elem_val>%\w+),\s*(?P<idx>\d+)",
            line,
        )
        if insertvalue_match:
            # breakpoint()
            elem_val = insertvalue_match.group("elem_val")
            if elem_val in getelementptr_type_dict:
                elem_type, addrspace = getelementptr_type_dict[elem_val]

                typed_ptr = f"{elem_type} addrspace({addrspace})*"
                new_agg_type = insertvalue_match.group("agg_type").replace(
                    "ptr addrspace(" + addrspace + ")", typed_ptr
                )
                # breakpoint()

                new_line = (
                    f"{insertvalue_match.group('indent')}{insertvalue_match.group('result')} = insertvalue {new_agg_type} "
                    f"{insertvalue_match.group('struct_vals')}, {typed_ptr} {insertvalue_match.group('elem_val')}, {insertvalue_match.group('idx')}"
                )
                new_lines.append(new_line)

                # track struct type in dict
                struct_type_dict[insertvalue_match.group("result")] = new_agg_type
            else:
                raise RuntimeError(
                    f"Elem val {elem_val} was not found in getelementptr_type_dict, perhaps this pointer was created from a different operation (not getelementptr)"
                )
        else:
            new_lines.append(line)

    ir = "\n".join(new_lines)
    return ir, struct_type_dict


def convert_opaque_ptrs_phi(ir: str, struct_type_dict: dict) -> tuple[str, dict]:
    new_lines = []
    for line in ir.split("\n"):
        phi_match = re.match(
            r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*phi\s+(?P<agg_type>\{[^}]+\})\s+(?P<pairs>.*)",
            line,
        )
        if phi_match and "ptr addrspace" in phi_match.group("agg_type"):
            incoming_vals = re.findall(r"\[\s*(%\w+),\s*%\w+\s*\]", phi_match.group("pairs"))
            new_agg_type = next((struct_type_dict[v] for v in incoming_vals if v in struct_type_dict), None)
            if new_agg_type:
                new_line = f"{phi_match.group('indent')}{phi_match.group('result')} = phi {new_agg_type} {phi_match.group('pairs')}"
                new_lines.append(new_line)
                # track so downstream can use this result
                struct_type_dict[phi_match.group("result")] = new_agg_type
            else:
                raise RuntimeError(f"Could not determine struct element type for phi: {line.strip()}")
        else:
            new_lines.append(line)

    ir = "\n".join(new_lines)
    return ir, struct_type_dict


def convert_opaque_ptrs_extractvalue(ir: str, struct_type_dict: dict) -> tuple[str, dict]:
    new_lines = []
    for line in ir.split("\n"):
        extractvalue_match = re.match(
            r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*extractvalue\s+(?P<agg_type>\{[^}]+\})\s+(?P<agg_val>%\w+),\s*(?P<idx>\d+)",
            line,
        )
        if extractvalue_match and "ptr addrspace" in extractvalue_match.group("agg_type"):
            agg_val = extractvalue_match.group("agg_val")
            new_agg_type = struct_type_dict.get(agg_val)
            if new_agg_type:
                new_line = f"{extractvalue_match.group('indent')}{extractvalue_match.group('result')} = extractvalue {new_agg_type} {agg_val}, {extractvalue_match.group('idx')}"
                new_lines.append(new_line)
                fields = [f.strip() for f in new_agg_type.strip("{}").split(",")]
                struct_type_dict[extractvalue_match.group("result")] = fields[int(extractvalue_match.group("idx"))]
            else:
                raise RuntimeError(f"Could not determine struct element type for extractvalue: {line.strip()}")
        else:
            new_lines.append(line)
    return "\n".join(new_lines), struct_type_dict


def convert_opaque_ptrs_to_typed(ir: str) -> str:
    """Convert opaque ptrs to typed ptrs

    Triton LLVM produces opaque ptrs, but metal's JIT can't compile for some reason.
    Metal likely uses older version of LLVM. Need to convert opaque ptrs to typed ptr (e.g. float addrspace(1)*).
    Determine types from load/store/getelementptr.

    Also need to:
    - cast scalar ptrs to vector ptrs when loading vector from scalar ptr.
    - modify function signature
    - modify metadata containing function ptr
    """
    # search for getelementptr/load/store instructions to determine types of the ptrs
    ptr_type_dict: dict = {}
    # e.g. getelementptr/load float, ptr addrspace(1) %1
    for m in re.finditer(
        r"(?:getelementptr(?:\s+inbounds)?|load)\s+(<\d+\s+x\s+\w+>|\w+),\s+ptr\s+addrspace\((\d+)\)\s+(%\w+)", ir
    ):
        # group(3): var name
        # group(1): ptr type
        # group(2): addr space
        ptr_type_dict[m.group(3)] = f"{m.group(1)} addrspace({m.group(2)})*"
    # e.g. store float %val, ptr addrspace(N) %ptr
    for m in re.finditer(r"\bstore\s+(<\d+\s+x\s+\w+>|\w+)\s+[^,\n]+,\s+ptr\s+addrspace\((\d+)\)\s+(%\w+)", ir):
        # group(3): ptr name
        # group(1): type being stored at the ptr
        # group(2): addr space
        ptr_type_dict[m.group(3)] = f"{m.group(1)} addrspace({m.group(2)})*"

    # rewrite ptrs to include types
    ir = rewrite_ptrs(ir)

    getelementptr_type_dict = generate_getelementptr_type_dict(ir)

    # handle insertvalue with opaque ptrs
    ir, struct_type_dict = convert_opaque_ptrs_insertvalue(ir, getelementptr_type_dict)

    # handle phi with opaque ptrs
    # TODO this only works for phi ops that use results of insertvalue as input values
    # the types of these input values are stored in struct_type_dict
    ir, struct_type_dict = convert_opaque_ptrs_phi(ir, struct_type_dict)

    # handle extractvalue with opaque ptrs
    ir, struct_type_dict = convert_opaque_ptrs_extractvalue(ir, struct_type_dict)

    # need to do this after converting opaque ptrs for insertvalue/phi/extractvalue
    ir = insert_bitcast_for_vecs(ir, getelementptr_type_dict, struct_type_dict)

    # TODO handle cases when loading vec from ptr that is not result of getelementptr?
    ir = modify_func_signature(ir, ptr_type_dict)
    ir = rewrite_metadata(ir)

    return ir
