from __future__ import annotations

import ast
from dataclasses import dataclass

from .model import DeviceHookSpec, EventSpec


class TranspileError(RuntimeError):
    pass


@dataclass(frozen=True)
class HookRender:
    signature_params: tuple[str, ...]
    body: str
    needs_addr: bool
    needs_is_load: bool


_BIN_OPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
}

_CMP_OPS = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


class _HookTranspiler:
    def __init__(self, hook: DeviceHookSpec, events: dict[str, EventSpec], counters: set[str]) -> None:
        self.hook = hook
        self.events = events
        self.counters = counters
        self.indent = 1
        self.lines: list[str] = []
        self.locals: set[str] = set()
        self.arg_names = set(hook.args)

    def render(self) -> HookRender:
        tree = ast.parse(self.hook.source)
        fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
        if fn is None:
            raise TranspileError(f"could not find function body for hook {self.hook.name}")

        signature_params = ["int pred"]
        if "addr" in self.arg_names:
            signature_params.append("uint64_t addr")
        if "is_load" in self.arg_names:
            signature_params.append("int is_load")
        for counter in sorted(self.counters):
            signature_params.append(f"uint64_t p_counter_{counter}")
        for event_name in sorted(self.events):
            signature_params.append(f"uint64_t p_event_{event_name}")

        self._emit("const int active_mask = __ballot_sync(__activemask(), 1);")
        self._emit("const int predicate_mask = __ballot_sync(__activemask(), pred);")
        self._emit("const int laneid = get_laneid();")
        self._emit("const int first_laneid = __ffs(active_mask) - 1;")
        self._emit("const int active_lanes = __popc(predicate_mask);")
        self._emit("const uint32_t sm_id = bpf_get_current_sm_id();")
        self._emit("const uint32_t warp_id = bpf_get_current_warp_id();")
        self._emit("const uint32_t cta_id_x = blockIdx.x;")
        self._emit("const uint32_t cta_id_y = blockIdx.y;")
        self._emit("const uint32_t cta_id_z = blockIdx.z;")
        self._emit("if (first_laneid != laneid) return;")

        for counter in sorted(self.counters):
            self._emit(f"uint64_t* counter_{counter} = (uint64_t*)p_counter_{counter};")
        for event_name, event in sorted(self.events.items()):
            self._emit(
                f"auto* ring_{event_name} = "
                f"(BpfRingBufMap<{event_name}_event_t, {event.capacity}>*)p_event_{event_name};"
            )

        for stmt in fn.body:
            self._stmt(stmt)

        body = "\n".join(self.lines)
        return HookRender(
            signature_params=tuple(signature_params),
            body=body,
            needs_addr="addr" in self.arg_names,
            needs_is_load="is_load" in self.arg_names,
        )

    def _emit(self, line: str) -> None:
        self.lines.append("    " * self.indent + line)

    def _stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.If):
            self._emit(f"if ({self._expr(stmt.test)}) {{")
            self.indent += 1
            for inner in stmt.body:
                self._stmt(inner)
            self.indent -= 1
            if stmt.orelse:
                self._emit("} else {")
                self.indent += 1
                for inner in stmt.orelse:
                    self._stmt(inner)
                self.indent -= 1
            self._emit("}")
            return

        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                raise TranspileError("return with a value is not supported in hook bodies")
            self._emit("return;")
            return

        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise TranspileError("only simple assignments are supported in hook bodies")
            name = stmt.targets[0].id
            expr = self._expr(stmt.value)
            if name in self.locals:
                self._emit(f"{name} = {expr};")
            else:
                self.locals.add(name)
                self._emit(f"auto {name} = {expr};")
            return

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == "count":
                self._emit(self._count_call(call))
                return
            if isinstance(call.func, ast.Name) and call.func.id == "emit":
                for line in self._emit_call(call):
                    self._emit(line)
                return

        raise TranspileError(
            f"unsupported statement in hook {self.hook.name}: {ast.dump(stmt, include_attributes=False)}"
        )

    def _count_call(self, call: ast.Call) -> str:
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            raise TranspileError("count() expects one string literal argument")
        counter_name = call.args[0].value
        if counter_name not in self.counters:
            raise TranspileError(f"count() references unknown counter {counter_name!r}")
        return f"atomicAdd((unsigned long long*)counter_{counter_name}, 1ULL);"

    def _emit_call(self, call: ast.Call) -> list[str]:
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            raise TranspileError("emit() expects an event name string literal as its first argument")
        event_name = call.args[0].value
        event = self.events.get(event_name)
        if event is None:
            raise TranspileError(f"emit() references unknown event {event_name!r}")

        kw = {item.arg: item.value for item in call.keywords}
        missing = [field.name for field in event.fields if field.name not in kw]
        if missing:
            raise TranspileError(f"emit({event_name!r}) missing fields: {', '.join(missing)}")

        lines = [f"{event_name}_event_t evt{{}};"]
        for field in event.fields:
            lines.append(f"evt.{field.name} = {self._expr(kw[field.name])};")
        lines.append(f"ring_{event_name}->output(&evt);")
        return lines

    def _expr(self, expr: ast.expr) -> str:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                return "true" if expr.value else "false"
            if isinstance(expr.value, str):
                return f'"{expr.value}"'
            return repr(expr.value)
        if isinstance(expr, ast.BinOp):
            op = _BIN_OPS.get(type(expr.op))
            if op is None:
                raise TranspileError(f"unsupported binary operator: {type(expr.op).__name__}")
            return f"({self._expr(expr.left)} {op} {self._expr(expr.right)})"
        if isinstance(expr, ast.BoolOp):
            op = "&&" if isinstance(expr.op, ast.And) else "||"
            return "(" + f" {op} ".join(self._expr(v) for v in expr.values) + ")"
        if isinstance(expr, ast.UnaryOp):
            if isinstance(expr.op, ast.Not):
                return f"!({self._expr(expr.operand)})"
            if isinstance(expr.op, ast.USub):
                return f"-({self._expr(expr.operand)})"
            raise TranspileError(f"unsupported unary operator: {type(expr.op).__name__}")
        if isinstance(expr, ast.Compare):
            if len(expr.ops) != 1 or len(expr.comparators) != 1:
                raise TranspileError("chained comparisons are not supported")
            op = _CMP_OPS.get(type(expr.ops[0]))
            if op is None:
                raise TranspileError(f"unsupported comparison operator: {type(expr.ops[0]).__name__}")
            return f"({self._expr(expr.left)} {op} {self._expr(expr.comparators[0])})"
        if isinstance(expr, ast.IfExp):
            return f"({self._expr(expr.test)} ? {self._expr(expr.body)} : {self._expr(expr.orelse)})"
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "int":
            if len(expr.args) != 1:
                raise TranspileError("int() expects one argument")
            return f"((int){self._expr(expr.args[0])})"

        raise TranspileError(
            f"unsupported expression in hook {self.hook.name}: {ast.dump(expr, include_attributes=False)}"
        )


def render_custom_hook(
    hook: DeviceHookSpec,
    events: dict[str, EventSpec],
    counters: set[str],
) -> HookRender:
    return _HookTranspiler(hook, events, counters).render()
