from nvbpf_py import counter, tool


@tool("memory_mix")
class MemoryMix:
    loads = counter(
        loads=True,
        description="Count non-constant load instructions",
    )
    stores = counter(
        stores=True,
        description="Count store instructions",
    )
