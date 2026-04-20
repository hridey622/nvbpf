from nvbpf_py import counter, tool


@tool("atomic_count")
class AtomicCount:
    atomics = counter(
        opcodes=["ATOM", "RED"],
        description="Count atomic and reduction instructions",
    )
