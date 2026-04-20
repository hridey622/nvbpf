from nvbpf_py import gemm_orchestration_map, tool


@tool(
    "gemm_orchestration_map_py",
    banner="GEMM_ORCHESTRATION_MAP_PY",
)
class GemmOrchestrationMapPy:
    analysis = gemm_orchestration_map()
