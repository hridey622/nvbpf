from nvbpf_py import array, gemm_wavefit, percpu_array, tool


@tool(
    "gemm_wavefit_trace_py",
    banner="GEMM_WAVEFIT_TRACE_PY",
)
class GemmWavefitTracePy:
    sm_cta_entries = percpu_array(
        type_name="u64",
        length=1,
        description="Per-SM CTA entry counts recorded at kernel entry",
    )
    active_sm_bitmap = array(
        type_name="u64",
        length=4,
        description="Bitmap of active SMs touched by the launch",
    )
    analysis = gemm_wavefit(
        sm_cta_entries_map="sm_cta_entries",
        active_sm_bitmap_map="active_sm_bitmap",
    )
