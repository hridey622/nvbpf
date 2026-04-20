from nvbpf_py import epilogue_fusion_trace, tool


@tool("epilogue_fusion_trace_py", banner="EPILOGUE_FUSION_TRACE_PY")
class EpilogueFusionTracePy:
    analysis = epilogue_fusion_trace()
