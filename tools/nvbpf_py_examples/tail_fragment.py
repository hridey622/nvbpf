from nvbpf_py import tail_fragment_tracker, tool


@tool("tail_fragment_tracker_py", banner="TAIL_FRAGMENT_TRACKER_PY")
class TailFragmentTrackerPy:
    analysis = tail_fragment_tracker()
