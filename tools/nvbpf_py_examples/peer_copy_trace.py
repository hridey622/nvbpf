from nvbpf_py import api_trace, tool


@tool("peer_copy_trace")
class PeerCopyTrace:
    peer_sync = api_trace(
        callbacks=["API_CUDA_cuMemcpyPeer", "API_CUDA_cuMemcpyPeer_ptds"],
        correlate_launches=True,
        description="Trace synchronous peer copies and correlate nearby launches",
    )
    peer_async = api_trace(
        callbacks=["API_CUDA_cuMemcpyPeerAsync", "API_CUDA_cuMemcpyPeerAsync_ptsz"],
        correlate_launches=True,
        description="Trace async peer copies and correlate nearby launches",
    )
