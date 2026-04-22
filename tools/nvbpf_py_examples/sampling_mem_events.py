from nvbpf_py import count, counter, device_hook, emit, event, tool


@tool("sampling_mem_events")
class SamplingMemEvents:
    sampled = counter(
        loads=True,
        stores=True,
        description="Count sampled memory instructions",
    )

    samples = event(
        fields={
            "addr": "u64",
            "sm_id": "u32",
            "warp_id": "u32",
            "cta_id_x": "u32",
            "cta_id_y": "u32",
            "is_load": "u8",
        },
        capacity=4096,
        description="Sampled memory events",
    )

    @device_hook(loads=True, stores=True, description="Emit one sample for 64B-aligned addresses")
    def on_memory(pred, addr, is_load, sm_id, warp_id, cta_id_x, cta_id_y):
        if active_lanes == 0:
            return
        if (addr & 0x3F) != 0:
            return
        count("sampled")
        emit(
            "samples",
            addr=addr,
            sm_id=sm_id,
            warp_id=warp_id,
            cta_id_x=cta_id_x,
            cta_id_y=cta_id_y,
            is_load=is_load,
        )
