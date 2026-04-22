from nvbpf_py import (
    array,
    counter,
    hook,
    map_value,
    on_launch_exit,
    short_kernel_name,
    tool,
)


@tool("loop_bucket_report_py", banner="LOOP_BUCKET_REPORT_PY")
class LoopBucketReportPy:
    sampled = counter(loads=True)
    lane_bucket_hits = array(type_name="u64", length=4)
    seen_mask = array(type_name="u64", length=1)

    @hook(loads=True)
    def on_load(ctx):
        for bucket in range(4):
            lower = bucket * 8
            upper = (bucket + 1) * 8
            if ctx.active_lanes > lower and ctx.active_lanes <= upper:
                ctx.atomic_add("lane_bucket_hits", bucket, 1)
                current = ctx.map_get("seen_mask", 0)
                ctx.map_set("seen_mask", 0, current | (1 << bucket))

    @on_launch_exit()
    def report():
        print(
            "kernel=", short_kernel_name(),
            "b0=", map_value("lane_bucket_hits", 0),
            "b1=", map_value("lane_bucket_hits", 1),
            "b2=", map_value("lane_bucket_hits", 2),
            "b3=", map_value("lane_bucket_hits", 3),
            "seen_mask=", map_value("seen_mask", 0),
        )
