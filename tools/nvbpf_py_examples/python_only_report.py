from nvbpf_py import (
    array,
    counter,
    counter_value,
    grid_dim_x,
    hook,
    map_value,
    on_launch_enter,
    on_launch_exit,
    short_kernel_name,
    tool,
)


@tool("python_only_report_py", banner="PYTHON_ONLY_REPORT_PY")
class PythonOnlyReportPy:
    sampled = counter(loads=True)
    site_totals = array(type_name="u64", length=1)

    @hook(loads=True)
    def on_load(ctx):
        if ctx.active_lanes == 0:
            return
        ctx.atomic_add("site_totals", 0, ctx.active_lanes)

    @on_launch_enter()
    def announce():
        print("launch", short_kernel_name(), "grid_x=", grid_dim_x())

    @on_launch_exit()
    def report():
        print(
            "kernel=", short_kernel_name(),
            "grid_x=", grid_dim_x(),
            "sampled=", counter_value("sampled"),
            "lanes=", map_value("site_totals", 0),
        )
