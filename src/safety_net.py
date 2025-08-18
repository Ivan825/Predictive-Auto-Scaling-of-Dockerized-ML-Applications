def capacity_latency(cpu_perc_per_over=2.5, base_latency=40, alpha=2.0, beta=1.5):
    """
    Returns a function that maps (req_per_min, containers, capacity_per_container)
    -> (latency_ms, avg_cpu_percent)
    """
    def f(demand, containers, C):
        containers = max(1, containers)
        per = demand / containers
        if per <= C:
            latency = base_latency
            cpu = min(100, (per/C)*70)
        else:
            latency = base_latency + alpha * ((per - C) ** beta)
            cpu = min(100, 70 + (per/C - 1)*cpu_perc_per_over*100/2.5)  # rough
        return latency, cpu
    return f

def safety_override(latency, cpu, latency_sla=150, cpu_sla=90):
    return (latency > latency_sla) or (cpu > cpu_sla)
