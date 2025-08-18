import pandas as pd
from math import ceil
from src.policies import simple_ceiling, buffered, confidence_aware, policy_aware
from src.safety_net import capacity_latency, safety_override

def simulate(df, forecasts, policy_name, capacity=25, init_containers=2,
             cooldown=10, latency_sla=150, cpu_sla=90):
    """
    df: ground truth with timestamp, req_per_min
    forecasts: dataframe aligned by timestamp with columns:
        - yhat (point) and optionally yhat_upper
    """
    f_cap = capacity_latency()  # latency + cpu model
    logs = []
    containers = init_containers
    last_scale_min = None
    for i, row in df.iterrows():
        t = i
        demand = row["req_per_min"]
        # fetch forecast at time t (assume forecasts aligned to timestamps)
        fc_row = forecasts.loc[forecasts["timestamp"]==row["timestamp"]]
        if fc_row.empty:
            pred = demand  # fallback
            upper = demand
        else:
            pred = float(fc_row["yhat"].iloc[0])
            upper = float(fc_row.get("yhat_upper", fc_row["yhat"]).iloc[0])

        # choose policy decision
        scaled = False
        if policy_name == "simple":
            containers_target = simple_ceiling(pred, capacity)
            if containers_target != containers:
                containers = containers_target
                scaled = True
        elif policy_name == "buffered":
            containers_target = buffered(pred, capacity, buffer=1.15)
            if containers_target != containers:
                containers = containers_target
                scaled = True
        elif policy_name == "conf":
            containers_target = confidence_aware(upper, capacity)
            if containers_target != containers:
                containers = containers_target
                scaled = True
        elif policy_name == "policy":
            containers_new, scaled = policy_aware(pred, capacity, containers, last_scale_min,
                                                  now_min=i, cooldown=cooldown)
            if scaled:
                containers = containers_new

        # compute latency/cpu under chosen containers
        latency, cpu = f_cap(demand, containers, capacity)

        # safety net override
        if safety_override(latency, cpu, latency_sla=latency_sla, cpu_sla=cpu_sla):
            containers += 1  # force single-step scale-up
            scaled = True
            latency, cpu = f_cap(demand, containers, capacity)

        if scaled:
            last_scale_min = i

        over = (containers*capacity) > demand * 1.15  # 15% buffer
        under = demand > (containers*capacity)

        logs.append({
            "timestamp": row["timestamp"],
            "demand": demand,
            "pred": pred,
            "containers": containers,
            "lat_ms": latency,
            "cpu_pct": cpu,
            "over": int(over),
            "under": int(under),
            "policy": policy_name
        })
    return pd.DataFrame(logs)
