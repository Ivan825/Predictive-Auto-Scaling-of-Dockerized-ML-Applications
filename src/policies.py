import math

def simple_ceiling(pred, capacity):
    return max(1, math.ceil(pred / capacity))

def buffered(pred, capacity, buffer=1.15):
    return max(1, math.ceil((pred * buffer) / capacity))

def confidence_aware(yhat_upper, capacity):
    return max(1, math.ceil(yhat_upper / capacity))

def policy_aware(pred, capacity, current, last_scale_min,
                 now_min, cooldown=10, down_thresh=0.6,
                 min_c=1, max_c=20):
    # cooldown: avoid thrashing
    if last_scale_min is not None and (now_min - last_scale_min) < cooldown:
        return current, False
    need = pred / capacity
    if need > current:
        return min(current+1, max_c), True
    elif need < current * down_thresh:
        return max(current-1, min_c), True
    return current, False
