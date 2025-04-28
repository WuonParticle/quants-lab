import time
import datetime

def perf_log_text(start_time: float) -> str:
    """Returns a human-readable duration between start_time and now."""
    return str(datetime.timedelta(seconds=time.perf_counter() - start_time)) 