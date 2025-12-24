from typing import List
from .schema import Run, ThrowEvent

def compute_runs(events: List[ThrowEvent], max_gap: float = 1.0) -> List[Run]:
    if not events:
        return []
        
    runs = []
    current_run_events = [events[0]]
    
    for i in range(1, len(events)):
        gap = events[i].timestamp - events[i-1].timestamp
        if gap < max_gap:
            current_run_events.append(events[i])
        else:
            if len(current_run_events) >= 3: # Minimum 3 throws for a run
                runs.append(Run(
                    start_time=current_run_events[0].timestamp,
                    end_time=current_run_events[-1].timestamp,
                    throw_count=len(current_run_events)
                ))
            current_run_events = [events[i]]
            
    if len(current_run_events) >= 3:
        runs.append(Run(
            start_time=current_run_events[0].timestamp,
            end_time=current_run_events[-1].timestamp,
            throw_count=len(current_run_events)
        ))
        
    return runs
