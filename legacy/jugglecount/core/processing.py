from typing import List
import numpy as np
from jugglecount.db.schema import Track, TrackPoint


def interpolate_tracks(tracks: List[Track], max_gap_frames: int = 10) -> List[Track]:
    """
    Fills in gaps in tracks. Uses quadratic interpolation for Y (gravity) and linear for X.
    """
    processed_tracks = []
    
    for track in tracks:
        if len(track.points) < 2:
            processed_tracks.append(track)
            continue
            
        new_points = []
        points = track.points
        n_points = len(points)
        
        for i in range(n_points - 1):
            p1 = points[i]
            p2 = points[i+1]
            new_points.append(p1)
            
            gap = p2.frame_idx - p1.frame_idx
            if 1 < gap <= max_gap_frames:
                # Prepare for interpolation
                # X is linear (constant horizontal velocity assumption)
                # Y is quadratic (gravity assumption)
                
                # Collect context points for polyfit
                # We need at least 3 points for a parabola.
                context_t = []
                context_y = []
                
                # Previous point
                if i > 0:
                    context_t.append(points[i-1].timestamp)
                    context_y.append(points[i-1].pos_y)
                
                # Gap start
                context_t.append(p1.timestamp)
                context_y.append(p1.pos_y)
                
                # Gap end
                context_t.append(p2.timestamp)
                context_y.append(p2.pos_y)
                
                # Next point
                if i + 2 < n_points:
                    context_t.append(points[i+2].timestamp)
                    context_y.append(points[i+2].pos_y)
                
                poly_y = None
                coeffs = None
                if len(context_t) >= 3:
                     try:
                         # Fit parabola (deg=2). 
                         coeffs = np.polyfit(context_t, context_y, 2)
                         poly_y = np.poly1d(coeffs)
                     except Exception:
                         poly_y = None
                         coeffs = None



                # Interpolate
                for j in range(1, gap):
                    alpha = j / gap
                    interp_f_idx = p1.frame_idx + j
                    interp_timestamp = p1.timestamp + (p2.timestamp - p1.timestamp) * alpha
                    
                    interp_x = p1.pos_x + (p2.pos_x - p1.pos_x) * alpha
                    
                    # Y is quadratic if possible
                    if poly_y is not None and coeffs is not None:
                        # Check for snap-to-peak
                        a, b, c = coeffs
                        if abs(a) > 1e-4:
                            t_vertex = -b / (2 * a)
                            fps_approx = 1.0 / ((p2.timestamp - p1.timestamp)/gap)
                            frame_dt = 1.0/fps_approx
                            
                            if abs(interp_timestamp - t_vertex) <= frame_dt * 0.7:
                                y_vertex = poly_y(t_vertex)
                                if y_vertex < p1.pos_y and y_vertex < p2.pos_y:
                                    interp_y = y_vertex
                                else:
                                    interp_y = poly_y(interp_timestamp)
                            else:
                                interp_y = poly_y(interp_timestamp)
                        else:
                            interp_y = poly_y(interp_timestamp)
                    else:
                        interp_y = p1.pos_y + (p2.pos_y - p1.pos_y) * alpha
                        
                    interp_conf = p1.confidence + (p2.confidence - p1.confidence) * alpha
                    
                    new_points.append(TrackPoint(
                        frame_idx=interp_f_idx,
                        timestamp=round(interp_timestamp, 4),
                        pos_x=round(interp_x, 4),
                        pos_y=round(interp_y, 4),
                        confidence=round(interp_conf, 4)
                    ))
        
        new_points.append(points[-1])
        track.points = new_points
        processed_tracks.append(track)
        
    return processed_tracks
