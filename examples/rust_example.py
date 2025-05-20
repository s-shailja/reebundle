import numpy as np
from reebundle.rust import find_connect_disconnect_events

def main():
    # Create two sample trajectories that will definitely have connect/disconnect events
    t1 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0]
    ]).tolist()
    
    t2 = np.array([
        [10.0, 0.0, 0.0],  # Start far away
        [2.1, 0.1, 0.0],   # Close to t1[2]
        [3.1, 0.1, 0.0],   # Close to t1[3]
        [4.1, 0.1, 0.0],   # Close to t1[4]
        [10.0, 0.0, 0.0]   # End far away
    ]).tolist()
    
    # Set parameters
    t1_id = "trajectory_1"
    t2_id = "trajectory_2"
    eps = 0.5  # Epsilon distance threshold
    
    print(f"Trajectory 1: {t1}")
    print(f"Trajectory 2: {t2}")
    print(f"Epsilon: {eps}")
    
    # Find connect and disconnect events using Rust implementation
    dic_t1, dic_t2 = find_connect_disconnect_events(t1_id, t2_id, t1, t2, eps)
    
    # Print results
    print("\nEvents for trajectory 1:")
    for key, events in dic_t1.items():
        print(f"  Position {key}:")
        for event in events:
            print(f"    {event.event} with {event.trajectory} at position {event.t}")
    
    print("\nEvents for trajectory 2:")
    for key, events in dic_t2.items():
        print(f"  Position {key}:")
        for event in events:
            print(f"    {event.event} with {event.trajectory} at position {event.t}")

if __name__ == "__main__":
    main() 