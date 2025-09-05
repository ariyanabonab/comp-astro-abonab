import argparse

def ball_height(h0, time, g=9.8):
    """
    Calculate the height of a ball dropped from height h0 at a given time.
    
    Parameters:
        h0 (float): Initial height in meters
        time (float): Time in seconds
        g (float): Acceleration due to gravity (default 9.8 m/s^2)
        
    Returns:
        float: Height of the ball at given time
    """
    height = h0 - 0.5 * g * time**2
    return max(height, 0)  # prevents negative height


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the height of a ball being dropped over time."
    )
    parser.add_argument(
        "h0", type=float,
        help="Initial height in meters"
    )
    parser.add_argument(
        "time", type=float,
        help="Time elapsed in seconds"
    )
    parser.add_argument(
        "-g", "--gravity", type=float, default=9.8,
        help="Acceleration due to gravity (default: 9.8 m/s^2)"
    )
    
    args = parser.parse_args()
    
    h = ball_height(args.h0, args.time, args.gravity)
    print(f"Height of the ball after {args.time} seconds: {h:.2f} meters")


if __name__ == "__main__":
    main()
