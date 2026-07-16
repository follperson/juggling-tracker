from argparse import ArgumentParser
from jugglecount.core.pipeline import run_analysis

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("output_dir", type=str, help="Output directory for results")
    parser.add_argument("--test", action="store_true", help="Run in test mode", default=False)
    parser.add_argument("--user_id", type=str, help="User ID", default="default_user")
    parser.add_argument("--no-roi", action="store_true", help="Disable ROI optimization", default=False)
    parser.add_argument("--interval", type=int, help="Frame processing interval", default=2)
    args = parser.parse_args()
    
    enable_roi = not args.no_roi
    run_analysis(args.video_path, args.output_dir, args.test, args.user_id, enable_roi=enable_roi, process_interval=args.interval)
