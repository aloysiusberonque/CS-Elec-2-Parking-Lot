import argparse
import yaml
from coordinates_generator import CoordinatesGenerator
from motion_detector import MotionDetector
from colors import *
import logging

# Define the main function that runs the program
def main():
    logging.basicConfig(level=logging.INFO)  # Set up logging

    args = parse_args() # Parse command line arguments

    image_file = args.image_file # Get image file from command line arguments
    data_file = args.data_file # Get data file from command line arguments
    start_frame = args.start_frame # Get start frame from command line arguments

    # Generate coordinates file from image file, if specified
    if image_file is not None:
        with open(data_file, "w+") as points:
            generator = CoordinatesGenerator(image_file, points, COLOR_RED)
            generator.generate()

    # Load coordinates from data file and detect motion in video
    with open(data_file, "r") as data:
        points = yaml.load(data, Loader=yaml.FullLoader)
        detector = MotionDetector(args.video_file, points, int(start_frame))
        detector.detect_motion()

# Define function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Generates Coordinates File')

    parser.add_argument("--image",
                        dest="image_file",
                        required=False,
                        help="Image file to generate coordinates on")

    parser.add_argument("--video",
                        dest="video_file",
                        required=True,
                        help="Video file to detect motion on")

    parser.add_argument("--data",
                        dest="data_file",
                        required=True,
                        help="Data file to be used with OpenCV")

    parser.add_argument("--start-frame",
                        dest="start_frame",
                        required=False,
                        default=1,
                        help="Starting frame on the video")

    return parser.parse_args()

# Run the main function if this script is being executed directly
if __name__ == '__main__':
    main()

# python main.py --image images/parking_lot_1.png --data data/coordinates_1.yml --video videos/parking_lot_1.mp4 --start-frame 400

# python main.py --data data/coordinates_1.yml --video videos/parking_lot_1.mp4 --start-frame 400


