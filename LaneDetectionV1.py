import numpy as np  # Importing NumPy for numerical operations
import cv2  # Importing OpenCV for image processing

def region_of_interest(image):
    """
    Applies a mask to keep only the region of interest (road area).
    
    This function defines a trapezoidal region where lanes are typically visible
    and masks out the irrelevant portions of the frame.
    
    Parameters:
    image (numpy.ndarray): Input image frame from the video.
    
    Returns:
    numpy.ndarray: Masked image containing only the region of interest.
    """
    height, width = image.shape[:2]  # Get the dimensions of the image

    # Define coordinates for the region of interest (ROI)
    bottom_left = (int(0.3 * width), int(0.8 * height))  # Bottom left point of the trapezoid
    bottom_right = (int(0.8 * width), int(0.8 * height))  # Bottom right point of the trapezoid
    top_left = (int(0.5 * width), int(0.5 * height))  # Top left, making the trapezoid narrower at the top
    top_right = (int(0.6 * width), int(0.5 * height))  # Top right, also narrowing at the top

    # Create a blank mask with the same shape as the input image
    mask = np.zeros_like(image)  

    # Define the polygon and fill it with white (255) to create the mask
    polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)

    # Apply mask using bitwise AND operation to extract only the ROI
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image  # Return the masked image

# Path to the input video file
video_path = r'F:\Projects\PycharmProjects\LaneDectionProject\drivingDataset\normalDay\nD_1.mp4'

# Open the video file for processing
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")  # Error message if file cannot be opened
    exit()  # Exit the script if video file is not accessible

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video

    if not ret:  # If frame reading fails (e.g., end of video)
        print("Video ended or cannot read frame.")  # Print message
        break  # Exit the loop

    # Apply Gaussian blur to reduce noise and smooth the image
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Apply region of interest mask
    mask = region_of_interest(frame)

    # Perform edge detection using Canny algorithm
    edges = cv2.Canny(mask, 30, 200)  # Thresholds set at 30 and 200

    # Apply Hough Line Transform to detect lane lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, maxLineGap=30)

    # If lines are detected, draw them on the frame
    if lines is not None and len(lines) > 0:
        print(f"Detected {len(lines)} lines")  # Debugging info to check number of detected lines
        for line in lines:
            if len(line[0]) == 4:  # Ensure the line contains four values (x1, y1, x2, y2)
                x1, y1, x2, y2 = line[0]  # Unpack coordinates
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 10)  # Draw line in blue with thickness of 10
            else:
                print(f"Skipping invalid line data: {line}")  # Debugging message for invalid line data

    # Display the processed frame with detected lane lines
    cv2.imshow('frame', frame)

    # Wait for a key press and check if 'q' is pressed to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Quit if 'q' is pressed
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
