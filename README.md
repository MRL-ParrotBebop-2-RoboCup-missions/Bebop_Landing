# Bebop_Landing

## Features

- Landing pad Detection
- pid Controller (decreasing errors in x,y,z dimensions)
- pid controller (adjusting Drone speed)
- passing through the window

## Landing pad detection 

Drone can detect special objects aiming its camera or sensors.
# Color Detection with OpenCV in HSV Color Space
![Landing Pad](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQHVGvEDs0B9swp2Kkvq025w8zcCTEVfj4awuM1dF6u6fiF9I8kWLue5CYTLKh1FcrlDsw&usqp=CAU)
This Python script detects a specific color in real-time video using OpenCV and the HSV (Hue-Saturation-Value) color space. It draws contours around the detected color in the video feed.

## Prerequisites

- Python 3.x
- OpenCV (`pip install opencv-python`)

## Usage

1. Run the `color_detection.py` script.
2. Adjust the HSV range parameters to match the color you want to detect:
   - `hue_min`, `hue_max`: Range of hue values
   - `saturation_min`, `saturation_max`: Range of saturation values
   - `value_min`, `value_max`: Range of value (brightness) values
3. The script will open a video window showing the original video feed with contours drawn around the detected color.

## Customization
```bash
  import cv2
import numpy as np

# Define the range of HSV values for the specific color
lower_color = np.array([hue_min, saturation_min, value_min])
upper_color = np.array([hue_max, saturation_max, value_max])

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask for the specified color
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours around the detected color
    for contour in contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    # Show the original frame with contours
    cv2.imshow("Color Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and destroy OpenCV windows
cap.release()
cv2.destroyAllWindows()
```
You can modify the script to work with a different video source (e.g., a video file) or implement additional features based on your specific use case.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This script was created for educational purposes and is based on the capabilities of OpenCV.

---

For more information on OpenCV, visit the [OpenCV Documentation](https://docs.opencv.org/).

If you have any questions or suggestions, please feel free to contribute or reach out!


Full tutorial :

https://www.linkedin.com/feed/update/urn:li:activity:7030598976512892928?utm_source=share&utm_medium=member_desktop
##  the Proportional-Integral-Derivative (PID) controller 


A PID (Proportional-Integral-Derivative) controller is a widely used feedback control mechanism in control systems. It is used to regulate a process based on feedback by adjusting a control output.

## Overview

The PID controller consists of three main components:

1. **Proportional (P) Term:** This term provides an output proportional to the current error. It helps bring the system closer to the desired setpoint but may result in steady-state error if used alone.

2. **Integral (I) Term:** This term accumulates past errors over time and helps eliminate steady-state error. It is useful when there's a constant bias or offset in the system.

3. **Derivative (D) Term:** This term considers the rate of change of the error. It helps improve the stability of the system and can prevent overshooting.

## Usage

In this repository, you'll find a simple PID controller implementation in [language]. The controller can be used for [describe the application, e.g., controlling a motor speed, temperature, etc.].

### Prerequisites

- [List any required libraries or tools]

### Installation

1. [Step-by-step installation instructions]

### Usage Example

```[language]
// Import the PID controller module
const PIDController = require('pid-controller');

// Create a new PID controller instance
const pid = new PIDController(Kp, Ki, Kd);

// Set the desired setpoint
pid.setSetpoint(desiredValue);

// Loop to control the process
while (true) {
  // Get the current value from the process
  const currentValue = getCurrentValue();

  // Compute the control output
  const controlOutput = pid.compute(currentValue);

  // Apply the control output to the process
  applyControlOutput(controlOutput);

  // Add a delay or use the appropriate timing mechanism
}
Full Tutorial :
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiAtZnK372AAxWDTaQEHUZ6DGMQFnoECBcQAw&url=https%3A%2F%2Fwww.omega.com%2Fen-us%2Fresources%2Fpid-controllers&usg=AOvVaw0uxGPDA38LGDhXTbVxHkeA&opi=89978449
