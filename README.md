# vehicle-speed-estimation
This project aims to detect and track vehicles on traffic recordings, then estimate the speed of the vehicles.



# Problem Definition and Purpose

The main purpose of this computer vision project is to detect vehicles on a traffic recording and track them to estimate the speed of the vehicles. Detecting the speed violations on traffic for the specific recording area and take actions according to the results is aimed. Also it can be used as a source for urban planning since it can give some insights about the traffic flow on the recorded area.


# Methodology and Approach

## Dataset Usage

This project utilizes the **Rouen Dataset** from the Urban Tracker project page. This dataset was selected for testing as it provides realistic urban traffic video footage suitable for vehicle detection and tracking tasks. For vehicle detection, a **pre-trained YOLOv11 model** was employed, leveraging its existing training on various vehicle types.

### Dataset Citation
Jodoin, J.-P., Bilodeau, G.-A., Saunier, N., Urban Tracker: Multiple Object Tracking in Urban Mixed Traffic, Accepted for IEEE Winter conference on Applications of Computer Vision (WACV14), Steamboat Springs, Colorado, USA, March 24-26, 2014

## Vehicle Detection

For the initial step of identifying vehicles within each video frame, a **pre-trained YOLOv11 model** was employed. This model was chosen for its efficiency and effectiveness in object detection. The model was configured to detect specific vehicle classes relevant to the project, including cars (COCO class ID 2), motorcycles (COCO class ID 3), buses (COCO class ID 5), trucks (COCO class ID 7), and bicycles (COCO class ID 1).


## Vehicle Tracking

Following the detection of vehicles by the YOLOv11 model, the **DeepSort algorithm** was utilized to track these detections across sequential video frames. DeepSort assigns a unique identification (ID) to each detected object and maintains its trajectory over time by associating detections in the current frame with existing tracks. This tracking capability is essential for monitoring individual vehicles and measuring the distance they travel.

To measure the displacement of a tracked vehicle, the position (specifically, the centroid of the bounding box) of objects with the same tracking ID was compared between frames. The distance covered in pixels by a tracked vehicle was calculated based on its change in position between frames corresponding to the video's frame rate (FPS).


## Camera Calibration and Scaling

To convert pixel distances obtained from tracking into real-world measurements necessary for speed estimation, a simplified camera calibration approach was implemented. Leveraging the known approximate real-world width of pedestrian crossing stripes (assumed to be **0.5 meters**), a scale factor was derived based on their corresponding pixel length in the video.

Initially, direct pixel measurements of different stripes yielded a median value of approximately **66 pixels** for 0.5 meters. However, using this directly resulted in unexpectedly low-speed estimations. Given that the video is captured from a near bird's-eye view, suggesting less severe perspective distortion, the inaccuracy was hypothesized to stem from imprecise pixel measurements or subtle perspective effects not accounted for.

To obtain more plausible speed results, an iterative approach was taken, making an assumption about the pixel-to-meter scale factor that yielded more promising speed estimations for the vehicles in the video. This resulted in an approximate conversion of **1 pixel equating to 0.05 meters**. While this introduces an assumption, it provided a working scale for speed calculation within the prototype. A more rigorous calibration would be necessary for precise measurements.



## Speed Estimation

The speed of each tracked vehicle was estimated based on its displacement between frames. Using the central (x, y) coordinates of the vehicle's bounding box provided by the tracking algorithm, the pixel distance covered by the vehicle over a specific time interval was calculated. A deque (double-ended queue) was used to store a history of recent bounding box positions for each tracked ID, enabling the calculation of displacement over an interval corresponding to the video's frame rate (FPS).

The calculated pixel displacement was then converted into a real-world distance in meters using the pixel-to-meter scale factor derived during the calibration step. The time interval between the frames used for displacement calculation was determined directly from the video's FPS (Time Interval = 1 / FPS).

Finally, the vehicle's speed was computed using the formula: Speed = Real-World Distance / Time Interval. The resulting speed, initially in meters per second (m/s), was then converted to kilometers per hour (km/h) for better interpretability.


## Challenges and Learnings

This project provided valuable insights into the practical challenges of applying computer vision techniques to real-world video analysis, particularly concerning accurate spatial measurements.

**Key Learnings:**

* Gained a deeper understanding of **camera calibration** concepts, including the principles behind **homography** and **vanishing point calculations**, and their importance in converting 2D pixel coordinates to 3D real-world distances in video analysis.
* Experienced firsthand how **perspective** significantly impacts measurements in images and videos, necessitating careful consideration during calibration.

**Challenges Faced:**

* A primary challenge was achieving accurate and consistent speed estimations due to variations in the pixel-to-real-world scale factor across the image, even with the near bird's-eye view of the dataset. Initial attempts to derive the scale factor from direct pixel measurements of known objects yielded inconsistent results.
* Although perspective transformation or vanishing point calculations were deemed less critical for this specific dataset's camera angle, relying solely on a single scale factor derived from one area introduced inaccuracies in speed calculations for vehicles in different parts of the frame.
* Using the central coordinates of detected bounding boxes for precise distance measurement proved challenging. The inherent "flicker" or slight variations in bounding box predictions by the object detection model can introduce noise into the tracking and subsequent speed calculations, making it difficult to achieve highly stable velocity estimations.

**Outcome:**

Despite these challenges, the project successfully implemented a pipeline for vehicle detection, tracking, and speed estimation. The challenges encountered highlight the complexities of real-world computer vision applications and underscore the importance of robust calibration and tracking methods for accurate quantitative analysis. An assumption regarding the pixel-to-meter scale factor was made to produce more plausible speed results within this prototype, acknowledging the need for more rigorous methods in a production environment.


## Results / Demo

Upon running the project, the output is a processed video stream of the traffic recording. The video displays the detected vehicles with their bounding boxes and unique tracking IDs. For each tracked vehicle, the estimated speed (in km/h) and its corresponding class name are overlaid onto the video frame.

![Vehicle Speed Estimation Demo](output-640.gif)


## How to Run

To run this project, follow these steps:

1.  Clone the repository:
    ```bash
    git clone [Your Repository URL]
    ```
2.  Navigate to the project directory:
    ```bash
    cd [Your Project Folder Name]
    ```
3.  (Optional) Create a virtual environment (recommended for managing dependencies):
    ```bash
    python -m venv venv
    ```
4.  Activate the virtual environment:
    * **macOS and Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows:**
        ```bash
        venv\Scripts\activate
        ```
5.  Install the required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
6.  Place the video file you want to analyze (e.g., the Rouen video) in a specified location and update the `video_path` variable in the `main` function of `speed_estimation_pipeline.py` accordingly.
7.  Run the main script:
    ```bash
    python speed_estimation_pipeline.py
    ```

The processed video with vehicle detections, tracks, and speed estimations will be saved as `output.mp4` (or the specified output path).



## Contact

You can reach out to me via:

* **GitHub:** https://github.com/eardage
* **LinkedIn:** https://www.linkedin.com/in/egeardaozturk/
