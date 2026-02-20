import time
import json
import cv2
import numpy as np
from pal.products.qcar import QCar, QCarCameras, IS_PHYSICAL_QCAR
from pal.utilities.lidar import Lidar



# =========================================================

# ---------------- Configuration --------------------------

# =========================================================

OUT_PATH = "/tmp/qcar_avoid_info.json"

STOP_DISTANCE = 0.40

FRONT_ARC_START = 45

FRONT_ARC_END = 135

NUM_MEASUREMENTS = 360

LIDAR_UPDATE_RATE = 10



# Lane following parameters

FORWARD_SPEED = 0.07

STEERING_GAIN = 0.005

STEERING_CLIP = 0.5

MIN_AREA = 500



# Path-based avoidance durations (~20 cm each)

SHIFT_TIME = .75

FORWARD_PASS_TIME =1

RETURN_TIME = 1

AVOID_SPEED = 0.07

RETURN_SPEED = 0.07



# Cooldown (to prevent repeated avoidance)

COOLDOWN_TIME = 5.0



# =========================================================

# ---------------- Steering Calibration -------------------

# =========================================================

# ✅ These have been corrected for your QCar hardware orientation.

SHIFT_STEER_RIGHT = -0.4    # steering to move RIGHT during SHIFT (left obstacle)

SHIFT_STEER_LEFT  = +0.4    # steering to move LEFT during SHIFT (right obstacle)

RETURN_STEER_RIGHT = -0.9  # steering to turn RIGHT back toward lane

RETURN_STEER_LEFT  = +0.9  # steering to turn LEFT back toward lane



# =========================================================

# ---------------- Helper Function ------------------------

# =========================================================

def write_json(data):

    try:

        with open(OUT_PATH, "w") as f:

            json.dump(data, f)

    except:

        pass





# =========================================================

# ---------------- Initialization -------------------------

# =========================================================

if not IS_PHYSICAL_QCAR:

    import qlabs_setup

    qlabs_setup.setup()



print("✅ Starting Final QCar Path-Planned Obstacle Avoidance")



lidar = Lidar(type='RPLidar', numMeasurements=NUM_MEASUREMENTS,

              rangingDistanceMode=2, interpolationMode=0)

write_json({"command": "STOP", "ts": time.time()})



cameras = QCarCameras(enableFront=True)

car = QCar(readMode=1, frequency=200)



# =========================================================

# ---------------- Main FSM -------------------------------

# =========================================================

try:

    with cameras, car:

        state = "LANE_FOLLOW"

        avoid_cmd = "GO"

        locked_avoid_cmd = None

        shift_dir = None

        maneuver_start = time.time()

        avoid_cooldown = 0.0



        lidar_period = 1.0 / LIDAR_UPDATE_RATE

        last_lidar_time = 0.0

        front_min = float("inf")

        mean_angle = None



        print("🚗 Running (Final version: symmetric, loop-free)")



        while True:

            t = time.time()



            # ---------- LiDAR Reading ----------

            if (t - last_lidar_time) >= lidar_period:

                last_lidar_time = t

                try:

                    lidar.read()

                    ang = np.degrees(lidar.angles)

                    dist = np.array(lidar.distances)

                    ang = (ang + 180) % 360 - 180



                    front_mask = (ang >= FRONT_ARC_START) & (ang <= FRONT_ARC_END)

                    dist_front = dist[front_mask]

                    valid_front = (dist_front > 0) & (dist_front < STOP_DISTANCE)



                    if np.any(valid_front):

                        close_angles = ang[front_mask][valid_front]

                        close_dists = dist_front[valid_front]

                        front_min = float(np.min(close_dists))

                        mean_angle = float(np.mean(close_angles))

                    else:

                        front_min = float("inf")

                        mean_angle = None

                except:

                    front_min = float("inf")

                    mean_angle = None



                # ✅ Detect obstacle only when not cooling down

                if state == "LANE_FOLLOW" and (t > avoid_cooldown):

                    if (mean_angle is not None) and (front_min < STOP_DISTANCE):

                        # mean_angle > 0 → obstacle on LEFT → go RIGHT

                        # mean_angle < 0 → obstacle on RIGHT → go LEFT

                        if mean_angle > 90:

                            avoid_cmd = "AVOID_RIGHT"

                        else:

                            avoid_cmd = "AVOID_LEFT"

                    else:

                        avoid_cmd = "GO"

                else:

                    avoid_cmd = "GO"  # ignore during cooldown



            # ---------- Camera Lane Detection ----------

            cameras.readAll()

            frame = cameras.csiFront.imageData

            if frame is None:

                continue



            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_yellow = np.array([18, 94, 140])

            upper_yellow = np.array([48, 255, 255])

            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            mask = cv2.GaussianBlur(mask, (7, 7), 0)



            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            steering = 0.0

            if contours:

                largest = max(contours, key=cv2.contourArea)

                if cv2.contourArea(largest) > MIN_AREA:

                    x, y, w, h = cv2.boundingRect(largest)

                    cx = x + w // 2

                    center_x = frame.shape[1] // 2

                    error = center_x - cx

                    steering = np.clip(error * STEERING_GAIN, -STEERING_CLIP, STEERING_CLIP)



            # ---------- FSM Behavior ----------

            forward = FORWARD_SPEED



            # LANE FOLLOW

            if state == "LANE_FOLLOW":

                if avoid_cmd in ["AVOID_LEFT", "AVOID_RIGHT"]:

                    locked_avoid_cmd = avoid_cmd

                    shift_dir = "LEFT" if avoid_cmd == "AVOID_RIGHT" else "RIGHT"

                    state = "SHIFT"

                    maneuver_start = t

                    print(f"⚠ Obstacle detected: mean_angle={mean_angle:.2f}, shift_dir={shift_dir}")



            # SHIFT (move sideways)

            elif state == "SHIFT":

                forward = AVOID_SPEED

                if shift_dir == "RIGHT":

                    steering = SHIFT_STEER_RIGHT

                elif shift_dir == "LEFT":

                    steering = SHIFT_STEER_LEFT

                if (t - maneuver_start) > SHIFT_TIME:

                    state = "FORWARD_PASS"

                    maneuver_start = t

                    print(f"➡ SHIFT done ({shift_dir}) → FORWARD_PASS")



            # FORWARD_PASS (go straight)

            elif state == "FORWARD_PASS":

                forward = AVOID_SPEED

                steering = 0.0

                if (t - maneuver_start) > FORWARD_PASS_TIME:

                    state = "RETURN"

                    maneuver_start = t

                    print(f"↩ FORWARD_PASS done → RETURN")



            # RETURN (rejoin lane)

            elif state == "RETURN":

                forward = RETURN_SPEED

                if shift_dir == "RIGHT":

                    steering = RETURN_STEER_LEFT

                elif shift_dir == "LEFT":

                    steering = RETURN_STEER_RIGHT

                if (t - maneuver_start) > RETURN_TIME:

                    state = "LANE_FOLLOW"

                    locked_avoid_cmd = None

                    shift_dir = None

                    avoid_cooldown = t + COOLDOWN_TIME

                    print(f"🏁 Avoidance complete → LANE_FOLLOW (cooldown {COOLDOWN_TIME}s)")



            # ---------- Drive ----------

            car.write(forward, steering, np.array([0, 0, 0, 0, 0, 0, 1, 1]))



            # ---------- Display ----------

            status = f"{state} | shift={shift_dir} | avoid={avoid_cmd} | mean={mean_angle if mean_angle else 'None'} | front={front_min:.2f}"

            cv2.putText(frame, status, (10, frame.shape[0]-10),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow("QCar Path-Planned Avoidance", frame)

            cv2.imshow("Mask", mask)



            print(f"[{state}] shift_dir={shift_dir}, avoid={avoid_cmd}, mean_angle={mean_angle}, steering={steering:+.2f}, front={front_min:.2f}")



            if cv2.waitKey(1) & 0xFF == ord('q'):

                break



except KeyboardInterrupt:

    print("\n🛑 Interrupted by user.")

finally:

    try:

        car.write(0.0, 0.0, np.zeros(8))

        lidar.terminate()

    except:

        pass

    write_json({"command": "STOP", "ts": time.time()})

    cv2.destroyAllWindows()
#!/usr/bin/env python3
"""
qcar_lane_follow_with_path_planned_avoidance_final.py
-----------------------------------------------------
✅ Final version — symmetric, correct for both sides, loop-free.
✅ If obstacle is on LEFT → move RIGHT → straight → LEFT.
✅ If obstacle is on RIGHT → move LEFT → straight → RIGHT.
"""

import time
import json
import cv2
import numpy as np
from pal.products.qcar import QCar, QCarCameras, IS_PHYSICAL_QCAR
from pal.utilities.lidar import Lidar

# =========================================================
# ---------------- Configuration --------------------------
# =========================================================
OUT_PATH = "/tmp/qcar_avoid_info.json"
STOP_DISTANCE = 0.40
FRONT_ARC_START = 45
FRONT_ARC_END = 135
NUM_MEASUREMENTS = 360
LIDAR_UPDATE_RATE = 10

# Lane following parameters
FORWARD_SPEED = 0.07
STEERING_GAIN = 0.005
STEERING_CLIP = 0.5
MIN_AREA = 500

# Path-based avoidance durations (~20 cm each)
SHIFT_TIME = .75
FORWARD_PASS_TIME =1
RETURN_TIME = 1
AVOID_SPEED = 0.07
RETURN_SPEED = 0.07

# Cooldown (to prevent repeated avoidance)
COOLDOWN_TIME = 5.0

# =========================================================
# ---------------- Steering Calibration -------------------
# =========================================================
# ✅ These have been corrected for your QCar hardware orientation.
SHIFT_STEER_RIGHT = -0.4    # steering to move RIGHT during SHIFT (left obstacle)
SHIFT_STEER_LEFT  = +0.4    # steering to move LEFT during SHIFT (right obstacle)
RETURN_STEER_RIGHT = -0.9  # steering to turn RIGHT back toward lane
RETURN_STEER_LEFT  = +0.9  # steering to turn LEFT back toward lane

# =========================================================
# ---------------- Helper Function ------------------------
# =========================================================
def write_json(data):
    try:
        with open(OUT_PATH, "w") as f:
            json.dump(data, f)
    except:
        pass


# =========================================================
# ---------------- Initialization -------------------------
# =========================================================
if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup()

print("✅ Starting Final QCar Path-Planned Obstacle Avoidance")

lidar = Lidar(type='RPLidar', numMeasurements=NUM_MEASUREMENTS,
              rangingDistanceMode=2, interpolationMode=0)
write_json({"command": "STOP", "ts": time.time()})

cameras = QCarCameras(enableFront=True)
car = QCar(readMode=1, frequency=200)

# =========================================================
# ---------------- Main FSM -------------------------------
# =========================================================
try:
    with cameras, car:
        state = "LANE_FOLLOW"
        avoid_cmd = "GO"
        locked_avoid_cmd = None
        shift_dir = None
        maneuver_start = time.time()
        avoid_cooldown = 0.0

        lidar_period = 1.0 / LIDAR_UPDATE_RATE
        last_lidar_time = 0.0
        front_min = float("inf")
        mean_angle = None

        print("🚗 Running (Final version: symmetric, loop-free)")

        while True:
            t = time.time()

            # ---------- LiDAR Reading ----------
            if (t - last_lidar_time) >= lidar_period:
                last_lidar_time = t
                try:
                    lidar.read()
                    ang = np.degrees(lidar.angles)
                    dist = np.array(lidar.distances)
                    ang = (ang + 180) % 360 - 180

                    front_mask = (ang >= FRONT_ARC_START) & (ang <= FRONT_ARC_END)
                    dist_front = dist[front_mask]
                    valid_front = (dist_front > 0) & (dist_front < STOP_DISTANCE)

                    if np.any(valid_front):
                        close_angles = ang[front_mask][valid_front]
                        close_dists = dist_front[valid_front]
                        front_min = float(np.min(close_dists))
                        mean_angle = float(np.mean(close_angles))
                    else:
                        front_min = float("inf")
                        mean_angle = None
                except:
                    front_min = float("inf")
                    mean_angle = None

                # ✅ Detect obstacle only when not cooling down
                if state == "LANE_FOLLOW" and (t > avoid_cooldown):
                    if (mean_angle is not None) and (front_min < STOP_DISTANCE):
                        # mean_angle > 0 → obstacle on LEFT → go RIGHT
                        # mean_angle < 0 → obstacle on RIGHT → go LEFT
                        if mean_angle > 90:
                            avoid_cmd = "AVOID_RIGHT"
                        else:
                            avoid_cmd = "AVOID_LEFT"
                    else:
                        avoid_cmd = "GO"
                else:
                    avoid_cmd = "GO"  # ignore during cooldown

            # ---------- Camera Lane Detection ----------
            cameras.readAll()
            frame = cameras.csiFront.imageData
            if frame is None:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([18, 94, 140])
            upper_yellow = np.array([48, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            steering = 0.0
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > MIN_AREA:
                    x, y, w, h = cv2.boundingRect(largest)
                    cx = x + w // 2
                    center_x = frame.shape[1] // 2
                    error = center_x - cx
                    steering = np.clip(error * STEERING_GAIN, -STEERING_CLIP, STEERING_CLIP)

            # ---------- FSM Behavior ----------
            forward = FORWARD_SPEED

            # LANE FOLLOW
            if state == "LANE_FOLLOW":
                if avoid_cmd in ["AVOID_LEFT", "AVOID_RIGHT"]:
                    locked_avoid_cmd = avoid_cmd
                    shift_dir = "LEFT" if avoid_cmd == "AVOID_RIGHT" else "RIGHT"
                    state = "SHIFT"
                    maneuver_start = t
                    print(f"⚠ Obstacle detected: mean_angle={mean_angle:.2f}, shift_dir={shift_dir}")

            # SHIFT (move sideways)
            elif state == "SHIFT":
                forward = AVOID_SPEED
                if shift_dir == "RIGHT":
                    steering = SHIFT_STEER_RIGHT
                elif shift_dir == "LEFT":
                    steering = SHIFT_STEER_LEFT
                if (t - maneuver_start) > SHIFT_TIME:
                    state = "FORWARD_PASS"
                    maneuver_start = t
                    print(f"➡ SHIFT done ({shift_dir}) → FORWARD_PASS")

            # FORWARD_PASS (go straight)
            elif state == "FORWARD_PASS":
                forward = AVOID_SPEED
                steering = 0.0
                if (t - maneuver_start) > FORWARD_PASS_TIME:
                    state = "RETURN"
                    maneuver_start = t
                    print(f"↩ FORWARD_PASS done → RETURN")

            # RETURN (rejoin lane)
            elif state == "RETURN":
                forward = RETURN_SPEED
                if shift_dir == "RIGHT":
                    steering = RETURN_STEER_LEFT
                elif shift_dir == "LEFT":
                    steering = RETURN_STEER_RIGHT
                if (t - maneuver_start) > RETURN_TIME:
                    state = "LANE_FOLLOW"
                    locked_avoid_cmd = None
                    shift_dir = None
                    avoid_cooldown = t + COOLDOWN_TIME
                    print(f"🏁 Avoidance complete → LANE_FOLLOW (cooldown {COOLDOWN_TIME}s)")

            # ---------- Drive ----------
            car.write(forward, steering, np.array([0, 0, 0, 0, 0, 0, 1, 1]))

            # ---------- Display ----------
            status = f"{state} | shift={shift_dir} | avoid={avoid_cmd} | mean={mean_angle if mean_angle else 'None'} | front={front_min:.2f}"
            cv2.putText(frame, status, (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("QCar Path-Planned Avoidance", frame)
            cv2.imshow("Mask", mask)

            print(f"[{state}] shift_dir={shift_dir}, avoid={avoid_cmd}, mean_angle={mean_angle}, steering={steering:+.2f}, front={front_min:.2f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")
finally:
    try:
        car.write(0.0, 0.0, np.zeros(8))
        lidar.terminate()
    except:
        pass
    write_json({"command": "STOP", "ts": time.time()})
    cv2.destroyAllWindows()
    print("✅ QCar stopped safely.")
    print("✅ QCar stopped safely.")
