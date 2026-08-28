# LaneInfo.msg - Lane detection and status information

# Lane gradient/slope
float32 slope

# Target points for path planning
TargetPoint[] target_points

# Car center position
float32 car_center_x
float32 car_center_y

# Lane status
string current_lane    # "lane1" or "lane2"
string target_lane     # "lane1" or "lane2"
bool lane_change       # true: lane change in progress, false: normal driving