import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray, BoundingBox2D, Detection
from .lib import camera_perception_func_lib as CPFL

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# 모드 관련 토픽들
SUB_DRIVING_MODE_TOPIC = "task_execution"  # "normal" or "task_execution"
SUB_NORMAL_MODE_TOPIC = "normal_driving_mode"  # "lane_keeping" or "lane_transition"
SUB_TASK_MODE_TOPIC = "both"  # "obstacle_avoidance" or "traffic_light" or "both"

#################################SELECT MODE#################################
#############################################################################
# 모드 설정
SELECT_DRIVING_MODE = "normal" # "normal" or "task_execution"
SELECT_LANE_MODE = "lane_keeping" # "lane_keeping" or "lane_transition"
SELECT_TASK_MODE = "both" # "obstacle_avoidance" or "traffic_light" or "both"
#############################################################################
#############################################################################

# 차선변경 명령 토픽들 (각 모드별로 구분)
SUB_NORMAL_LANE_CHANGE_TOPIC = "normal_lane_change_command"  # 일반 주행 차선변경
SUB_OBSTACLE_LANE_CHANGE_TOPIC = "obstacle_lane_change_command"  # 장애물 회피 차선변경

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"

# 초기 차선 설정
INITIAL_LANE = "lane2"  # "lane1" or "lane2"

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1하여 입력
SHOW_IMAGE = True
#----------------------------------------------


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        # 파라미터 선언
        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.sub_driving_mode_topic = self.declare_parameter('sub_driving_mode_topic', SUB_DRIVING_MODE_TOPIC).value
        self.sub_normal_mode_topic = self.declare_parameter('sub_normal_mode_topic', SUB_NORMAL_MODE_TOPIC).value
        self.sub_task_mode_topic = self.declare_parameter('sub_task_mode_topic', SUB_TASK_MODE_TOPIC).value
        self.sub_normal_lane_change_topic = self.declare_parameter('sub_normal_lane_change_topic', SUB_NORMAL_LANE_CHANGE_TOPIC).value
        self.sub_obstacle_lane_change_topic = self.declare_parameter('sub_obstacle_lane_change_topic', SUB_OBSTACLE_LANE_CHANGE_TOPIC).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value
        self.initial_lane = self.declare_parameter('initial_lane', INITIAL_LANE).value  # ⭐ 추가
        

        default_src_mat = [258, 301, 371, 301, 489, 479, 130, 479]
        self.declare_parameter('src_mat', default_src_mat)

        src_points_list = self.get_parameter('src_mat').get_parameter_value().integer_array_value

        if len(src_points_list) == 8:
            self.src_mat = [
                [src_points_list[0], src_points_list[1]], # [p1_x, p1_y]
                [src_points_list[2], src_points_list[3]], # [p2_x, p2_y]
                [src_points_list[4], src_points_list[5]], # [p3_x, p3_y]
                [src_points_list[6], src_points_list[7]]  # [p4_x, p4_y]
            ]
        else:
            self.get_logger().error("src_mat_points 파라미터는 반드시 8개의 정수여야 합니다! 기본값을 사용합니다.")
            self.src_mat = [
                [default_src_mat[0], default_src_mat[1]],
                [default_src_mat[2], default_src_mat[3]],
                [default_src_mat[4], default_src_mat[5]],
                [default_src_mat[6], default_src_mat[7]]
            ]
            
        self.get_logger().info(f"Bird-eye view src_mat 설정됨: {self.src_mat}")

        self.cv_bridge = CvBridge()

        # QoS settings
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Subscribers
        self.subscriber = self.create_subscription(
            DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        
        # 모드 관련 구독자
        self.driving_mode_subscriber = self.create_subscription(
            String, self.sub_driving_mode_topic, self.driving_mode_callback, self.qos_profile)
        self.normal_mode_subscriber = self.create_subscription(
            String, self.sub_normal_mode_topic, self.normal_mode_callback, self.qos_profile)
        self.task_mode_subscriber = self.create_subscription(
            String, self.sub_task_mode_topic, self.task_mode_callback, self.qos_profile)
        
        # 차선변경 명령 구독자 (각 모드별)
        self.normal_lane_change_subscriber = self.create_subscription(
            Bool, self.sub_normal_lane_change_topic, self.normal_lane_change_callback, self.qos_profile)
        self.obstacle_lane_change_subscriber = self.create_subscription(
            Bool, self.sub_obstacle_lane_change_topic, self.obstacle_lane_change_callback, self.qos_profile)
        
        # Publishers
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)
        self.lane_debug_img_pub = self.create_publisher(Image, "lane_debug_img", self.qos_profile)

        # 상태 변수들
        self.current_lane = self.initial_lane  # ⭐ 초기 차선 설정
        self.target_lane = self.initial_lane
        self.car_center_point = (320, 460) # (320, 400)

        # 모드 설정 파라미터
        self.select_driving_mode = self.declare_parameter('select_driving_mode', SELECT_DRIVING_MODE).value
        self.select_lane_mode = self.declare_parameter('select_lane_mode', SELECT_LANE_MODE).value
        self.select_task_mode = self.declare_parameter('select_task_mode', SELECT_TASK_MODE).value
        
        # 모드 상태
        self.driving_mode = self.select_driving_mode
        self.normal_driving_mode = self.select_lane_mode
        self.task_execution_mode = self.select_task_mode
        
        # 차선변경 플래그 (각 모드별)
        self.normal_lane_change = False
        self.obstacle_lane_change = False
        
        # 통합 차선변경 플래그 (LaneInfo로 전송용)
        self.lane_change = False
        
        self.get_logger().info(f"Lane Info Extractor initialized - Initial Lane: {self.initial_lane}")

    def draw_filled_masks(self, detection_msg, cls_name, color=255):
        h, w = 480, 640
        mask_image = np.zeros((h, w), dtype=np.uint8)
        
        for detection in detection_msg.detections:
            if detection.class_name == cls_name:
                if len(detection.mask.data) > 0:
                    points = []
                    for point in detection.mask.data:
                        x = int(point.x)
                        y = int(point.y)
                        points.append([x, y])
                    
                    if len(points) > 0:
                        points = np.array(points, dtype=np.int32)
                        cv2.fillPoly(mask_image, [points], color)
        
        return mask_image

    def driving_mode_callback(self, msg: String):
        self.driving_mode = msg.data
        self.get_logger().info(f"Driving mode changed to: {self.driving_mode}")
        self.update_lane_change_state()

    def normal_mode_callback(self, msg: String):
        self.normal_driving_mode = msg.data
        self.get_logger().info(f"Normal driving mode set to: {self.normal_driving_mode}")
        
        # lane_keeping 모드로 전환 시 normal_lane_change를 False로
        if self.normal_driving_mode == "lane_keeping":
            self.normal_lane_change = False
            self.get_logger().info("Lane keeping mode - normal lane change disabled")
        
        self.update_lane_change_state()

    def task_mode_callback(self, msg: String):
        self.task_execution_mode = msg.data
        self.get_logger().info(f"Task execution mode set to: {self.task_execution_mode}")
        
        # traffic_light 전용 모드일 때는 차선변경 비활성화
        if self.task_execution_mode == "traffic_light":
            self.obstacle_lane_change = False
            self.get_logger().info("Traffic light mode - obstacle lane change disabled")
        
        self.update_lane_change_state()

    def normal_lane_change_callback(self, msg: Bool):
        if self.driving_mode == "normal" and self.normal_driving_mode == "lane_transition":
            prev_state = self.normal_lane_change
            self.normal_lane_change = msg.data
            self.get_logger().info(f"Normal lane change: {self.normal_lane_change}")
            
            # 상태 전환 시 target_lane 설정
            if not prev_state and self.normal_lane_change:
                self.target_lane = 'lane2' if self.current_lane == 'lane1' else 'lane1'
                self.get_logger().info(f"Normal lane change ON - target: {self.target_lane}")
            elif prev_state and not self.normal_lane_change:
                self.target_lane = self.current_lane
                self.get_logger().info(f"Normal lane change OFF - target reset to current: {self.target_lane}")
        else:
            if msg.data:
                self.get_logger().warn("Normal lane change command ignored - not in lane_transition mode")
        
        self.update_lane_change_state()

    def obstacle_lane_change_callback(self, msg: Bool):
        if self.driving_mode == "task_execution" and self.task_execution_mode in ["obstacle_avoidance", "both"]:
            prev_state = self.obstacle_lane_change
            self.obstacle_lane_change = msg.data
            self.get_logger().info(f"Obstacle avoidance lane change: {self.obstacle_lane_change}")
            
            # 상태 전환 시 target_lane 설정
            if not prev_state and self.obstacle_lane_change:
                self.target_lane = 'lane2' if self.current_lane == 'lane1' else 'lane1'
                self.get_logger().info(f"Obstacle lane change ON - target: {self.target_lane}")
            elif prev_state and not self.obstacle_lane_change:
                self.target_lane = self.current_lane
                self.get_logger().info(f"Obstacle lane change OFF - target reset to current: {self.target_lane}")
        else:
            if msg.data:
                self.get_logger().warn("Obstacle lane change command ignored - not in obstacle_avoidance mode")
        
        self.update_lane_change_state()

    def update_lane_change_state(self):
        prev_lane_change = self.lane_change
        
        if self.driving_mode == "normal":
            # 일반 주행 모드
            if self.normal_driving_mode == "lane_transition":
                self.lane_change = self.normal_lane_change
            else:
                self.lane_change = False
        elif self.driving_mode == "task_execution":
            # 미션 수행 모드
            if self.task_execution_mode in ["obstacle_avoidance", "both"]:
                self.lane_change = self.obstacle_lane_change
            else:
                self.lane_change = False
        else:
            self.lane_change = False
        
        if prev_lane_change != self.lane_change:
            self.get_logger().info(f"Lane change state updated: {self.lane_change}")

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            return

        # edge 이미지 (경로 계획용) - 미리 준비
        lane1_edge_image = CPFL.draw_edges(detection_msg, cls_name='lane1', color=255)
        lane2_edge_image = CPFL.draw_edges(detection_msg, cls_name='lane2', color=255)

        # filled mask 이미지 (차선 변경 시 픽셀 비교용)
        lane1_mask_image = self.draw_filled_masks(detection_msg, cls_name='lane1', color=255)
        lane2_mask_image = self.draw_filled_masks(detection_msg, cls_name='lane2', color=255)

        # 차량 중심점 (LaneInfo 메시지로 전송하여 motion planner에서 장애물 감지에 활용)
        cx, cy = self.car_center_point

        # ============================================================
        # 현재 차선 업데이트 로직 (상태 기반)
        # - 현재 차선은 초기값(initial_lane)에서 시작
        # - 차선 변경 명령 시에만 target_lane이 설정됨 (callback에서 설정, 절대 변경 안됨)
        # - 차선 변경 중: x=320 세로선에서 각 차선 mask의 픽셀 개수 비교
        #   -> 더 많이 겹치는 차선으로 current_lane 업데이트
        # ============================================================
        if self.driving_mode == "normal" and self.normal_driving_mode == "lane_keeping":
            # Lane Keeping 모드: initial_lane만 유지
            self.current_lane = self.initial_lane
            self.target_lane = self.initial_lane
        else:
            # Lane Transition 또는 Task Execution 모드
            if self.lane_change:
                # 차선변경 진행 중
                # x=320 세로선에서 각 차선 mask의 픽셀 개수 비교
                x = 320
                lane1_pixels = np.sum(lane1_mask_image[:, x] > 0)
                lane2_pixels = np.sum(lane2_mask_image[:, x] > 0)

                self.get_logger().info(f"x={x} lane pixels: lane1={lane1_pixels}, lane2={lane2_pixels}")

                # 더 많이 겹치는 차선으로 현재 차선 업데이트
                if lane1_pixels > lane2_pixels and lane1_pixels > 0:
                    if self.current_lane != 'lane1':
                        self.get_logger().info(f"Lane changed: {self.current_lane} -> lane1")
                        self.current_lane = 'lane1'
                elif lane2_pixels > lane1_pixels and lane2_pixels > 0:
                    if self.current_lane != 'lane2':
                        self.get_logger().info(f"Lane changed: {self.current_lane} -> lane2")
                        self.current_lane = 'lane2'
                # else: 동점이거나 둘 다 0이면 current_lane 유지
            else:
                # 차선변경 명령 없음: target_lane을 current_lane과 동일하게 유지
                self.target_lane = self.current_lane
        
        mode_info = f"Mode: {self.driving_mode}"
        if self.driving_mode == "normal":
            mode_info += f"/{self.normal_driving_mode}"
        else:
            mode_info += f"/{self.task_execution_mode}"
        self.get_logger().info(f"📊 {mode_info} - Current: {self.current_lane}, Target: {self.target_lane}, LC: {self.lane_change}")

        # 항상 현재 차선 또는 목표 차선의 segment만 사용
        if self.lane_change:
            # 차선변경 중에는 목표 차선 사용
            if self.target_lane == 'lane1':
                lane_edge_image = lane1_edge_image
            elif self.target_lane == 'lane2':
                lane_edge_image = lane2_edge_image
            else:
                self.get_logger().warn("No valid target lane selected")
                return
        else:
            # 일반 주행 시 현재 차선 사용
            if self.current_lane == 'lane1':
                lane_edge_image = lane1_edge_image
            elif self.current_lane == 'lane2':
                lane_edge_image = lane2_edge_image
            else:
                self.get_logger().warn("No valid current lane selected")
                return

        # Edge image 유효성 검사
        if lane_edge_image is None or lane_edge_image.size == 0:
            self.get_logger().warn(f"Empty edge image for {'target' if self.lane_change else 'current'} lane: {self.target_lane if self.lane_change else self.current_lane}")
            return

        (h, w) = (lane_edge_image.shape[0], lane_edge_image.shape[1])
        dst_mat = [[round(w * 0.3), round(h * 0.0)], [round(w * 0.7), round(h * 0.0)], [round(w * 0.7), h], [round(w * 0.3), h]]
        # src_mat = [[281, 298],[407, 298], [545, 479], [127, 479]]
        #src_mat = [[258, 301],[371, 301], [489, 479], [130, 479]]
        lane_bird_image = CPFL.bird_convert(lane_edge_image, srcmat=self.src_mat, dstmat=dst_mat)
        #lane_bird_image = CPFL.bird_convert(lane_edge_image, srcmat=src_mat, dstmat=dst_mat)
        roi_image = CPFL.roi_rectangle_below(lane_bird_image, cutting_idx=300)

        
        mode_text = f"Mode: {self.driving_mode}"
        if self.driving_mode == "normal":
            sub_mode_text = f"Sub: {self.normal_driving_mode}"
        else:
            sub_mode_text = f"Task: {self.task_execution_mode}"
            
        lane_text = f"Current: {self.current_lane}"
        target_text = f"Target: {self.target_lane}"
        change_text = f"Lane Change: {'ON' if self.lane_change else 'OFF'}"

        mode_color = (255, 255, 255)
        sub_mode_color = (200, 200, 200)  
        lane_color = (255, 255, 0)
        target_color = (0, 255, 255)
        change_color = (0, 0, 255) if self.lane_change else (0, 255, 0)

        #display_img = lane_edge_image.copy()

        # ⭐️ 1. 엣지 이미지(아마도 CV_64F)를 8비트(CV_8U) 흑백 이미지로 변환
        display_img_8u = cv2.convertScaleAbs(lane_edge_image)
        # ⭐️ 2. 8비트 흑백 이미지를 BGR(컬러) 이미지로 변환 (텍스트를 입히기 위함)
        display_img_bgr = cv2.cvtColor(display_img_8u, cv2.COLOR_GRAY2BGR)

        cv2.putText(display_img_bgr, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2, cv2.LINE_AA)
        cv2.putText(display_img_bgr, sub_mode_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sub_mode_color, 2, cv2.LINE_AA)
        cv2.putText(display_img_bgr, lane_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lane_color, 2, cv2.LINE_AA)
        cv2.putText(display_img_bgr, target_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2, cv2.LINE_AA)
        cv2.putText(display_img_bgr, change_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, change_color, 2, cv2.LINE_AA)
            
        # 활성 차선변경 모드 표시
        if self.driving_mode == "normal" and self.normal_driving_mode == "lane_transition":
            status_text = f"Normal LC: {'ON' if self.normal_lane_change else 'OFF'}"
            cv2.putText(display_img_bgr, status_text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
        elif self.driving_mode == "task_execution" and self.task_execution_mode in ["obstacle_avoidance", "both"]:
            status_text = f"Obstacle LC: {'ON' if self.obstacle_lane_change else 'OFF'}"
            cv2.putText(display_img_bgr, status_text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2, cv2.LINE_AA)

        #cv2.imshow('lane_edge_image', display_img)
        #cv2.imshow('lane_bird_image', lane_bird_image)
        #cv2.imshow('roi_img', roi_image)
        #cv2.waitKey(1)

        roi_image = cv2.convertScaleAbs(roi_image)

        try:
            debug_img_msg = self.cv_bridge.cv2_to_imgmsg(display_img_bgr, encoding="bgr8")
            self.lane_debug_img_pub.publish(debug_img_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish lane debug image: {e}")
        
        grad = CPFL.dominant_gradient(roi_image, theta_limit=70)
                
        target_points = []
        for target_point_y in range(5, 155, 50):
            target_point_x = CPFL.get_lane_center(roi_image, detection_height=target_point_y, 
                                                detection_thickness=10, road_gradient=grad, lane_width=250)
            
            target_point = TargetPoint()
            target_point.target_x = round(target_point_x+5)
            target_point.target_y = round(target_point_y)
            target_points.append(target_point)

        # LaneInfo 메시지 생성 및 퍼블리시
        lane = LaneInfo()
        lane.slope = grad
        lane.target_points = target_points
        lane.car_center_x = float(cx)
        lane.car_center_y = float(cy)
        lane.current_lane = self.current_lane
        lane.target_lane = self.target_lane
        lane.lane_change = self.lane_change  # 통합 차선변경 플래그
        self.publisher.publish(lane)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        #cv2.destroyAllWindows()
        rclpy.shutdown()
  
if __name__ == '__main__':
    main()