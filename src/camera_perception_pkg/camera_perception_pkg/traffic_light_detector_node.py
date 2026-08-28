import cv2
import random
import numpy as np
from typing import Tuple
import sys, os
import matplotlib.pyplot as plt
 
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
 
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
 
from sensor_msgs.msg import Image
from interfaces_pkg.msg import DetectionArray, BoundingBox2D, Detection
from std_msgs.msg import String
 
from .lib import camera_perception_func_lib as CPFL
 
# ---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_IMAGE_TOPIC_NAME = "image_raw"
 
# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_traffic_light_info"
 
# ----------------------------------------------
 
class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector_node')
 
        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_image_topic = self.declare_parameter('sub_image_topic', SUB_IMAGE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
 
        self.cv_bridge = CvBridge()
 
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
 
        # =========================================================
        # HSV 색상 범위 정의 (신규)
        # =========================================================
        # 이 값들은 환경에 맞게 미세 조정(Tuning)이 필요합니다.
        self.hsv_ranges = {
            # 빨간색 (Hue가 0/179 양쪽에 걸쳐있음)
            'red1': (np.array([0, 70, 50]), np.array([10, 255, 255])),
            'red2': (np.array([170, 70, 50]), np.array([179, 255, 255])),
            # 노란색
            'yellow': (np.array([20, 100, 100]), np.array([35, 255, 255])),
            # 초록색
            'green': (np.array([40, 70, 50]), np.array([80, 255, 255])),
        }
        # =========================================================
 
        self.detection_sub = Subscriber(self, DetectionArray, self.sub_detection_topic, qos_profile=self.qos_profile)
        self.image_sub = Subscriber(self, Image, self.sub_image_topic, qos_profile=self.qos_profile)
        self.ts = ApproximateTimeSynchronizer([self.detection_sub, self.image_sub], queue_size=1, slop=0.5)
        self.ts.registerCallback(self.sync_callback)
 
        self.publisher = self.create_publisher(String, self.pub_topic, self.qos_profile)
 
    def sync_callback(self, detection_msg: DetectionArray, image_msg: Image):
        # -----------------------------
        # 1. 기본 설정 및 원본 영상 표시
        # -----------------------------
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg)
        #cv2.imshow("Camera Feed", cv_image) # 원본 영상 창
        
        traffic_light_detected = False
        traffic_light_color = 'Unknown'

        for detection in detection_msg.detections:
            if detection.class_name == 'traffic_light':
                # -----------------------------
                # 2. 신호등 영역 추출 (ROI)
                # -----------------------------
                bbox = detection.bbox
                x_min = int(bbox.center.position.x - bbox.size.x / 2)
                x_max = int(bbox.center.position.x + bbox.size.x / 2)
                y_min = int(bbox.center.position.y - bbox.size.y / 2)
                y_max = int(bbox.center.position.y + bbox.size.y / 2)
                
                output = cv_image[y_min:y_max, x_min:x_max]
                if output.size == 0:
                    continue

                visual_output = output.copy()

                # -----------------------------
                # 3. 이미지 전처리 및 색상 마스크 생성 (신규 방식)
                # -----------------------------
                gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                hsv_roi = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

                # HSV 범위로 전체 ROI에 대한 색상 마스크 생성
                red_mask1 = cv2.inRange(hsv_roi, self.hsv_ranges['red1'][0], self.hsv_ranges['red1'][1])
                red_mask2 = cv2.inRange(hsv_roi, self.hsv_ranges['red2'][0], self.hsv_ranges['red2'][1])
                red_mask = red_mask1 + red_mask2
                yellow_mask = cv2.inRange(hsv_roi, self.hsv_ranges['yellow'][0], self.hsv_ranges['yellow'][1])
                green_mask = cv2.inRange(hsv_roi, self.hsv_ranges['green'][0], self.hsv_ranges['green'][1])
                
                # -----------------------------
                # 4. 원 검출 (기존 방식)
                # -----------------------------
                circles = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, dp=3, minDist=75,
                    param1=100, param2=30, minRadius=25, maxRadius=40
                )

                # -----------------------------
                # 5. 하이브리드 판별 로직 (신규)
                # -----------------------------
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    
                    best_color = "Unknown"
                    max_pixel_count = 0 # 켜진 등을 찾기 위한 픽셀 수 임계값
                    best_circle = None

                    for c in circles[0, :]:
                        cx, cy, r = c[0], c[1], c[2]
                        # (A) 모든 검출된 원 그리기 (파란색)
                        cv2.circle(visual_output, (cx, cy), c[2], (255, 0, 0), 2)
                        
                        # (B) 현재 원(Circle)에 대한 마스크 생성
                        circle_mask = np.zeros_like(gray)
                        cv2.circle(circle_mask, (cx, cy), r, 255, -1)
                        
                        # (C) 원 안의 색상 픽셀 수 계산
                        red_pixels = cv2.countNonZero(cv2.bitwise_and(red_mask, circle_mask))
                        yellow_pixels = cv2.countNonZero(cv2.bitwise_and(yellow_mask, circle_mask))
                        green_pixels = cv2.countNonZero(cv2.bitwise_and(green_mask, circle_mask))

                        # (D) 이 원에서 가장 많이 검출된 색상 찾기
                        current_max_pixels = max(red_pixels, yellow_pixels, green_pixels)
                        
                        # (E) 이 원이 이전에 찾은 '켜진 등'보다 더 확실한지 확인
                        if current_max_pixels > max_pixel_count:
                            max_pixel_count = current_max_pixels
                            best_circle = (cx, cy, r)
                            
                            # 이 원의 색상 결정
                            if max_pixel_count == red_pixels:
                                best_color = "Red"
                            elif max_pixel_count == yellow_pixels:
                                best_color = "Yellow"
                            elif max_pixel_count == green_pixels:
                                best_color = "Green"
                    
                    traffic_light_color = best_color
                    
                    # (F) 가장 확실한 원(켜진 등)을 초록색으로 다시 그리기
                    if best_circle is not None:
                        cx, cy, r = best_circle
                        cv2.circle(visual_output, (cx, cy), r, (0, 255, 0), 3)

                
                # (G) 최종 판별 결과 텍스트로 표시
                cv2.putText(visual_output, f"Detected: {traffic_light_color}", 
                            (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # (H) 분석 결과 창 표시
                #cv2.imshow("Traffic Light Analysis", visual_output)
                
                traffic_light_detected = True
                break

        # -----------------------------
        # 6. 최종 결과 Publish
        # -----------------------------
        color_msg = String()
        color_msg.data = traffic_light_color if traffic_light_detected else 'None'
        self.get_logger().info(f'Traffic light: {color_msg.data}')
        self.publisher.publish(color_msg)
        
        #cv2.waitKey(1)
 
def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
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