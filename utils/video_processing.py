import cv2
import numpy as np
from typing import Optional, Tuple, List
import time
import threading
from collections import deque

class VideoProcessor:
    """Video processing utilities for the sign language system"""
    
    def __init__(self, buffer_size: int = 30):
        """
        Initialize video processor
        
        Args:
            buffer_size: Size of frame buffer for processing
        """
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Video enhancement settings
        self.brightness_adjustment = 0
        self.contrast_adjustment = 1.0
        self.blur_kernel_size = 0  # 0 means no blur
        
        print(f"✅ Video processor initialized with buffer size: {buffer_size}")
    
    def enhance_frame(self, frame: np.ndarray, 
                     brightness: int = 0, 
                     contrast: float = 1.0,
                     apply_denoising: bool = False) -> np.ndarray:
        """
        Enhance frame quality for better gesture recognition
        
        Args:
            frame: Input frame
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast adjustment (0.0 to 3.0)
            apply_denoising: Whether to apply denoising
            
        Returns:
            Enhanced frame
        """
        try:
            enhanced = frame.copy()
            
            # Apply brightness and contrast adjustments
            if brightness != 0 or contrast != 1.0:
                enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=brightness)
            
            # Apply denoising if requested
            if apply_denoising:
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            # Apply slight Gaussian blur to reduce noise (optional)
            if self.blur_kernel_size > 0:
                enhanced = cv2.GaussianBlur(enhanced, (self.blur_kernel_size, self.blur_kernel_size), 0)
            
            return enhanced
            
        except Exception as e:
            print(f"❌ Error enhancing frame: {str(e)}")
            return frame
    
    def resize_frame(self, frame: np.ndarray, 
                    target_width: int = 640, 
                    target_height: int = 480,
                    maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize frame to target dimensions
        
        Args:
            frame: Input frame
            target_width: Target width
            target_height: Target height
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized frame
        """
        try:
            if maintain_aspect:
                h, w = frame.shape[:2]
                aspect_ratio = w / h
                
                if aspect_ratio > (target_width / target_height):
                    # Width is the limiting factor
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:
                    # Height is the limiting factor
                    new_height = target_height
                    new_width = int(target_height * aspect_ratio)
                
                resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Pad if necessary to reach target dimensions
                if new_width < target_width or new_height < target_height:
                    delta_w = target_width - new_width
                    delta_h = target_height - new_height
                    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                    left, right = delta_w // 2, delta_w - (delta_w // 2)
                    
                    resized = cv2.copyMakeBorder(
                        resized, top, bottom, left, right, 
                        cv2.BORDER_CONSTANT, value=[0, 0, 0]
                    )
                
                return resized
            else:
                return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                
        except Exception as e:
            print(f"❌ Error resizing frame: {str(e)}")
            return frame
    
    def add_frame_to_buffer(self, frame: np.ndarray):
        """Add frame to processing buffer"""
        try:
            timestamp = time.time()
            self.frame_buffer.append({
                'frame': frame.copy(),
                'timestamp': timestamp
            })
        except Exception as e:
            print(f"❌ Error adding frame to buffer: {str(e)}")
    
    def get_recent_frames(self, count: int = 10) -> List[np.ndarray]:
        """
        Get recent frames from buffer
        
        Args:
            count: Number of recent frames to return
            
        Returns:
            List of recent frames
        """
        try:
            if len(self.frame_buffer) == 0:
                return []
            
            recent_frames = list(self.frame_buffer)[-count:]
            return [item['frame'] for item in recent_frames]
            
        except Exception as e:
            print(f"❌ Error getting recent frames: {str(e)}")
            return []
    
    def calculate_fps(self) -> float:
        """
        Calculate current FPS
        
        Returns:
            Current FPS
        """
        try:
            self.fps_counter += 1
            current_time = time.time()
            
            if current_time - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = current_time
            
            return self.current_fps
            
        except Exception as e:
            print(f"❌ Error calculating FPS: {str(e)}")
            return 0.0
    
    def detect_motion(self, threshold: float = 0.1) -> bool:
        """
        Detect motion between recent frames
        
        Args:
            threshold: Motion detection threshold
            
        Returns:
            True if motion detected
        """
        try:
            if len(self.frame_buffer) < 2:
                return False
            
            # Get last two frames
            frame1 = self.frame_buffer[-2]['frame']
            frame2 = self.frame_buffer[-1]['frame']
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Calculate motion score
            motion_score = np.mean(diff) / 255.0
            
            return motion_score > threshold
            
        except Exception as e:
            print(f"❌ Error detecting motion: {str(e)}")
            return False
    
    def stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply simple frame stabilization
        
        Args:
            frame: Input frame
            
        Returns:
            Stabilized frame
        """
        try:
            if len(self.frame_buffer) < 2:
                return frame
            
            # Get previous frame
            prev_frame = self.frame_buffer[-2]['frame']
            
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect features
            corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
            
            if corners is not None:
                # Calculate optical flow
                next_corners, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)
                
                # Filter good points
                good_corners = corners[status == 1]
                good_next = next_corners[status == 1]
                
                if len(good_corners) >= 4:
                    # Estimate transformation
                    transform = cv2.estimateAffinePartial2D(good_corners, good_next)[0]
                    
                    if transform is not None:
                        # Apply inverse transformation for stabilization
                        stabilized = cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]))
                        return stabilized
            
            return frame
            
        except Exception as e:
            print(f"❌ Error stabilizing frame: {str(e)}")
            return frame
    
    def add_overlay_info(self, frame: np.ndarray, 
                        gesture_info: Optional[str] = None,
                        confidence: Optional[float] = None,
                        fps: Optional[float] = None) -> np.ndarray:
        """
        Add overlay information to frame
        
        Args:
            frame: Input frame
            gesture_info: Gesture information to display
            confidence: Confidence score
            fps: FPS to display
            
        Returns:
            Frame with overlay
        """
        try:
            overlay_frame = frame.copy()
            
            # Add semi-transparent background for text
            overlay = overlay_frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay_frame, 0.7, overlay, 0.3, 0, overlay_frame)
            
            y_offset = 30
            
            # Add FPS info
            if fps is not None:
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(overlay_frame, fps_text, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            # Add gesture info
            if gesture_info:
                gesture_text = f"Gesture: {gesture_info}"
                cv2.putText(overlay_frame, gesture_text, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            # Add confidence info
            if confidence is not None:
                confidence_text = f"Confidence: {confidence:.2f}"
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
                cv2.putText(overlay_frame, confidence_text, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return overlay_frame
            
        except Exception as e:
            print(f"❌ Error adding overlay: {str(e)}")
            return frame
    
    def extract_hand_region(self, frame: np.ndarray, 
                           hand_bbox: Tuple[int, int, int, int],
                           padding: int = 20) -> Optional[np.ndarray]:
        """
        Extract hand region from frame
        
        Args:
            frame: Input frame
            hand_bbox: Hand bounding box (x_min, y_min, x_max, y_max)
            padding: Padding around hand region
            
        Returns:
            Extracted hand region
        """
        try:
            x_min, y_min, x_max, y_max = hand_bbox
            h, w = frame.shape[:2]
            
            # Add padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Extract region
            hand_region = frame[y_min:y_max, x_min:x_max]
            
            return hand_region if hand_region.size > 0 else None
            
        except Exception as e:
            print(f"❌ Error extracting hand region: {str(e)}")
            return None
    
    def apply_background_subtraction(self, frame: np.ndarray, 
                                   background_subtractor = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply background subtraction to focus on moving objects (hands)
        
        Args:
            frame: Input frame
            background_subtractor: OpenCV background subtractor
            
        Returns:
            Tuple of (foreground_mask, processed_frame)
        """
        try:
            # Create background subtractor if not provided
            if background_subtractor is None:
                background_subtractor = cv2.createBackgroundSubtractorMOG2(
                    detectShadows=True, varThreshold=50
                )
            
            # Apply background subtraction
            fg_mask = background_subtractor.apply(frame)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Apply mask to original frame
            processed_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
            
            return fg_mask, processed_frame
            
        except Exception as e:
            print(f"❌ Error in background subtraction: {str(e)}")
            return np.zeros(frame.shape[:2], dtype=np.uint8), frame
    
    def get_buffer_stats(self) -> dict:
        """Get statistics about the frame buffer"""
        try:
            if not self.frame_buffer:
                return {'count': 0, 'duration': 0.0}
            
            timestamps = [item['timestamp'] for item in self.frame_buffer]
            duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
            
            return {
                'count': len(self.frame_buffer),
                'duration': duration,
                'fps': self.current_fps,
                'buffer_size': self.buffer_size
            }
            
        except Exception as e:
            print(f"❌ Error getting buffer stats: {str(e)}")
            return {'count': 0, 'duration': 0.0}
    
    def clear_buffer(self):
        """Clear the frame buffer"""
        try:
            self.frame_buffer.clear()
            print("✅ Frame buffer cleared")
        except Exception as e:
            print(f"❌ Error clearing buffer: {str(e)}")
