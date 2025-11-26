"""
Night-Time and Low-Light Image Enhancement
For improved detection in poor lighting conditions
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class LowLightEnhancer:
    """
    Enhance images captured in low-light conditions
    Critical for Indian roads with poor street lighting
    """
    
    def __init__(self, enhancement_mode='auto'):
        """
        Args:
            enhancement_mode: 'auto', 'clahe', 'gamma', 'adaptive', 'fusion'
        """
        self.mode = enhancement_mode
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Adaptive parameters
        self.brightness_threshold = 80  # Below this is considered low-light
        
    def detect_lighting_condition(self, image: np.ndarray) -> str:
        """
        Detect if image is in low-light condition
        
        Args:
            image: Input image (BGR)
            
        Returns:
            condition: 'bright', 'normal', 'low', 'very_low'
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        if avg_brightness > 150:
            return 'bright'
        elif avg_brightness > 100:
            return 'normal'
        elif avg_brightness > 50:
            return 'low'
        else:
            return 'very_low'
    
    def enhance_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE enhancement
        Best for moderate low-light conditions
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def enhance_gamma(self, image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        """
        Apply gamma correction
        Good for very dark images
        
        Args:
            image: Input image
            gamma: Gamma value (>1 brightens, <1 darkens)
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in range(256)]).astype("uint8")
        
        # Apply gamma correction
        enhanced = cv2.LUT(image, table)
        
        return enhanced
    
    def enhance_adaptive(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive enhancement based on image statistics
        """
        # Split channels
        b, g, r = cv2.split(image)
        
        # Apply CLAHE to each channel
        b_enhanced = self.clahe.apply(b)
        g_enhanced = self.clahe.apply(g)
        r_enhanced = self.clahe.apply(r)
        
        # Merge channels
        enhanced = cv2.merge([b_enhanced, g_enhanced, r_enhanced])
        
        return enhanced
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce noise while preserving edges
        Important after brightness enhancement
        """
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None, h=10, hColor=10, 
            templateWindowSize=7, searchWindowSize=21
        )
        
        return denoised
    
    def enhance_headlight_glare(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce headlight glare effect
        Common issue in night-time rear-view cameras
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect bright regions (likely headlights)
        _, bright_mask = cv2.threshold(hsv[:, :, 2], 200, 255, cv2.THRESH_BINARY)
        
        # Dilate mask to cover glare area
        kernel = np.ones((15, 15), np.uint8)
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
        
        # Reduce brightness in glare regions
        hsv[:, :, 2] = np.where(bright_mask == 255, 
                                hsv[:, :, 2] * 0.7, 
                                hsv[:, :, 2]).astype(np.uint8)
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    def multi_scale_retinex(self, image: np.ndarray) -> np.ndarray:
        """
        Multi-Scale Retinex (MSR) algorithm
        Advanced technique for low-light enhancement
        """
        def single_scale_retinex(img, sigma):
            """Single scale retinex"""
            retinex = np.log10(img + 1.0) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1.0)
            return retinex
        
        # Convert to float
        img_float = image.astype(np.float64) + 1.0
        
        # Apply MSR on each channel
        scales = [15, 80, 250]
        msr = np.zeros_like(img_float)
        
        for scale in scales:
            msr += single_scale_retinex(img_float, scale)
        
        msr = msr / len(scales)
        
        # Normalize
        msr = (msr - np.min(msr)) / (np.max(msr) - np.min(msr)) * 255
        
        return msr.astype(np.uint8)
    
    def enhance(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Main enhancement function with automatic mode selection
        
        Args:
            image: Input image (BGR)
            
        Returns:
            enhanced: Enhanced image
            method: Method used
        """
        if self.mode == 'auto':
            # Detect lighting condition
            condition = self.detect_lighting_condition(image)
            
            if condition == 'bright':
                # No enhancement needed
                return image, 'none'
            
            elif condition == 'normal':
                # Light CLAHE enhancement
                enhanced = self.enhance_clahe(image)
                return enhanced, 'clahe'
            
            elif condition == 'low':
                # CLAHE + gamma correction
                enhanced = self.enhance_clahe(image)
                enhanced = self.enhance_gamma(enhanced, gamma=1.3)
                return enhanced, 'clahe+gamma'
            
            else:  # very_low
                # Multi-scale retinex for extreme low light
                enhanced = self.multi_scale_retinex(image)
                enhanced = self.reduce_noise(enhanced)
                return enhanced, 'msr'
        
        elif self.mode == 'clahe':
            return self.enhance_clahe(image), 'clahe'
        
        elif self.mode == 'gamma':
            return self.enhance_gamma(image), 'gamma'
        
        elif self.mode == 'adaptive':
            return self.enhance_adaptive(image), 'adaptive'
        
        elif self.mode == 'fusion':
            # Fusion of multiple techniques
            clahe_result = self.enhance_clahe(image)
            gamma_result = self.enhance_gamma(image, gamma=1.4)
            
            # Weighted average
            enhanced = cv2.addWeighted(clahe_result, 0.6, gamma_result, 0.4, 0)
            enhanced = self.reduce_noise(enhanced)
            
            return enhanced, 'fusion'
        
        else:
            return image, 'none'


class NightTimeDetector:
    """
    Detect if camera is operating in night-time conditions
    Can switch models or adjust parameters accordingly
    """
    
    def __init__(self):
        self.is_night = False
        self.confidence = 0.0
        
    def detect_night_time(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if image is captured at night
        
        Args:
            image: Input image (BGR)
            
        Returns:
            is_night: True if night-time
            confidence: Confidence score (0-1)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Features for night detection
        avg_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Detect bright spots (headlights, street lights)
        _, bright_regions = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(bright_regions > 0) / (gray.shape[0] * gray.shape[1])
        
        # Night-time characteristics:
        # 1. Low average brightness
        # 2. High contrast (bright headlights vs dark background)
        # 3. Small bright spots
        
        is_night = (avg_brightness < 80 and 
                   std_brightness > 30 and 
                   bright_ratio < 0.1)
        
        # Calculate confidence
        brightness_score = max(0, 1 - (avg_brightness / 100))
        contrast_score = min(1, std_brightness / 60)
        bright_spot_score = 1 if 0.01 < bright_ratio < 0.1 else 0
        
        confidence = (brightness_score + contrast_score + bright_spot_score) / 3
        
        self.is_night = is_night
        self.confidence = confidence
        
        return is_night, confidence


if __name__ == "__main__":
    print("Night-Time Enhancement Module")
    print("=" * 60)
    print("\nFeatures:")
    print("✅ CLAHE enhancement")
    print("✅ Gamma correction")
    print("✅ Multi-Scale Retinex")
    print("✅ Headlight glare reduction")
    print("✅ Adaptive noise reduction")
    print("✅ Automatic mode selection")
    print("\nUsage:")
    print("  enhancer = LowLightEnhancer(mode='auto')")
    print("  enhanced_frame, method = enhancer.enhance(frame)")
