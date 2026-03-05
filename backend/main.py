"""
CleanFrame Editor - Backend API
================================

A FastAPI-based backend for video inpainting that removes unwanted objects
or user-selected regions from videos.

ETHICAL / LEGAL NOTICE:
-----------------------
This tool is designed for LEGITIMATE use cases only:
  - Removing unwanted objects in user-owned or properly licensed footage
  - Protecting privacy (blurring faces, license plates, addresses)
  - Cleaning up personal video projects
  
This tool is NOT intended for:
  - Removing watermarks, platform logos, or DRM protection
  - Violating copyright or intellectual property rights
  - Any unauthorized modification of content you don't own

By using this tool, you agree to use it responsibly and legally.
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip

# EasyOCR for automatic text detection (lazy loaded to speed up startup)
easyocr_reader = None

def get_ocr_reader():
    """Lazy load EasyOCR reader to avoid slow startup."""
    global easyocr_reader
    if easyocr_reader is None:
        import easyocr
        logger.info("Loading EasyOCR model (first time may take a moment)...")
        easyocr_reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA
        logger.info("EasyOCR model loaded successfully")
    return easyocr_reader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CleanFrame Editor API",
    description="Video inpainting API for removing unwanted objects from videos",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint for Render."""
    return {"status": "ok"}

# CORS configuration — allow all origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
FRAMES_DIR = BASE_DIR / "frames"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for directory in [UPLOADS_DIR, FRAMES_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Mount static file directories for serving frames and results
# Using /static/ prefix to avoid conflicting with API endpoints
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/static/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")
app.mount("/static/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Serve frontend static build (dist folder)
FRONTEND_DIR = BASE_DIR.parent / "frontend" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="frontend_assets")

# Allowed video file extensions
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class Region(BaseModel):
    """Represents a rectangular region to be inpainted."""
    x: int
    y: int
    width: int
    height: int


class MaskWithRange(BaseModel):
    """Represents a mask with explicit frame range."""
    region: Region
    start_frame: int
    end_frame: int


class FrameMask(BaseModel):
    """Represents masks for a specific frame (legacy format)."""
    frame_index: int
    regions: List[Region]


class ProcessRequest(BaseModel):
    """Request body for video processing.
    
    Supports new format with explicit frame ranges for each mask.
    """
    video_id: str
    masks: List[MaskWithRange]  # New format with start/end frames



class UploadResponse(BaseModel):
    """Response after successful video upload."""
    video_id: str
    filename: str
    message: str


class FramesResponse(BaseModel):
    """Response containing frame information."""
    video_id: str
    total_frames: int
    frame_urls: List[str]
    fps: float
    duration: float
    width: int
    height: int


class ProcessResponse(BaseModel):
    """Response after video processing."""
    video_id: str
    result_url: str
    message: str


class DetectedText(BaseModel):
    """Represents a detected text region."""
    text: str
    confidence: float
    region: Region
    frame_index: int


class AutoDetectResponse(BaseModel):
    """Response from auto-detect endpoint."""
    video_id: str
    detections: List[DetectedText]
    frames_analyzed: int
    message: str


# ============================================================================
# Core Video Processing Functions
# ============================================================================

def extract_frames(video_path: Path, output_folder: Path, max_frames: int = 900) -> dict:
    """
    Extract frames from a video file and save them as JPEG images.
    
    This function uses OpenCV to read the video frame by frame and saves
    each frame as a numbered JPEG file in the output folder.
    
    Args:
        video_path: Path to the input video file
        output_folder: Directory where frames will be saved
        max_frames: Maximum number of frames to extract (default 900 for ~30 seconds at 30fps)
    
    Returns:
        dict containing:
            - total_frames: Number of frames extracted
            - fps: Original video frame rate
            - duration: Video duration in seconds
            - width: Frame width in pixels
            - height: Frame height in pixels
    
    How it works:
        1. Opens video file with OpenCV VideoCapture
        2. Reads metadata (fps, dimensions)
        3. Iterates through frames, saving each as frame_XXXX.jpg
        4. Limits extraction to max_frames to prevent memory issues
    """
    logger.info(f"Extracting frames from {video_path} to {output_folder}")
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_video_frames / fps if fps > 0 else 0
    
    logger.info(f"Video info: {total_video_frames} frames, {fps} fps, {width}x{height}, {duration:.2f}s")
    
    # Limit frames to extract
    frames_to_extract = min(total_video_frames, max_frames)
    
    frame_count = 0
    while frame_count < frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame as JPEG (numbered with zero-padding)
        frame_filename = output_folder / f"frame_{frame_count:04d}.jpg"
        cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_count += 1
        
        # Log progress every 100 frames
        if frame_count % 100 == 0:
            logger.info(f"Extracted {frame_count}/{frames_to_extract} frames")
    
    cap.release()
    logger.info(f"Frame extraction complete: {frame_count} frames saved")
    
    return {
        "total_frames": frame_count,
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height
    }


def create_mask_image(frame_shape: tuple, regions: List[Region]) -> np.ndarray:
    """
    Create a binary mask image from a list of rectangular regions.
    
    The mask is used by OpenCV's inpaint function to know which areas
    to fill in. White (255) pixels indicate areas to inpaint, black (0)
    pixels are preserved.
    
    Args:
        frame_shape: Tuple of (height, width) for the mask
        regions: List of Region objects defining rectangles to mask
    
    Returns:
        numpy array of shape (height, width) with dtype uint8
        - 255 (white) where inpainting should occur
        - 0 (black) everywhere else
    """
    # Create black mask (all zeros)
    mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
    
    # Draw white rectangles for each region
    for region in regions:
        x1 = max(0, region.x)
        y1 = max(0, region.y)
        x2 = min(frame_shape[1], region.x + region.width)
        y2 = min(frame_shape[0], region.y + region.height)
        
        # Fill rectangle with white (255)
        mask[y1:y2, x1:x2] = 255
    
    return mask


def inpaint_frame(frame_path: Path, mask_regions: List[Region], output_path: Path) -> bool:
    """
    Apply inpainting to a single frame to remove masked regions.
    
    Uses OpenCV's inpainting algorithm (Telea method) to fill in the
    masked regions with surrounding pixel information, creating a
    seamless removal of unwanted objects.
    
    Args:
        frame_path: Path to the input frame image
        mask_regions: List of Region objects to inpaint
        output_path: Path where the inpainted frame will be saved
    
    Returns:
        True if inpainting was successful, False otherwise
    """
    try:
        # Read the frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return False
        
        # Create the binary mask
        mask = create_mask_image(frame.shape[:2], mask_regions)
        
        # Apply inpainting using Telea algorithm (faster than Navier-Stokes)
        # Radius 2 is faster while still maintaining good quality
        inpainted = cv2.inpaint(frame, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        
        # Save the inpainted frame (quality 90 for faster write)
        cv2.imwrite(str(output_path), inpainted, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        return True
    except Exception as e:
        return False


def process_single_frame(frame_info: tuple) -> bool:
    """
    Process a single frame - used for parallel processing.
    
    Args:
        frame_info: Tuple of (frame_path, output_path, regions_list or None)
    
    Returns:
        True if successful
    """
    frame_path, output_path, regions = frame_info
    
    if regions:
        success = inpaint_frame(frame_path, regions, output_path)
        if not success:
            shutil.copy(frame_path, output_path)
    else:
        shutil.copy(frame_path, output_path)
    
    return True


def reassemble_video(frames_folder: Path, original_video_path: Path, output_path: Path) -> bool:
    """
    Reassemble processed frames back into a video with original audio.
    
    This function takes a folder of processed frames and combines them
    into a video file, preserving the original frame rate and adding
    back the audio track from the original video.
    
    Args:
        frames_folder: Directory containing processed frame images
        original_video_path: Path to original video (for audio extraction)
        output_path: Path where the final video will be saved
    
    Returns:
        True if reassembly was successful, False otherwise
    
    How it works:
        1. Get list of frame files sorted by frame number
        2. Load original video to get fps and audio track
        3. Create ImageSequenceClip from the frames at original fps
        4. Add audio from original video
        5. Write final video with H.264 codec for broad compatibility
    """
    try:
        logger.info(f"Reassembling video from {frames_folder}")
        
        # Get sorted list of frame files
        frame_files = sorted(frames_folder.glob("frame_*.jpg"))
        if not frame_files:
            logger.error("No frames found for reassembly")
            return False
        
        frame_paths = [str(f) for f in frame_files]
        logger.info(f"Found {len(frame_paths)} frames to reassemble")
        
        # Load original video to get properties
        original_clip = VideoFileClip(str(original_video_path))
        fps = original_clip.fps
        audio = original_clip.audio
        
        # Create video from image sequence
        # fps parameter ensures we use the same frame rate as original
        video_clip = ImageSequenceClip(frame_paths, fps=fps)
        
        # Add audio from original video if it exists
        if audio is not None:
            # Make sure audio duration matches video duration
            video_clip = video_clip.set_audio(audio.set_duration(video_clip.duration))
        
        # Write the final video
        # Using libx264 codec for wide compatibility
        # audio_codec='aac' for audio compatibility
        video_clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            logger=None  # Suppress moviepy's verbose logging
        )
        
        # Clean up
        video_clip.close()
        original_clip.close()
        
        logger.info(f"Video reassembly complete: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error reassembling video: {e}")
        return False


def propagate_masks_across_frames(
    masks: List[FrameMask], 
    total_frames: int, 
    track_range: int = 5
) -> List[FrameMask]:
    """
    Propagate masks to neighboring frames for object tracking.
    
    This is a simple implementation that copies the same mask regions
    to nearby frames. More sophisticated approaches would use optical
    flow or object tracking.
    
    Args:
        masks: Original list of frame masks from user
        total_frames: Total number of frames in the video
        track_range: Number of frames before and after to propagate to
    
    Returns:
        Expanded list of FrameMask objects including propagated masks
    
    How it works:
        1. For each original mask, copy it to neighboring frames
        2. Track_range determines how many frames before/after
        3. Merge masks if multiple exist for the same frame
    """
    # Create a dictionary to collect masks per frame
    frame_masks_dict = {}
    
    for mask in masks:
        # Original frame
        if mask.frame_index not in frame_masks_dict:
            frame_masks_dict[mask.frame_index] = []
        frame_masks_dict[mask.frame_index].extend(mask.regions)
        
        # Propagate to neighboring frames
        for offset in range(-track_range, track_range + 1):
            if offset == 0:
                continue  # Skip original frame
            
            target_frame = mask.frame_index + offset
            if 0 <= target_frame < total_frames:
                if target_frame not in frame_masks_dict:
                    frame_masks_dict[target_frame] = []
                frame_masks_dict[target_frame].extend(mask.regions)
    
    # Convert back to list of FrameMask
    propagated_masks = [
        FrameMask(frame_index=idx, regions=regions)
        for idx, regions in sorted(frame_masks_dict.items())
    ]
    
    return propagated_masks


def detect_emoji_graphics(frame: np.ndarray, text_regions: list, search_margin: int = 40) -> list:
    """
    Detect emoji-like graphics and icons DIRECTLY adjacent to text watermarks.
    
    This function is focused on detecting colorful icons/emojis that are part of
    watermarks (like cloud/ghost emojis), NOT random objects in the video.
    
    Detection strategy focuses on:
    1. High-saturation colorful elements (emojis are usually colorful)
    2. Elements that are distinctly different from the background
    
    Args:
        frame: The video frame (BGR format)
        text_regions: List of detected text regions (dicts with x, y, width, height)
        search_margin: Pixels to search around each text region (default 40 for tight focus)
    
    Returns:
        List of additional regions containing detected graphics
    """
    graphic_regions = []
    height, width = frame.shape[:2]
    
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # For each text region, search for nearby icons
    for text_reg in text_regions:
        # Define search area around the text
        search_x = max(0, text_reg['x'] - search_margin)
        search_y = max(0, text_reg['y'] - search_margin)
        search_x2 = min(width, text_reg['x'] + text_reg['width'] + search_margin)
        search_y2 = min(height, text_reg['y'] + text_reg['height'] + search_margin)
        
        # Extract the search region
        roi = frame[search_y:search_y2, search_x:search_x2]
        roi_hsv = hsv[search_y:search_y2, search_x:search_x2]
        roi_gray = gray[search_y:search_y2, search_x:search_x2]
        
        if roi.size == 0:
            continue
        
        # Strategy 1: Detect colorful regions (high saturation)
        saturation = roi_hsv[:, :, 1]
        high_sat_mask = (saturation > 100).astype(np.uint8) * 255
        
        # Strategy 2: Detect WHITE/BRIGHT icons (high value, low saturation)
        # Cloud icons are white, so they have low saturation but high brightness
        value = roi_hsv[:, :, 2]
        bright_white_mask = ((value > 200) & (saturation < 50)).astype(np.uint8) * 255
        
        # Strategy 3: Detect elements distinctly different from background
        bg_color = np.median(roi, axis=(0, 1))
        color_diff = np.sqrt(np.sum((roi.astype(float) - bg_color) ** 2, axis=2))
        distinct_mask = (color_diff > 50).astype(np.uint8) * 255
        
        # Combine all detection strategies
        combined_mask = cv2.bitwise_or(high_sat_mask, bright_white_mask)
        combined_mask = cv2.bitwise_or(combined_mask, distinct_mask)
        
        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        # Find contours in the mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Icons/emojis are typically 80-8000 pixels
            if 80 < area < 8000:
                x, y, w, h = cv2.boundingRect(contour)
                
                abs_x = search_x + x
                abs_y = search_y + y
                
                # Check aspect ratio - icons are roughly square (0.4-2.5)
                aspect_ratio = w / h if h > 0 else 0
                if 0.4 < aspect_ratio < 2.5:
                    # Check if this overlaps with the text region itself (skip if so)
                    text_center_x = text_reg['x'] + text_reg['width'] // 2
                    text_center_y = text_reg['y'] + text_reg['height'] // 2
                    
                    region_center_x = abs_x + w // 2
                    region_center_y = abs_y + h // 2
                    
                    # Must be close to text but not overlapping center
                    distance = np.sqrt((text_center_x - region_center_x)**2 + (text_center_y - region_center_y)**2)
                    if 10 < distance < search_margin + 40:  # Between 10 and 100 pixels from text
                        # Add padding around icon
                        padding = 10
                        graphic_regions.append({
                            'x': max(0, abs_x - padding),
                            'y': max(0, abs_y - padding),
                            'width': w + padding * 2,
                            'height': h + padding * 2,
                            'type': 'emoji_icon'
                        })
    
    return graphic_regions


def detect_floating_overlays(frame: np.ndarray, edge_margin: int = 100) -> list:
    """
    Detect floating overlay elements like watermarks at corners or edges.
    
    Watermarks with emojis/logos often appear at consistent positions
    (corners, edges) and have distinct visual properties.
    
    Args:
        frame: The video frame (BGR format)
        edge_margin: Pixels from edges to analyze
    
    Returns:
        List of detected overlay regions
    """
    overlay_regions = []
    height, width = frame.shape[:2]
    
    # Define corner regions to analyze
    corners = [
        (0, 0, edge_margin * 2, edge_margin),  # Top-left
        (width - edge_margin * 2, 0, edge_margin * 2, edge_margin),  # Top-right
        (0, height - edge_margin, edge_margin * 2, edge_margin),  # Bottom-left
        (width - edge_margin * 2, height - edge_margin, edge_margin * 2, edge_margin),  # Bottom-right
    ]
    
    for cx, cy, cw, ch in corners:
        cx = max(0, cx)
        cy = max(0, cy)
        cw = min(cw, width - cx)
        ch = min(ch, height - cy)
        
        if cw <= 0 or ch <= 0:
            continue
            
        roi = frame[cy:cy+ch, cx:cx+cw]
        
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # If there are significant edges, there might be overlay content
        edge_density = np.sum(edges > 0) / (cw * ch)
        
        if edge_density > 0.05:  # More than 5% edge pixels
            # Find contours of potential overlay elements
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get bounding box of all contours combined
                all_points = np.vstack([c for c in contours if len(c) > 0])
                x, y, w, h = cv2.boundingRect(all_points)
                
                if w > 20 and h > 10:  # Minimum size threshold
                    padding = 10
                    overlay_regions.append({
                        'x': max(0, cx + x - padding),
                        'y': max(0, cy + y - padding),
                        'width': w + padding * 2,
                        'height': h + padding * 2,
                        'type': 'overlay'
                    })
    
    return overlay_regions


def detect_bubble_icons(frame: np.ndarray, min_size: int = 20, max_size: int = 150) -> list:
    """
    Detect speech bubble icons and similar rounded overlay shapes anywhere in the frame.
    
    This function specifically targets:
    - Speech bubble / comment icons
    - Rounded rectangular overlays
    - Semi-transparent UI elements
    
    Uses template matching concepts and shape analysis to find these elements.
    
    Args:
        frame: The video frame (BGR format)
        min_size: Minimum size of icon to detect
        max_size: Maximum size of icon to detect
    
    Returns:
        List of detected bubble icon regions
    """
    bubble_regions = []
    height, width = frame.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use multiple thresholding approaches to catch various overlay types
    
    # Approach 1: Detect light-colored overlays (like white/light blue speech bubbles)
    # Look for pixels that are brighter than local average
    local_mean = cv2.blur(gray, (51, 51))
    bright_diff = gray.astype(float) - local_mean.astype(float)
    light_overlay_mask = (bright_diff > 15).astype(np.uint8) * 255
    
    # Approach 2: Edge-based detection for outlined shapes
    edges = cv2.Canny(blurred, 20, 80)
    
    # Dilate edges to connect nearby pixels
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Approach 3: Adaptive thresholding for semi-transparent overlays
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 8
    )
    adaptive_inv = cv2.bitwise_not(adaptive)
    
    # Combine detection masks
    combined = cv2.bitwise_or(light_overlay_mask, edges_dilated)
    combined = cv2.bitwise_or(combined, adaptive_inv)
    
    # Clean up the mask
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area (size of speech bubble)
        min_area = min_size * min_size
        max_area = max_size * max_size
        
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio - speech bubbles are usually roughly square or slightly wide
            aspect_ratio = w / h if h > 0 else 0
            if 0.5 < aspect_ratio < 2.5:
                # Check for circular/rounded shape using convex hull
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    # Solidity: ratio of contour area to hull area
                    # Higher solidity = more filled/rounded shape
                    solidity = area / hull_area
                    
                    # Speech bubbles and icons tend to have high solidity (> 0.5)
                    if solidity > 0.4:
                        # Check if this looks like a UI overlay (not just noise)
                        # Extract the region and check for uniformity
                        roi = gray[y:y+h, x:x+w]
                        if roi.size > 0:
                            std_dev = np.std(roi)
                            mean_val = np.mean(roi)
                            
                            # Overlays tend to have relatively uniform color (low std) 
                            # OR be distinctly different from surroundings
                            # Also check if it's in an overlay-typical position (corners, edges)
                            is_near_edge = (x < 200 or x + w > width - 200 or 
                                          y < 150 or y + h > height - 150)
                            
                            if std_dev < 60 or is_near_edge:
                                padding = 10
                                bubble_regions.append({
                                    'x': max(0, x - padding),
                                    'y': max(0, y - padding),
                                    'width': w + padding * 2,
                                    'height': h + padding * 2,
                                    'type': 'bubble_icon'
                                })
    
    return bubble_regions


def group_similar_regions(detections: list, threshold: int = 40) -> list:
    """
    Group similar text detections from different frames.
    
    This identifies persistent text (like watermarks) that appears in the same
    position across multiple frames.
    
    Args:
        detections: List of detection dicts with x, y, width, height, frame_index
        threshold: Pixel tolerance for considering regions as "same position"
    
    Returns:
        List of grouped regions with bounding boxes covering all instances
    """
    groups = []
    
    for detection in detections:
        # Try to find an existing group that matches this detection
        found_group = None
        for group in groups:
            # Check if this detection is close enough to the group's position
            x_diff = abs(group['x'] - detection['x'])
            y_diff = abs(group['y'] - detection['y'])
            w_diff = abs(group['width'] - detection['width'])
            h_diff = abs(group['height'] - detection['height'])
            
            if x_diff < threshold and y_diff < threshold and w_diff < threshold * 2 and h_diff < threshold * 2:
                found_group = group
                break
        
        if found_group:
            # Expand the group's bounding box to include this detection
            new_x = min(found_group['x'], detection['x'])
            new_y = min(found_group['y'], detection['y'])
            new_x2 = max(found_group['x'] + found_group['width'], detection['x'] + detection['width'])
            new_y2 = max(found_group['y'] + found_group['height'], detection['y'] + detection['height'])
            
            found_group['x'] = new_x
            found_group['y'] = new_y
            found_group['width'] = new_x2 - new_x
            found_group['height'] = new_y2 - new_y
            found_group['count'] += 1
            found_group['min_frame'] = min(found_group['min_frame'], detection['frame_index'])
            found_group['max_frame'] = max(found_group['max_frame'], detection['frame_index'])
        else:
            # Create a new group
            groups.append({
                'x': detection['x'],
                'y': detection['y'],
                'width': detection['width'],
                'height': detection['height'],
                'count': 1,
                'min_frame': detection['frame_index'],
                'max_frame': detection['frame_index']
            })
    
    return groups


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve the React frontend at root."""
    frontend_index = FRONTEND_DIR / "index.html"
    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    return {"message": "CleanFrame Editor API is running", "version": "1.0.0"}


@app.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for processing.
    
    Accepts video files (mp4, mov, avi, mkv, webm) and saves them
    to the uploads directory with a unique ID.
    
    Args:
        file: The uploaded video file
    
    Returns:
        UploadResponse with video_id and confirmation message
    
    Raises:
        HTTPException 400: If file type is not allowed
        HTTPException 500: If file saving fails
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed types: {ALLOWED_EXTENSIONS}"
        )
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())[:8]
    
    # Save file with video_id as prefix
    safe_filename = f"{video_id}{file_ext}"
    file_path = UPLOADS_DIR / safe_filename
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Video uploaded: {safe_filename}")
        
        return UploadResponse(
            video_id=video_id,
            filename=safe_filename,
            message="Video uploaded successfully"
        )
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


@app.get("/frames/{video_id}", response_model=FramesResponse)
async def get_frames(video_id: str):
    """
    Extract frames from an uploaded video and return their URLs.
    
    If frames have already been extracted, returns the existing frames.
    Otherwise, extracts frames from the video and saves them.
    
    Args:
        video_id: The unique identifier for the video
    
    Returns:
        FramesResponse with frame URLs and video metadata
    
    Raises:
        HTTPException 404: If video file not found
        HTTPException 500: If frame extraction fails
    """
    # Find the video file
    video_files = list(UPLOADS_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    
    video_path = video_files[0]
    frames_folder = FRAMES_DIR / video_id
    
    # Check if frames already extracted
    if frames_folder.exists() and list(frames_folder.glob("frame_*.jpg")):
        logger.info(f"Using existing frames for {video_id}")
    else:
        # Extract frames
        try:
            extract_frames(video_path, frames_folder)
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise HTTPException(status_code=500, detail=f"Error extracting frames: {str(e)}")
    
    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    # Get list of frame files
    frame_files = sorted(frames_folder.glob("frame_*.jpg"))
    frame_urls = [f"/static/frames/{video_id}/{f.name}" for f in frame_files]
    
    return FramesResponse(
        video_id=video_id,
        total_frames=len(frame_files),
        frame_urls=frame_urls,
        fps=fps,
        duration=duration,
        width=width,
        height=height
    )


@app.post("/process", response_model=ProcessResponse)
async def process_video(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process video by applying inpainting to masked regions.
    
    Takes the video ID and mask definitions, applies inpainting to
    each specified frame, then reassembles into a new video.
    
    Args:
        request: ProcessRequest containing video_id and masks
    
    Returns:
        ProcessResponse with URL to download the processed video
    
    Raises:
        HTTPException 404: If video or frames not found
        HTTPException 400: If no masks provided
        HTTPException 500: If processing fails
    """
    video_id = request.video_id
    
    # Validate video exists
    video_files = list(UPLOADS_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    
    video_path = video_files[0]
    frames_folder = FRAMES_DIR / video_id
    
    if not frames_folder.exists():
        raise HTTPException(status_code=404, detail="Frames not extracted. Call /frames/{video_id} first")
    
    if not request.masks:
        raise HTTPException(status_code=400, detail="No masks provided")
    
    # Create output folder for processed frames
    processed_folder = FRAMES_DIR / f"{video_id}_processed"
    processed_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get total frames
        frame_files = sorted(frames_folder.glob("frame_*.jpg"))
        total_frames = len(frame_files)
        
        # Build a lookup dictionary: frame_index -> list of regions to inpaint
        # Each mask has a start_frame, end_frame, and region
        mask_lookup = {}
        for mask in request.masks:
            start = max(0, min(mask.start_frame, mask.end_frame))
            end = min(total_frames - 1, max(mask.start_frame, mask.end_frame))
            
            for frame_idx in range(start, end + 1):
                if frame_idx not in mask_lookup:
                    mask_lookup[frame_idx] = []
                mask_lookup[frame_idx].append(mask.region)
        
        logger.info(f"Processing {total_frames} frames with {len(mask_lookup)} frames needing inpainting")
        logger.info(f"Total mask regions: {sum(len(v) for v in mask_lookup.values())}")
        
        # Process each frame
        for i, frame_file in enumerate(frame_files):
            output_path = processed_folder / frame_file.name
            
            if i in mask_lookup:
                # Apply inpainting to this frame
                success = inpaint_frame(frame_file, mask_lookup[i], output_path)
                if not success:
                    # If inpainting fails, copy original
                    shutil.copy(frame_file, output_path)
            else:
                # No mask for this frame, copy as-is
                shutil.copy(frame_file, output_path)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{total_frames} frames")
        
        # Reassemble video
        result_filename = f"{video_id}_processed.mp4"
        result_path = RESULTS_DIR / result_filename
        
        success = reassemble_video(processed_folder, video_path, result_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reassemble video")
        
        # Clean up processed frames (optional, keep for debugging)
        # shutil.rmtree(processed_folder)
        
        return ProcessResponse(
            video_id=video_id,
            result_url=f"/static/results/{result_filename}",
            message="Video processing complete"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.get("/download/{video_id}")
async def download_video(video_id: str):
    """
    Download the processed video file.
    
    Args:
        video_id: The unique identifier for the processed video
    
    Returns:
        FileResponse for downloading the video
    
    Raises:
        HTTPException 404: If processed video not found
    """
    result_path = RESULTS_DIR / f"{video_id}_processed.mp4"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found")
    
    return FileResponse(
        path=result_path,
        media_type="video/mp4",
        filename=f"cleanframe_{video_id}.mp4"
    )


@app.delete("/cleanup/{video_id}")
async def cleanup_video(video_id: str):
    """
    Clean up all files associated with a video ID.
    
    Removes uploaded video, extracted frames, and processed results.
    
    Args:
        video_id: The unique identifier for the video
    
    Returns:
        Confirmation message
    """
    cleaned = []
    
    # Remove uploaded video
    for video_file in UPLOADS_DIR.glob(f"{video_id}.*"):
        video_file.unlink()
        cleaned.append(f"upload: {video_file.name}")
    
    # Remove frames folder
    frames_folder = FRAMES_DIR / video_id
    if frames_folder.exists():
        shutil.rmtree(frames_folder)
        cleaned.append(f"frames: {video_id}/")
    
    # Remove processed frames folder
    processed_folder = FRAMES_DIR / f"{video_id}_processed"
    if processed_folder.exists():
        shutil.rmtree(processed_folder)
        cleaned.append(f"processed frames: {video_id}_processed/")
    
    # Remove result video
    result_path = RESULTS_DIR / f"{video_id}_processed.mp4"
    if result_path.exists():
        result_path.unlink()
        cleaned.append(f"result: {video_id}_processed.mp4")
    
    return {"message": "Cleanup complete", "cleaned": cleaned}


@app.post("/auto-detect/{video_id}", response_model=AutoDetectResponse)
async def auto_detect_text(video_id: str, sample_interval: int = 30):
    """
    Automatically detect text regions in video frames using OCR.
    
    This endpoint analyzes video frames and returns detected text regions
    that can be used as masks for inpainting.
    
    Args:
        video_id: The unique identifier for the video
        sample_interval: Analyze every Nth frame (default 30 = ~1 per second at 30fps)
    
    Returns:
        AutoDetectResponse with list of detected text regions
    
    How it works:
        1. Sample frames at regular intervals
        2. Run EasyOCR on each sampled frame
        3. Return bounding boxes for detected text
        4. User can review and adjust before processing
    """
    # Validate video exists
    frames_folder = FRAMES_DIR / video_id
    if not frames_folder.exists():
        raise HTTPException(status_code=404, detail="Frames not extracted. Call /frames/{video_id} first")
    
    frame_files = sorted(frames_folder.glob("frame_*.jpg"))
    if not frame_files:
        raise HTTPException(status_code=404, detail="No frames found")
    
    total_frames = len(frame_files)
    
    try:
        # Get OCR reader (lazy loaded)
        reader = get_ocr_reader()
        
        detections = []
        frames_analyzed = 0
        
        # Sample frames at interval
        for i in range(0, total_frames, sample_interval):
            frame_file = frame_files[i]
            frames_analyzed += 1
            
            # Read frame
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
            
            # Run OCR detection
            results = reader.readtext(frame)
            
            for (bbox, text, confidence) in results:
                # Skip low confidence detections
                if confidence < 0.3:
                    continue
                
                # Convert bbox to rectangle (bbox is 4 points)
                # Format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x = int(min(x_coords))
                y = int(min(y_coords))
                width = int(max(x_coords) - min(x_coords))
                height = int(max(y_coords) - min(y_coords))
                
                # Add some padding around the text
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = width + (padding * 2)
                height = height + (padding * 2)
                
                detections.append(DetectedText(
                    text=text,
                    confidence=round(confidence, 3),
                    region=Region(x=x, y=y, width=width, height=height),
                    frame_index=i
                ))
            
            logger.info(f"Analyzed frame {i}/{total_frames}, found {len(results)} text regions")
        
        # Group similar detections (same position = same text appearing across frames)
        # This helps identify persistent text like watermarks
        logger.info(f"Auto-detect complete: {len(detections)} text regions found in {frames_analyzed} frames")
        
        return AutoDetectResponse(
            video_id=video_id,
            detections=detections,
            frames_analyzed=frames_analyzed,
            message=f"Found {len(detections)} text regions across {frames_analyzed} sampled frames"
        )
        
    except Exception as e:
        logger.error(f"Error in auto-detect: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-detection failed: {str(e)}")


@app.post("/auto-process/{video_id}")
async def auto_process_video(video_id: str, min_confidence: float = 0.4, sample_interval: int = 30):
    """
    Automatically detect and remove all text/watermarks from a video in one step.
    
    OPTIMIZED FOR SPEED:
    - Uses parallel frame processing with ThreadPoolExecutor
    - Increased default sample_interval to 30 for faster detection
    - Reduced inpainting radius and JPEG quality for faster processing
    
    Args:
        video_id: The unique identifier for the video
        min_confidence: Minimum OCR confidence to include (0.0-1.0)
        sample_interval: Analyze every Nth frame for detection (default 30 = faster)
    
    Returns:
        ProcessResponse with URL to download the processed video
    """
    # First, ensure frames are extracted
    frames_folder = FRAMES_DIR / video_id
    if not frames_folder.exists():
        # Try to extract frames first
        video_files = list(UPLOADS_DIR.glob(f"{video_id}.*"))
        if not video_files:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        extract_frames(video_files[0], frames_folder)
    
    frame_files = sorted(frames_folder.glob("frame_*.jpg"))
    total_frames = len(frame_files)
    
    if total_frames == 0:
        raise HTTPException(status_code=404, detail="No frames found")
    
    try:
        # Step 1: Detect all text across sampled frames
        logger.info(f"Auto-process: Starting smart watermark detection for {video_id}")
        reader = get_ocr_reader()
        
        # Collect all detected regions with their frame indices
        raw_detections = []
        frames_analyzed = 0
        
        for i in range(0, total_frames, sample_interval):
            frame = cv2.imread(str(frame_files[i]))
            if frame is None:
                continue
            
            frames_analyzed += 1
            results = reader.readtext(frame)
            
            for (bbox, text, confidence) in results:
                if confidence < min_confidence:
                    continue
                
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                # Add GENEROUS padding around text to ensure full watermark is covered
                padding = 20
                x = int(min(x_coords)) - padding
                y = int(min(y_coords)) - padding
                width = int(max(x_coords) - min(x_coords)) + (padding * 2)
                height = int(max(y_coords) - min(y_coords)) + (padding * 2)
                
                # Clamp to frame bounds
                x = max(0, x)
                y = max(0, y)
                
                raw_detections.append({
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'frame_index': i,
                    'text': text,
                    'confidence': confidence
                })
        
        if not raw_detections:
            # No text detected - return without processing
            # We only remove TEXT-based watermarks and icons directly next to them
            logger.info("Auto-process: No text watermarks detected in video.")
            return ProcessResponse(
                video_id=video_id,
                result_url="",
                message="No text watermarks detected in video. No processing needed."
            )
        
        # Step 1.5: Detect emojis/icons ONLY directly adjacent to detected text
        # This catches icons like cloud/ghost emojis that are part of the watermark
        logger.info(f"Auto-process: Found {len(raw_detections)} text detections, scanning for adjacent icons/emojis...")
        
        graphics_detected = 0
        for i in range(0, total_frames, sample_interval):
            frame = cv2.imread(str(frame_files[i]))
            if frame is None:
                continue
            
            # Get text regions from this frame
            frame_text_regions = [d for d in raw_detections if d['frame_index'] == i]
            
            if frame_text_regions:
                # Detect emoji-like graphics adjacent to text (wider margin for full coverage)
                # search_margin=60 ensures we catch icons like cloud emojis near the watermark text
                graphic_regions = detect_emoji_graphics(frame, frame_text_regions, search_margin=60)
                
                for graphic in graphic_regions:
                    graphic['frame_index'] = i
                    graphic['text'] = '[icon/emoji]'
                    graphic['confidence'] = 0.8
                    raw_detections.append(graphic)
                    graphics_detected += 1
        
        logger.info(f"Auto-process: Detected {graphics_detected} adjacent icons/emojis")
        
        logger.info(f"Auto-process: Total {len(raw_detections)} detections in {frames_analyzed} frames")
        
        # Step 2: Group similar regions (persistent watermarks appear in similar positions)
        grouped_regions = group_similar_regions(raw_detections, threshold=40)
        logger.info(f"Auto-process: Grouped into {len(grouped_regions)} unique regions")
        
        # Step 3: Identify persistent watermarks (appear in many frames) and apply to full video
        persistent_regions = []
        temporary_regions = []
        
        for group in grouped_regions:
            if group['count'] >= 2 or frames_analyzed <= 3:
                # This is likely a watermark - apply to all frames
                persistent_regions.append(Region(
                    x=group['x'],
                    y=group['y'],
                    width=group['width'],
                    height=group['height']
                ))
            else:
                # This appeared only once - apply to nearby frames only
                temporary_regions.append({
                    'region': Region(
                        x=group['x'],
                        y=group['y'],
                        width=group['width'],
                        height=group['height']
                    ),
                    'frame_index': group['min_frame']
                })
        
        logger.info(f"Auto-process: {len(persistent_regions)} persistent watermarks, {len(temporary_regions)} temporary text")
        
        # Step 4: Create mask lookup for all frames
        mask_lookup = {}
        
        # Apply persistent regions (watermarks) to ALL frames
        for frame_idx in range(total_frames):
            mask_lookup[frame_idx] = list(persistent_regions)
        
        # Apply temporary regions to nearby frames only
        for temp in temporary_regions:
            region = temp['region']
            center_frame = temp['frame_index']
            spread = sample_interval // 2
            start_frame = max(0, center_frame - spread)
            end_frame = min(total_frames - 1, center_frame + spread)
            
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx not in mask_lookup:
                    mask_lookup[frame_idx] = []
                mask_lookup[frame_idx].append(region)
        
        # Count frames that actually have masks
        frames_with_masks = sum(1 for regions in mask_lookup.values() if regions)
        logger.info(f"Auto-process: Will apply inpainting to {frames_with_masks} frames")
        
        # Step 5: Process all frames IN PARALLEL for speed
        video_files = list(UPLOADS_DIR.glob(f"{video_id}.*"))
        video_path = video_files[0]
        
        processed_folder = FRAMES_DIR / f"{video_id}_processed"
        processed_folder.mkdir(parents=True, exist_ok=True)
        
        # Prepare frame processing tasks
        frame_tasks = []
        for i, frame_file in enumerate(frame_files):
            output_path = processed_folder / frame_file.name
            regions_for_frame = mask_lookup.get(i, [])
            frame_tasks.append((frame_file, output_path, regions_for_frame if regions_for_frame else None))
        
        # Process frames in parallel using ThreadPoolExecutor
        # Use 4 workers for good balance of speed and memory usage
        num_workers = min(4, os.cpu_count() or 2)
        logger.info(f"Auto-process: Processing {total_frames} frames with {num_workers} parallel workers...")
        
        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_frame, task): task for task in frame_tasks}
            for future in as_completed(futures):
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"Auto-process: Processed {completed}/{total_frames} frames")
        
        logger.info(f"Auto-process: Frame processing complete")
        
        # Step 6: Reassemble video
        result_filename = f"{video_id}_processed.mp4"
        result_path = RESULTS_DIR / result_filename
        
        success = reassemble_video(processed_folder, video_path, result_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reassemble video")
        
        total_removed = len(persistent_regions) + len(temporary_regions)
        logger.info(f"Auto-process complete for {video_id}: removed {total_removed} regions")
        
        return ProcessResponse(
            video_id=video_id,
            result_url=f"/static/results/{result_filename}",
            message=f"Automatically removed {len(persistent_regions)} watermarks and {len(temporary_regions)} text regions"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in auto-process: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-processing failed: {str(e)}")


# Catch-all route: serve React frontend for any non-API path
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve the React frontend for any path not matched by API routes."""
    frontend_index = FRONTEND_DIR / "index.html"
    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    return {"message": "CleanFrame Editor API is running. Frontend not built yet."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
