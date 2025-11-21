# 🎬 Video/Frames Conversion Tools

Two scripts for converting between video files and image frame sequences.

---

## 📹 frames_to_video.py - Create Video from Frames

Convert a folder of images into a video file.

### Basic Usage

```bash
# Simple conversion
python frames_to_video.py --input frames_folder/ --output output.mp4
```

### Advanced Options

```bash
# Custom FPS (30 fps)
python frames_to_video.py --input frames/ --output video.mp4 --fps 30

# High FPS (60 fps)
python frames_to_video.py --input frames/ --output video.mp4 --fps 60

# Low FPS for slow motion effect (15 fps)
python frames_to_video.py --input frames/ --output video.mp4 --fps 15

# Resize video
python frames_to_video.py --input frames/ --output video.mp4 --width 1920 --height 1080

# Use different codec
python frames_to_video.py --input frames/ --output video.mp4 --codec XVID

# High quality
python frames_to_video.py --input frames/ --output video.mp4 --quality 100

# Specific pattern
python frames_to_video.py --input frames/ --output video.mp4 --pattern "frame_*.jpg"
```

### Supported Codecs
- `mp4v` - MPEG-4 (default, good compatibility)
- `XVID` - Xvid (good quality, larger files)
- `H264` - H.264 (best quality, requires codec)
- `MJPG` - Motion JPEG (large files, fast encoding)

### All Options

```bash
python frames_to_video.py \
    --input frames_folder/ \
    --output video.mp4 \
    --fps 30 \
    --width 1920 \
    --height 1080 \
    --codec mp4v \
    --quality 95 \
    --pattern "*.jpg"
```

---

## 📸 video_to_frames.py - Extract Frames from Video

Extract image frames from a video file.

### Basic Usage

```bash
# Extract all frames
python video_to_frames.py --input video.mp4 --output frames/
```

### Advanced Options

```bash
# Extract every 10th frame (for faster processing)
python video_to_frames.py --input video.mp4 --output frames/ --skip 10

# Extract every 30th frame (1 frame per second at 30fps)
python video_to_frames.py --input video.mp4 --output frames/ --skip 30

# Extract specific time range (5 to 30 seconds)
python video_to_frames.py --input video.mp4 --output frames/ --start 5 --end 30

# Extract first 10 seconds
python video_to_frames.py --input video.mp4 --output frames/ --end 10

# Save as PNG
python video_to_frames.py --input video.mp4 --output frames/ --format png

# Custom filename prefix
python video_to_frames.py --input video.mp4 --output frames/ --prefix "img"

# High quality JPEG
python video_to_frames.py --input video.mp4 --output frames/ --quality 100
```

### Supported Formats
- `jpg` - JPEG (default, smaller files)
- `png` - PNG (lossless, larger files)
- `bmp` - Bitmap (uncompressed, very large)

### All Options

```bash
python video_to_frames.py \
    --input video.mp4 \
    --output frames/ \
    --skip 1 \
    --start 0 \
    --end 60 \
    --prefix "frame" \
    --format jpg \
    --quality 95
```

---

## 💡 Common Use Cases

### 1. Create Timelapse from Photos

```bash
# Combine photos into 30fps video
python frames_to_video.py --input photos/ --output timelapse.mp4 --fps 30
```

### 2. Extract Frames for Dataset

```bash
# Extract every 10th frame for training data
python video_to_frames.py --input video.mp4 --output dataset/images/ --skip 10
```

### 3. Convert Video Format

```bash
# Extract frames
python video_to_frames.py --input input.avi --output temp_frames/

# Create new video with different settings
python frames_to_video.py --input temp_frames/ --output output.mp4 --fps 30
```

### 4. Speed Up / Slow Down Video

```bash
# Extract all frames
python video_to_frames.py --input normal_video.mp4 --output frames/

# Speed up (60 fps = 2x speed if original was 30fps)
python frames_to_video.py --input frames/ --output fast_video.mp4 --fps 60

# Slow down (15 fps = 0.5x speed if original was 30fps)
python frames_to_video.py --input frames/ --output slow_video.mp4 --fps 15
```

### 5. Extract Sample Frames for Analysis

```bash
# Extract 1 frame per second (assuming 30fps video)
python video_to_frames.py --input video.mp4 --output samples/ --skip 30
```

### 6. Create HD Video from Images

```bash
python frames_to_video.py \
    --input images/ \
    --output hd_video.mp4 \
    --fps 30 \
    --width 1920 \
    --height 1080 \
    --quality 100
```

---

## 🔧 Requirements

Both scripts require:
- Python 3.6+
- OpenCV (`cv2`)

Install dependencies:
```bash
pip install opencv-python
```

---

## 📝 Examples with CNN Detection

### Create Video from Detection Results

If you saved screenshots during camera detection:

```bash
# Create video from screenshots
python frames_to_video.py \
    --input ./ \
    --output detection_video.mp4 \
    --pattern "screenshot_*.jpg" \
    --fps 2
```

### Extract Frames for Manual Review

```bash
# Extract every 30th frame from detection video
python video_to_frames.py \
    --input detection_recording.mp4 \
    --output review_frames/ \
    --skip 30
```

---

## ⚡ Performance Tips

### For Large Videos:
- Use `--skip` to extract fewer frames
- Use JPEG format for smaller file sizes
- Lower quality setting for faster processing

### For High Quality:
- Use `--quality 100` for best JPEG quality
- Use PNG format for lossless compression
- Use higher resolution with `--width` and `--height`

### For Fast Processing:
- Use `mp4v` codec (default)
- Use JPEG format
- Use lower resolution

---

## 🐛 Troubleshooting

### "Could not open video"
- Check video file path
- Ensure video format is supported (mp4, avi, mov, etc.)
- Try different video

### "No images found"
- Check input directory path
- Verify image extensions (jpg, png, etc.)
- Try using `--pattern` to specify exact pattern

### "Video file was not created"
- Check output directory permissions
- Try different codec (`--codec XVID`)
- Verify enough disk space

### Poor Video Quality
- Increase `--quality` (up to 100)
- Use different codec (try `XVID` or `H264`)
- Use higher resolution `--width` and `--height`

---

## 📚 Help

Get full help for each script:

```bash
python frames_to_video.py --help
python video_to_frames.py --help
```

---

**Happy converting! 🎬📸**
