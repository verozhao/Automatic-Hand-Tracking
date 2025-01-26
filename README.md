# Automatic Hand Tracking with Mobile SAM

Automatic hand tracking pipeline using MediaPipe and Mobile SAM.

## Setup

1. Create environment:
```bash
conda create -n hand_tracking python=3.12
conda activate hand_tracking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download Mobile SAM weights:
```bash
wget https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt
```

## Usage
```python
from hand_tracking import track_hands
track_hands('input.mp4', 'output.mp4')
```

## Features
- Hand detection using MediaPipe
- Segmentation using Mobile SAM
- Multi-hand tracking support
- Progress visualization
- GPU acceleration when available
