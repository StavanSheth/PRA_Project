"""
Generates a modern, high-resolution app icon for Hand Gesture Studio.
Outputs both PNG and multi-resolution Windows ICO format.
"""

import os
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

size = 512
img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# 1. Dark Rounded Background with subtle cyan border
margin = 24
draw.rounded_rectangle(
    [(margin, margin), (size - margin, size - margin)],
    radius=110,
    fill=(15, 23, 42, 255),       # Slate dark
    outline=(56, 189, 248, 220),  # Cyan glow
    width=14,
)

# 2. Glowing circular backdrop for the hand
center = (size // 2, size // 2 + 10)
radius = 170
draw.ellipse(
    [
        (center[0] - radius, center[1] - radius),
        (center[0] + radius, center[1] + radius),
    ],
    fill=(14, 116, 144, 70),       # Translucent cyan
    outline=(56, 189, 248, 120),
    width=6,
)

# 3. Stylized Hand Pose (Victory / Wave gesture)
# Palm base
draw.rounded_rectangle(
    [(200, 260), (312, 380)],
    radius=30,
    fill=(56, 189, 248, 255),
)

# Thumb
draw.polygon([(200, 310), (145, 275), (160, 240), (210, 280)], fill=(56, 189, 248, 255))
draw.ellipse([(140, 240), (170, 275)], fill=(56, 189, 248, 255))

# Index Finger (Extended)
draw.rounded_rectangle(
    [(210, 130), (250, 270)],
    radius=20,
    fill=(56, 189, 248, 255),
)

# Middle Finger (Extended)
draw.rounded_rectangle(
    [(262, 110), (302, 270)],
    radius=20,
    fill=(56, 189, 248, 255),
)

# Ring Finger (Curled)
draw.rounded_rectangle(
    [(314, 230), (354, 310)],
    radius=20,
    fill=(30, 64, 175, 255),
)

# Pinky Finger (Curled)
draw.rounded_rectangle(
    [(364, 250), (400, 320)],
    radius=18,
    fill=(30, 64, 175, 255),
)

# Landmark dots & connections (accent lines)
points = [
    (230, 140), (282, 120), (256, 320), (155, 257)
]
for p in points:
    draw.ellipse([(p[0] - 10, p[1] - 10), (p[0] + 10, p[1] + 10)], fill=(245, 158, 11, 255))

# Save PNG
png_path = os.path.join(ASSETS_DIR, "app_icon.png")
img.save(png_path, "PNG")

# Save multi-size Windows ICO
ico_path = os.path.join(ASSETS_DIR, "app_icon.ico")
img.save(
    ico_path,
    format="ICO",
    sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
)

print(f"Generated icon assets:\n- {png_path}\n- {ico_path}")
