# Watermark Guide for Project Images

<!--
Project: Fraud Detection System using ML
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
-->

## Adding Watermark to Generated Images

### Method 1: Using Python (PIL/Pillow)

```python
"""
Add watermark to images
Developer: Molla Samser (Founder) - https://rskworld.in
"""

from PIL import Image, ImageDraw, ImageFont
import os

def add_watermark(image_path, output_path, watermark_text="rskworld.in"):
    """
    Add watermark to an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save watermarked image
        watermark_text: Text for watermark
    """
    # Open image
    img = Image.open(image_path)
    width, height = img.size
    
    # Create watermark layer
    watermark = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    
    # Try to use a nice font, fallback to default
    try:
        font_size = int(width / 30)  # Scale font with image size
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position: bottom right with padding
    x = width - text_width - 20
    y = height - text_height - 20
    
    # Draw watermark with transparency
    draw.text(
        (x, y),
        watermark_text,
        font=font,
        fill=(255, 255, 255, 128)  # White with 50% opacity
    )
    
    # Composite watermark onto image
    watermarked = Image.alpha_composite(img.convert('RGBA'), watermark)
    watermarked = watermarked.convert('RGB')
    
    # Save
    watermarked.save(output_path, quality=95)
    print(f"Watermarked image saved to: {output_path}")

# Usage
if __name__ == '__main__':
    # Add watermark to your generated images
    add_watermark('image1.png', 'image1_watermarked.png')
    add_watermark('image2.png', 'image2_watermarked.png')
    add_watermark('image3.png', 'image3_watermarked.png')
```

### Method 2: Using ImageMagick (Command Line)

```bash
# Install ImageMagick first, then:
convert input_image.png -gravity southeast \
  -pointsize 24 -fill "rgba(255,255,255,0.5)" \
  -annotate +20+20 "rskworld.in" \
  output_image.png
```

### Method 3: Using Online Tools

1. **Watermark.ws** - https://watermark.ws
2. **Canva** - Add text element, position bottom right, reduce opacity
3. **Photopea** - Free online Photoshop alternative
4. **GIMP** - Free image editor

### Method 4: Using Canva Template

1. Go to Canva.com
2. Create custom size: 1920x1080
3. Add your generated image as background
4. Add text: "rskworld.in"
5. Position: Bottom right corner
6. Font: Modern sans-serif
7. Color: White
8. Opacity: 30-50%
9. Export as PNG/JPG

## Watermark Specifications

### Recommended Settings:
- **Text**: "rskworld.in"
- **Position**: Bottom right corner
- **Padding**: 20-30 pixels from edges
- **Font**: Modern sans-serif (Arial, Roboto, Montserrat)
- **Size**: 2-3% of image width
- **Color**: White or light gray
- **Opacity**: 30-50% (semi-transparent)
- **Style**: Subtle, professional, not distracting

### Alternative Styles:
1. **Subtle**: White text, 30% opacity
2. **Bold**: White text with black outline, 50% opacity
3. **Minimal**: Small gray text, 40% opacity
4. **Branded**: Add small logo + text

## Batch Processing Script

```python
"""
Batch add watermark to multiple images
Developer: Molla Samser (Founder) - https://rskworld.in
"""

from PIL import Image, ImageDraw, ImageFont
import os
import glob

def batch_watermark(input_folder, output_folder, watermark_text="rskworld.in"):
    """
    Add watermark to all images in a folder.
    
    Args:
        input_folder: Folder containing images
        output_folder: Folder to save watermarked images
        watermark_text: Watermark text
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image formats
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    
    for ext in extensions:
        for image_path in glob.glob(os.path.join(input_folder, ext)):
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_folder, filename)
            
            # Open and process
            img = Image.open(image_path)
            width, height = img.size
            
            # Create watermark
            watermark = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(watermark)
            
            try:
                font_size = int(width / 30)
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Position
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = width - text_width - 20
            y = height - text_height - 20
            
            # Draw
            draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 128))
            
            # Composite and save
            watermarked = Image.alpha_composite(img.convert('RGBA'), watermark)
            watermarked.convert('RGB').save(output_path, quality=95)
            print(f"Processed: {filename}")

# Usage
if __name__ == '__main__':
    batch_watermark('images/', 'images_watermarked/')
```

## Quick Reference

### For AI Image Generators:
- **DALL-E**: Mention "watermark text 'rskworld.in' in bottom right corner" in prompt
- **Midjourney**: Add `--watermark "rskworld.in"` or mention in prompt
- **Stable Diffusion**: Use ControlNet or add in post-processing
- **ChatGPT Image**: Include watermark request in prompt

### For Design Software:
- **Photoshop**: Layer → Text → Position bottom right → Opacity 30-50%
- **GIMP**: Text tool → Bottom right → Layer opacity
- **Canva**: Text element → Position → Opacity slider
- **Figma**: Text frame → Position → Opacity

## Contact

**Founder**: Molla Samser  
**Designer & Tester**: Rima Khatun  
**Website**: https://rskworld.in  
**Email**: help@rskworld.in, support@rskworld.in

© 2025 RSK World. All rights reserved.

