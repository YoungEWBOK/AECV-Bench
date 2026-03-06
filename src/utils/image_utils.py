"""
Utility functions for image processing.
"""
import base64
import os


def get_image_mime_type(image_path: str) -> str:
    """
    Determine the MIME type of an image by inspecting file magic bytes,
    falling back to the file extension.

    Args:
        image_path: Path to the image file

    Returns:
        MIME type string (e.g. 'image/png', 'image/jpeg')
    """
    # Detect from magic bytes first
    try:
        with open(image_path, 'rb') as f:
            header = f.read(16)
        if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            return 'image/webp'
        if header[:3] == b'\xff\xd8\xff':
            return 'image/jpeg'
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            return 'image/png'
        if header[:6] in (b'GIF87a', b'GIF89a'):
            return 'image/gif'
        if header[:2] == b'BM':
            return 'image/bmp'
    except OSError:
        pass

    # Fallback to extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    return mime_types.get(ext, 'image/png')


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

