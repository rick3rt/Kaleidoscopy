import imageio
import cv2


def write_gif(output_path, frames, fps=10):
    """
    Creates a GIF from a list of frames.

    Parameters:
    - frames: List of images (NumPy arrays in BGR format from OpenCV).
    - output_path: Path to save the GIF (e.g., 'output.gif').
    - fps: Frames per second for the GIF.
    """
    # Convert frames from BGR (OpenCV) to RGB (for GIF)
    rgb_frames = frames
    # rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    # Save frames as a GIF
    imageio.mimsave(output_path, rgb_frames, fps=fps)


def write_video(output_path, frames, fps=30):
    """
    Writes a stack of images to an MP4 video file.

    Parameters:
    - frames: List of images (NumPy arrays in BGR format).
    - output_path: Path to save the MP4 file (e.g., 'output.mp4').
    - fps: Frames per second for the video.
    """
    if not frames:
        raise ValueError("No frames provided to write to video.")

    # Get frame dimensions from the first frame
    height, width, channels = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        if frame.shape != (height, width, channels):
            raise ValueError("All frames must have the same dimensions.")
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")
