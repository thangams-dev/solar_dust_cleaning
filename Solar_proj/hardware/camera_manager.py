from PIL import Image


class CameraManager:
    @staticmethod
    def capture(device_index: int = 0) -> Image.Image:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is not installed; camera inspection is unavailable.") from error
        camera = cv2.VideoCapture(device_index)
        try:
            if not camera.isOpened():
                raise RuntimeError(f"Webcam device {device_index} is unavailable.")
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Webcam did not return an image.")
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            camera.release()
