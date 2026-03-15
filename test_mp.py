import mediapipe
import sys
import os

print(f"Mediapipe module file: {mediapipe.__file__}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

try:
    print(f"Has solutions? {hasattr(mediapipe, 'solutions')}")
    import mediapipe.solutions.pose
    print("Sucessfully imported mediapipe.solutions.pose")
except Exception as e:
    print(f"Error: {e}")
