import torch
import torch.nn as nn
import cv2
import time
import keyboard
import pyautogui
import numpy as np
import math

class ConvLSTMClassifier(nn.Module):
    def __init__(self, input_dim=2, conv_out_channels=32, lstm_hidden=64, lstm_layers=1):
        super(ConvLSTMClassifier, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_out_channels, kernel_size=5, padding=2)
        self.reLU = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, 2, seq_len)
        x = self.conv1(x)
        x = self.reLU(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last time step
        x = self.fc(x)
        return x

def predict_sequence(model, sequence, device, x_divisor=1280, y_divisor=720):
    model.eval()

    # Normalize and flatten
    normalized = [
        val / x_divisor if i % 2 == 0 else val / y_divisor
        for i, val in enumerate(sequence)
    ]

    # Pad to 385 points (770 values)
    required_len = 385 * 2
    if len(normalized) < required_len:
        normalized.extend([0.0] * (required_len - len(normalized)))
    else:
        normalized = normalized[:required_len]

    seq_tensor = torch.tensor(normalized, dtype=torch.float32).reshape(1, 385, 2).to(device)

    with torch.no_grad():
        output = model(seq_tensor)
        prob = torch.sigmoid(output).item()

    prediction = 1 if prob > 0.5 else 0
    return prediction, prob

def draw_heart(canvas, center, size=30, color=(0, 0, 255), thickness=2):
    """Draw a heart shape using OpenCV."""
    x, y = center
    # Heart parametric equations
    t = np.linspace(0, 2 * math.pi, 100)
    heart_x = size * 16 * np.sin(t) ** 3
    heart_y = -size * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    # Scale and shift
    heart_x = (heart_x + x).astype(int)
    heart_y = (heart_y + y).astype(int)
    # Draw the heart
    for i in range(1, len(heart_x)):
        cv2.line(canvas, (heart_x[i-1], heart_y[i-1]), (heart_x[i], heart_y[i]), color, thickness)

# === Load model ===
model_path = "convlstm_classifier.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConvLSTMClassifier().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Config ===
FPS = 120
sequence = []
last_capture_time = time.time()
recording = False
last_prediction = None

print("Hold SPACE to record gesture")
print("Release SPACE to predict")
print("Press 'Esc' to quit")

# Create canvas
window_size = (1280, 720)
canvas = 255 * np.ones((window_size[1], window_size[0], 3), dtype=np.uint8)
cv2.namedWindow("Mouse Tracking")
cv2.moveWindow("Mouse Tracking", 0, 0)

while True:
    current_time = time.time()
    space_pressed = keyboard.is_pressed('space')

    if space_pressed:
        if not recording:
            print("Recording started...")
            recording = True
            sequence = []
            canvas[:] = 255  # Clear canvas

        if (current_time - last_capture_time) >= 1.0 / FPS:
            last_capture_time = current_time
            x, y = pyautogui.position()
            rel_x = min(max(x, 0), window_size[0] - 1)
            rel_y = min(max(y, 0), window_size[1] - 1)
            sequence.append((rel_x, rel_y))

    # Draw the recorded path
    if sequence:
        for i in range(1, len(sequence)):
            cv2.line(canvas, sequence[i - 1], sequence[i], (0, 255, 0), 2)
        cv2.circle(canvas, sequence[-1], 5, (0, 0, 255), -1)

    if not space_pressed and recording:
        recording = False
        if sequence:
            flat_sequence = [val for pt in sequence for val in pt]
            prediction, prob = predict_sequence(model, flat_sequence, device)
            last_prediction = prediction
            print(f"Prediction: {prediction} | Probability: {prob:.4f}")
            
            # Draw the predicted shape
            canvas[:] = 255  # Clear canvas
            center = (window_size[0] // 2, window_size[1] // 2)
            if prediction == 1:
                cv2.circle(canvas, center, 100, (0, 0, 255), 3)
            else:
                draw_heart(canvas, center, size=10, color=(255, 0, 0), thickness=3)
        else:
            print("No data captured.")
        sequence = []

    # Display the canvas
    cv2.imshow("Mouse Tracking", canvas)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()