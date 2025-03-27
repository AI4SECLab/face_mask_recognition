# Face Recognition with Masks

A deep learning-based solution for face recognition that works with people wearing masks.

## Features

- Face detection with mask support
- Face recognition with/without masks
- Real-time processing capability
- Easy-to-use interface

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow 2.x
- NumPy
- dlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AI4SECLab/face_mask_recognition.git
cd face_mask_recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. To train the model:
```bash
python mask_train.py
```

2. To run face recognition:

```bash
python mask_detection.py
```

3. To run the web detection

```bash
cd web
python .\manage.py runserver 5000
```

## Web Demo Interface

![image](https://raw.githubusercontent.com/AI4SECLab/face_mask_recognition/refs/heads/dev/docs/bg.png)


![image](https://raw.githubusercontent.com/AI4SECLab/face_mask_recognition/refs/heads/dev/docs/mask.png)

![image](https://raw.githubusercontent.com/AI4SECLab/face_mask_recognition/refs/heads/dev/docs/no_mask.png)

![image](https://raw.githubusercontent.com/AI4SECLab/face_mask_recognition/refs/heads/dev/docs/login.png)

![image](https://raw.githubusercontent.com/AI4SECLab/face_mask_recognition/refs/heads/dev/docs/dashboard.png)

![image](https://raw.githubusercontent.com/AI4SECLab/face_mask_recognition/refs/heads/dev/docs/stat.png)

![image](https://raw.githubusercontent.com/AI4SECLab/face_mask_recognition/refs/heads/dev/docs/stat2.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


