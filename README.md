# 🧠 Real-Time Object, Age & Gender Detection

This project combines **YOLOv8** for object detection with **OpenCV Deep Neural Networks (DNN)** for real-time **age and gender prediction** using a connected USB camera.

---

## 🚀 Features
- Real-time **object detection** using YOLOv8.
- **Face detection** with Haar Cascade.
- **Age & Gender classification** using Caffe models.
- Works with **any USB or laptop camera**.
- Easy to extend for emotion or face recognition.

---

## 🛠️ Requirements

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📁 Folder Setup

Create a folder named `models` and place these files inside it:

- `deploy_age.prototxt`
- `age_net.caffemodel`
- `deploy_gender.prototxt`
- `gender_net.caffemodel`
- `haarcascade_frontalface_default.xml`

---

## ▶️ Run the Project

```bash
python main.py
```

If you have multiple cameras, change this line in `main.py`:
```python
usb_camera_index = 0  # or 1 depending on your system
```

Press **`q`** to exit the program.

---

## 📷 Output Example

The app shows live camera feed with:
- Green boxes for detected objects (via YOLOv8)
- Blue boxes for detected faces
- Labels showing `Gender, Age`

---

## 📚 Model References
- YOLOv8 from [Ultralytics](https://github.com/ultralytics/ultralytics)
- Age & Gender model from [OpenCV’s Deep Learning Module (Caffe)](https://github.com/spmallick/learnopencv/tree/master/AgeGender)

---

## 🧑‍💻 Author
Developed by **Mack Shema**  
Cybersecurity & AI Enthusiast
