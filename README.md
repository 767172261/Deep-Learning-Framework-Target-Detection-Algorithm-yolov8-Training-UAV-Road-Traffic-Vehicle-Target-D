深度学习框架目标检测算法yolov8训练无人机道路交通车辆目标检测数据集 通过训练出的无人机交通车辆检测数据集权重，建立基于深度学习的无人机交通车辆检测系统
@[toc]
以下文字及代码仅供参考学习。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a1e686209b8f4dafaa5ec4be73947d73.png)

无人机道路交通车辆目标检测数据集
无人机：DJI mini 3
数据类型：原始图片+打好的目标检测标签
总内存大小：940M（2708张）
图片分辨率：1920*1080
采集高度：40～90m
采集角度：90度
类别：工程车辆，铰接式公共汽车，公共汽车，汽车，货运车，摩托车，小型公共汽车，卡车
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2143bdb313814ed880fcf1ca334f0330.png)

如何使用YOLOv8训练这个数据集，并建立一个车辆目标检测系统。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/776ce9074cb7491d9d2135e02c62eea4.png)

### 环境配置

首先确保环境中已安装必要的库和依赖项。这里以YOLOv8为例进行说明：

1. **创建并激活虚拟环境**（可选但推荐）：
    ```bash
    python -m venv yolov8-env
    source yolov8-env/bin/activate  # 在Windows上使用 `yolov8-env\Scripts\activate`
    ```

2. **安装YOLOv8及依赖**：
    ```bash
    pip install ultralytics
    ```

### 数据准备

创建一个`data.yaml`文件来定义数据集路径和类别信息。假设你的数据集结构如下：

```
/path/to/dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

那么，你的`data.yaml`文件内容可能如下所示：

```yaml
train: /path/to/dataset/images/train
val: /path/to/dataset/images/val

nc: 8  # number of classes
names: ['Engineering_Vehicle', 'Articulated_Bus', 'Bus', 'Car', 'Freight_Car', 'Motorcycle', 'Mini_Bus', 'Truck']  # class names
```

确保每张图片都有对应的`.txt`标签文件，格式遵循YOLO标准（即每个标签包含一个边界框的中心点坐标、宽度、高度和类别ID）。

### 模型训练

使用以下Python脚本来启动YOLOv8的训练过程：

```python
from ultralytics import YOLO

# 加载预训练的YOLOv8模型或从头开始训练
model = YOLO('yolov8n.yaml')  # 使用YOLOv8 nano架构作为起点，或者加载预训练权重如 'yolov8n.pt'

# 开始训练
results = model.train(data='/path/to/data.yaml', epochs=100, imgsz=640)

# 训练完成后，结果会保存在 runs/detect/expN 文件夹内
```

- `data`: 指向你的`data.yaml`文件路径。
- `epochs`: 设置训练周期数，可以根据需要调整。
- `imgsz`: 输入图像尺寸，通常设置为640或其他适合你数据集大小的值。

### 推理与验证

训练完成后，可以使用以下代码对新的图像或视频进行推理：

```python
# 加载训练好的模型
model = YOLO('/path/to/runs/detect/exp/weights/best.pt')  # 替换为你的最佳模型路径

def detect_image(image_path):
    """
    对单张图像执行车辆检测并显示结果。
    :param image_path: 图像路径
    """
    results = model(image_path)

    for result in results:
        annotated_frame = result.plot()  # 自动绘制边界框和标签
        cv2.imshow("Vehicle Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def detect_video(video_path=0):
    """
    对视频流或视频文件进行实时检测。
    :param video_path: 视频路径或摄像头ID，默认为0表示使用默认摄像头
    """
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 推理
        results = model(frame)

        for result in results:
            annotated_frame = result.plot()
            cv2.imshow("Vehicle Detection", annotated_frame)

        if cv2.waitKey(1) == ord('q'):  # 按 Q 键退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 示例：检测一张图片
    detect_image("path/to/test/image.jpg")

    # 示例：检测视频或摄像头
    # detect_video("videos/test_video.mp4")  # 使用视频文件
    # detect_video(0)  # 使用摄像头
```c从境配置、数据准备、模型训练以及推理。请根据实际情况替换所有路径为你自己的实际路径。

为了建立一个基于YOLOv8的深度学习无人机车辆检测系统，并提供一个简单的用户界面，我们可以使用Python的`tkinter`库来创建图形用户界面（GUI），让用户可以选择图片或视频文件进行目标检测。以下是详细的步骤和代码示例。

### 环境配置

确保你已经按照之前的说明配置好了环境，并安装了必要的依赖项：

```bash
pip install ultralytics opencv-python-headless pillow
```

### 代码实现

#### 1. 创建项目结构

首先，建议将你的项目结构组织如下：

```
drone_vehicle_detection/
├── weights/
│   └── best.pt                # 训练好的权重文件
├── data/
│   └── data.yaml              # 数据集配置文件
├── images/
│   ├── test1.jpg
│   └── test2.jpg
├── videos/
│   └── test_video.mp4
├── main.py                    # 主程序入口
└── requirements.txt           # 依赖文件
```

#### 2. 编写主程序代码 (`main.py`)

以下是一个完整的Python脚本，用于加载训练好的模型权重，并提供了一个简单的GUI界面让用户选择图片或视频文件进行车辆检测。

```python
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import threading

class VehicleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("无人机车辆检测系统")
        
        # 加载模型
        self.model = YOLO('weights/best.pt')  # 替换为你的权重路径
        
        # GUI组件
        self.select_image_button = tk.Button(root, text="选择图片", command=self.detect_image)
        self.select_image_button.pack(pady=10)
        
        self.select_video_button = tk.Button(root, text="选择视频", command=self.detect_video)
        self.select_video_button.pack(pady=10)
        
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
    
    def detect_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self._run_detection(file_path, is_video=False)
    
    def detect_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if file_path:
            threading.Thread(target=self._run_detection, args=(file_path, True)).start()
    
    def _run_detection(self, file_path, is_video):
        if is_video:
            cap = cv2.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.model(frame)
                annotated_frame = results[0].plot()
                
                # 转换颜色通道并显示
                img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)
                self.root.update_idletasks()
                self.root.update()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            results = self.model(file_path)
            annotated_image = results[0].plot()
            
            # 转换颜色通道并调整大小以适应窗口
            img = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((640, 480), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)
            messagebox.showinfo("完成", "图像处理完成")

if __name__ == '__main__':
    root = tk.Tk()
    app = VehicleDetectionApp(root)
    root.mainloop()
```

### 解释

- **GUI界面**：通过`tkinter`创建了一个简单的界面，包含两个按钮：“选择图片”和“选择视频”，以及一个用于显示结果的标签。
- **模型加载**：在类初始化时加载了训练好的YOLOv8模型。
- **图像检测**：当用户点击“选择图片”按钮后，会弹出文件选择对话框，用户可以选择一张图片进行检测。检测结果会在界面上显示。
- **视频检测**：当用户点击“选择视频”按钮后，同样会弹出文件选择对话框，用户可以选择一个视频文件进行实时检测。由于视频处理可能需要一些时间，这里使用了多线程处理，避免阻塞主界面。
- **结果显示**：对于图像，直接在界面上显示处理后的图像；对于视频，则逐帧处理并在界面上更新每一帧的结果。

请根据实际情况调整路径和参数设置。这段代码提供了一个基础框架，
