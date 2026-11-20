# 姿态模型

应用默认读取 `pose_landmarker_lite.task`。

该文件来自 Google 官方 MediaPipe 模型仓库：

- 模型页：https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- 下载地址：https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

若模型缺失，应用首次分析时会尝试自动下载；也可以运行：

```powershell
.\start.ps1
```

或直接执行：

```powershell
C:\Users\ROG\AppData\Local\Programs\Python\Python312\python.exe scripts\download_model.py
```
