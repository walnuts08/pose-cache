# 智姿护航

基于计算机视觉的跑步姿态分析与科学训练平台原型。用户上传一段侧面跑步视频，系统完成姿态识别、骨架回放、关键指标计算和针对性建议。

这个版本对应比赛 PPT 中的程序层闭环：

```text
上传视频 → OpenCV 解码 → MediaPipe 识别 33 点 → 指标计算
        → 骨架视频写回 → Streamlit 展示 → 下载报告
```

## 已实现功能

- 支持 MP4、MOV、M4V、AVI、WebM、MKV 视频上传与预览。
- 使用 MediaPipe 1.2 Tasks API 的 `PoseLandmarker` 视频模式逐帧识别。
- 使用 OpenCV 绘制 33 点骨架和实时膝角，生成浏览器可播放的 WebM 结果视频。
- 计算左右膝角、左右髋角、躯干前倾、姿态识别覆盖率。
- 从踝/脚跟/脚尖的垂直运动轨迹估计触地事件、步频和左右脚着地方式。
- 根据本次指标、画面质量和用户目标步频生成训练建议。
- 展示姿态曲线、着地方式占比和关键指标表。
- 下载骨架视频、逐帧指标 CSV、JSON 分析报告。
- 默认预置 MediaPipe Lite 模型，比赛现场可离线运行。

## 运行

当前机器已配置 Python 3.12.10 和三个核心依赖。推荐双击 `start.bat`，或在 PowerShell 中运行：

```powershell
.\start.ps1
```

浏览器打开：

```text
http://localhost:8501
```

也可以显式指定解释器：

```powershell
$env:POSECOACH_PYTHON = "C:\Users\ROG\AppData\Local\Programs\Python\Python312\python.exe"
.\start.ps1
```

## 环境版本

当前项目按本机实际环境实现：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.10 |
| OpenCV | 5.1.0 |
| MediaPipe | 1.2.3 |
| Streamlit | 1.67.0 |

重新安装依赖：

```powershell
C:\Users\ROG\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
```

## 拍摄要求

测量质量首先取决于视频质量：

- 单人、全身入镜，头顶到脚尖不能出画。
- 手机垂直于跑动方向，镜头高度尽量与髋部一致。
- 推荐连续拍摄 8 到 15 秒，包含 6 次以上触地。
- 保持日常配速，背景干净，避免宽松衣物遮住膝踝。
- 固定机位，避免数码变焦、慢动作和剧烈防抖裁切。

## 目录

| 文件 | 作用 |
| --- | --- |
| `app.py` | Streamlit 上传、进度、结果和下载页面 |
| `posecoach/pipeline.py` | OpenCV 视频解码、任务调度和结果视频写入 |
| `posecoach/model.py` | MediaPipe 模型下载与加载 |
| `posecoach/metrics.py` | 关节角度、触地、步频、着地方式和质量评估 |
| `posecoach/recommendations.py` | 跑姿规则与训练建议生成 |
| `posecoach/render.py` | OpenCV 骨架和 HUD 绘制 |
| `models/pose_landmarker_lite.task` | Google 官方姿态识别模型 |
| `tests/` | 指标单元测试与完整处理链冒烟测试 |

运行结果保存在 `data/runs/<run_id>/`，该目录不会提交到 Git。

## 指标口径

### 关节角度

系统把归一化关键点转换到视频像素坐标，再计算三点夹角。数值 180° 表示接近伸直，数值越小表示弯曲越明显。

- 膝角：髋–膝–踝。
- 髋角：肩–髋–膝。
- 躯干前倾：肩部中点与髋部中点连线，相对于画面竖直方向的夹角。

### 触地与步频

把脚部相对髋部的垂直位置按躯干长度归一化，平滑后寻找每一侧的局部最低点。先从同侧两个触地峰的间隔估计步幅周期，再换算为每分钟步数。该方法适合固定机位、连续跑步、侧面拍摄的短视频。

### 着地方式

在触地帧比较脚跟与脚尖的相对高度：

- 脚跟显著低于脚尖：偏后掌。
- 脚跟与脚尖接近：中掌。
- 脚尖显著低于脚跟：偏前掌。

鞋子、草地、拍摄角度和关键点抖动都会影响该分类，因此页面把它作为参考，不把它当作临床结论。

## 测试

运行单元测试和冒烟测试：

```powershell
C:\Users\ROG\AppData\Local\Programs\Python\Python312\python.exe -m unittest discover -s tests -v
```

单独检查模型能否加载：

```powershell
C:\Users\ROG\AppData\Local\Programs\Python\Python312\python.exe -c "from posecoach.model import create_pose_landmarker; m=create_pose_landmarker(); print('model ok'); m.close()"
```

## 已知边界

- 单相机二维关键点不能替代三维动作捕捉，关节角度存在透视误差。
- 宽松衣物、低照度、跑出画面和多人重叠会降低识别率。
- 慢动作视频需要恢复到真实时间播放后再分析，否则步频会被低估。
- 当前建议引擎是透明规则系统，适合比赛演示；后续可在受控、脱敏样本上训练个性化模型。
- 结果用于运动教学和产品原型演示，不构成医疗诊断。

## 后续可扩展

- 增加正面 + 侧面双机位同步，计算膝内扣和左右对称性。
- 增加脚着地瞬间与支撑阶段切分，让膝角建议更贴近运动生物力学定义。
- 增加历史训练档案和同条件对比。
- 用匿名样本校准步频、着地分类和不同速度下的参考区间。
