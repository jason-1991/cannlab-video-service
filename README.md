# cannlab-video-service

在华为昇腾（Ascend）NPU 上运行的视频生成服务：基于 CANN + PyTorch（torch_npu）与 Wan2.1 系列模型，为短剧生产画布提供「关键帧 → 短片」的自有出片通道。

## 接口（异步任务制）

所有业务接口需带请求头 `X-Service-Token`。

- `GET /health` — 服务与模型状态
- `POST /generate-video` — 提交生成任务，返回 `job_id`
  ```json
  {
    "prompt": "暴风雨夜的灯塔，海浪拍打礁石，电影感",
    "image_url": "https://example.com/first-frame.png",
    "width": 832,
    "height": 480,
    "seconds": 5
  }
  ```
  首帧图也可以用 `image_b64` 以 Base64 直接传入（二选一，可都省略走纯文生视频）。
- `GET /jobs/{job_id}` — 查询任务状态（queued / running / done / failed）
- `GET /jobs/{job_id}/result` — 下载成片（MP4，H.264）

## 模型路线

- 首选 Wan2.1-T2V-1.3B（480p），在 910B 单卡上已有社区实测先例
- 跑通后升档尝试 Wan2.1-I2V-14B-480P 图生视频
- 服务层用 diffusers 的 WanPipeline / WanImageToVideoPipeline，上机前先用官方 `generate.py` 做最低成本验证

## 生命周期

- 服务进程维护活动时间戳；看门狗（watchdog.sh）在任务完成且 10 分钟无新任务时自动关机，配合云平台「关机不计费」实现按需开关、省算力额度
- 服务只监听本机 8900 端口，对外通过反向 SSH 隧道（tunnel.sh）经跳板机暴露，并有共享令牌校验

## 文件

| 文件 | 说明 |
| --- | --- |
| `video_service.py` | 服务主体（FastAPI + torch_npu + diffusers） |
| `setup_video.sh` | 环境一键搭建：探活 → 权重下载 → 官方 CLI 证明片 → 服务启动 |
| `watchdog.sh` | 闲置自动关机看门狗 |
| `tunnel.sh` | 反向 SSH 隧道 |

## 环境

- 华为 CANNLab（HiDevLab）开发环境，1×NPU，官方 CANN + PyTorch 镜像
- Python 3，依赖：torch / torch_npu / diffusers / fastapi / uvicorn / imageio[ffmpeg]
