import os
from cog import BasePredictor, Input, Path
import sys
sys.path.append('/content/DiffSynth-Studio')
os.chdir('/content/DiffSynth-Studio')

from diffsynth import SDVideoPipelineRunner

class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        input_video: Path = Input(description="Input video"),
        prompt: str = Input(description="Prompt for the video", default="best quality, perfect anime illustration, light"),
    ) -> Path:
        config_template = {
            "models": {
                "model_list": [
                    "models/stable_diffusion/aingdiffusion_v12.safetensors",
                    "models/AnimateDiff/mm_sd_v15_v2.ckpt",
                    "models/ControlNet/control_v11f1e_sd15_tile.pth",
                    "models/ControlNet/control_v11p_sd15_lineart.pth"
                ],
                "textual_inversion_folder": "models/textual_inversion",
                "device": "cuda",
                "lora_alphas": [],
                "controlnet_units": [
                    {
                        "processor_id": "tile",
                        "model_path": "models/ControlNet/control_v11f1e_sd15_tile.pth",
                        "scale": 0.5
                    },
                    {
                        "processor_id": "lineart",
                        "model_path": "models/ControlNet/control_v11p_sd15_lineart.pth",
                        "scale": 0.5
                    }
                ]
            },
            "data": {
                "input_frames": {
                    "video_file": input_video,
                    "image_folder": None,
                    "height": 1024,
                    "width": 1024,
                    "start_frame_id": 0,
                    "end_frame_id": None  # Process all frames
                },
                "controlnet_frames": [
                    {
                        "video_file": input_video,
                        "image_folder": None,
                        "height": 1024,
                        "width": 1024,
                        "start_frame_id": 0,
                        "end_frame_id": None  # Process all frames
                    },
                    {
                        "video_file": input_video,
                        "image_folder": None,
                        "height": 1024,
                        "width": 1024,
                        "start_frame_id": 0,
                        "end_frame_id": None  # Process all frames
                    }
                ],
                "output_folder": "/content/toon_video",
                "fps": 25
            },
            "pipeline": {
                "seed": 0,
                "pipeline_inputs": {
                    "prompt": "best quality, perfect anime illustration, light",
                    "negative_prompt": "verybadimagenegative_v1.3",
                    "cfg_scale": 7.0,
                    "clip_skip": 2,
                    "denoising_strength": 1.0,
                    "num_inference_steps": 10,
                    "animatediff_batch_size": 16,
                    "animatediff_stride": 8,
                    "unet_batch_size": 1,
                    "controlnet_batch_size": 1,
                    "cross_frame_attention": False,
                    "input_frames": [],
                    "num_frames": 30,
                    "width": 1536,
                    "height": 1536,
                    "controlnet_frames": []
                }
            }
        }

        config = config_template.copy()
        config["data"]["input_frames"]["video_file"] = input_video
        config["data"]["controlnet_frames"][0]["video_file"] = input_video
        config["data"]["controlnet_frames"][1]["video_file"] = input_video
        config["data"]["output_folder"] = "/content/toon_video"
        config["data"]["fps"] = 25
        config["pipeline"]["pipeline_inputs"]["prompt"] = prompt

        runner = SDVideoPipelineRunner()
        runner.run(config)

        return Path("/content/toon_video/video.mp4")