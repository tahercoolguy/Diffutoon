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
        prompt: str = Input(description="Prompt for stage 1", default="best quality, perfect anime illustration, orange clothes, night, a girl is dancing, smile, solo, black silk stockings"),
        prompt_2: str = Input(description="Prompt for stage 2", default="best quality, perfect anime illustration, light, a girl is dancing, smile, solo"),
    ) -> Path:
        fps = 20
        end_frame_id = 5 * fps  # 5 seconds at 20 fps
        config_stage_1_template = {
            "models": {
                "model_list": [
                    "models/stable_diffusion/aingdiffusion_v12.safetensors",
                    "models/ControlNet/control_v11p_sd15_softedge.pth",
                    "models/ControlNet/control_v11f1p_sd15_depth.pth"
                ],
                "textual_inversion_folder": "models/textual_inversion",
                "device": "cuda",
                "lora_alphas": [],
                "controlnet_units": [
                    {
                        "processor_id": "softedge",
                        "model_path": "models/ControlNet/control_v11p_sd15_softedge.pth",
                        "scale": 0.5
                    },
                    {
                        "processor_id": "depth",
                        "model_path": "models/ControlNet/control_v11f1p_sd15_depth.pth",
                        "scale": 0.5
                    }
                ]
            },
            "data": {
                "input_frames": {
                    "video_file": input_video,
                    "image_folder": None,
                    "height": 512,
                    "width": 512,
                    "start_frame_id": 0,
                    "end_frame_id": None
                },
                "controlnet_frames": [
                    {
                        "video_file": input_video,
                        "image_folder": None,
                        "height": 512,
                        "width": 512,
                        "start_frame_id": 0,
                        "end_frame_id": None
                    },
                    {
                        "video_file": input_video,
                        "image_folder": None,
                        "height": 512,
                        "width": 512,
                        "start_frame_id": 0,
                        "end_frame_id": None
                    }
                ],
                "output_folder": "data/examples/diffutoon_edit/color_video",
                "fps": 25
            },
            "smoother_configs": [
                {
                    "processor_type": "FastBlend",
                    "config": {}
                }
            ],
            "pipeline": {
                "seed": 0,
                "pipeline_inputs": {
                    "prompt": "best quality, perfect anime illustration",
                    "negative_prompt": "verybadimagenegative_v1.3",
                    "cfg_scale": 7.0,
                    "clip_skip": 1,
                    "denoising_strength": 0.9,
                    "num_inference_steps": 20,
                    "animatediff_batch_size": 8,
                    "animatediff_stride": 4,
                    "unet_batch_size": 8,
                    "controlnet_batch_size": 8,
                    "cross_frame_attention": True,
                    "smoother_progress_ids": [-1],
                    # The following parameters will be overwritten. You don't need to modify them.
                    "input_frames": [],
                    "num_frames": 30,
                    "width": 512,
                    "height": 512,
                    "controlnet_frames": []
                }
            }
        }

        config_stage_1 = config_stage_1_template.copy()
        config_stage_1["data"]["input_frames"] = {
            "video_file": input_video,
            "image_folder": None,
            "height": 512,
            "width": 512,
            "start_frame_id": 0,
            "end_frame_id": end_frame_id
        }
        config_stage_1["data"]["controlnet_frames"] = [config_stage_1["data"]["input_frames"], config_stage_1["data"]["input_frames"]]
        config_stage_1["data"]["output_folder"] = "/content/color_video"
        config_stage_1["data"]["fps"] = fps
        config_stage_1["pipeline"]["pipeline_inputs"]["prompt"] = prompt

        runner = SDVideoPipelineRunner()
        runner.run(config_stage_1)

        config_stage_2_template = {
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
                    "end_frame_id": None
                },
                "controlnet_frames": [
                    {
                        "video_file": input_video,
                        "image_folder": None,
                        "height": 1024,
                        "width": 1024,
                        "start_frame_id": 0,
                        "end_frame_id": None
                    },
                    {
                        "video_file": input_video,
                        "image_folder": None,
                        "height": 1024,
                        "width": 1024,
                        "start_frame_id": 0,
                        "end_frame_id": None
                    }
                ],
                "output_folder": "/content/output",
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
                    # The following parameters will be overwritten. You don't need to modify them.
                    "input_frames": [],
                    "num_frames": 30,
                    "width": 1536,
                    "height": 1536,
                    "controlnet_frames": []
                }
            }
        }

        config_stage_2 = config_stage_2_template.copy()
        config_stage_2["data"]["input_frames"] = {
            "video_file": input_video,
            "image_folder": None,
            "height": 1024,
            "width": 1024,
            "start_frame_id": 0,
            "end_frame_id": end_frame_id
        }
        config_stage_2["data"]["controlnet_frames"][0] = {
            "video_file": "/content/color_video/video.mp4",
            "image_folder": None,
            "height": config_stage_2["data"]["input_frames"]["height"],
            "width": config_stage_2["data"]["input_frames"]["width"],
            "start_frame_id": 0,
            "end_frame_id": end_frame_id
        }
        config_stage_2["data"]["controlnet_frames"][1] = config_stage_2["data"]["input_frames"]
        config_stage_2["data"]["output_folder"] = "/content/edit_video"
        config_stage_2["data"]["fps"] = fps
        config_stage_2["pipeline"]["pipeline_inputs"]["prompt"] = prompt_2

        runner = SDVideoPipelineRunner()
        runner.run(config_stage_2)

        return Path("/content/edit_video/video.mp4")