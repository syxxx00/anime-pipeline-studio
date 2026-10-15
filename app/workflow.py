from __future__ import annotations

from copy import deepcopy

from .config import Settings
from .prompting import DEFAULT_NEGATIVE, ensure_system_prefix
from .schemas import GenerateRequest, ModelProfile
from .model_conditioning import NEGATIVE_BASELINES
from .regional_conditioning import apply_regional_conditioning


def _link(node: str, output: int = 0) -> list[object]:
    return [node, output]


def _negative(request: GenerateRequest, fallback: str) -> str:
    text = request.negative_prompt.strip()
    return (text or fallback) if request.use_default_negative_prompt else text


def _apply_model_highres(workflow: dict, request: GenerateRequest, *,
                         sampler_id: str, decoder_id: str) -> dict:
    """Refine this model's own latent/conditions; never substitute another model.

    The base sampler and its text/LoRA connections stay intact. NewBie's
    established highres/face graph is deliberately not routed through here.
    """
    if not request.highres_enabled:
        return workflow
    upscale_id, refine_id = 'highres_upscale', 'highres_sample'
    if upscale_id in workflow or refine_id in workflow:
        raise ValueError('高细节节点标识与已有工作流冲突')
    base = workflow[sampler_id]
    decoder = workflow[decoder_id]
    if base['class_type'] != 'KSampler' or decoder['class_type'] != 'VAEDecode':
        raise ValueError('高细节需要当前模型自身的基础采样与VAE解码')
    workflow[upscale_id] = {'class_type': 'LatentUpscaleBy', 'inputs': {
        'samples': _link(sampler_id), 'upscale_method': 'bislerp', 'scale_by': request.highres_scale,
    }}
    workflow[refine_id] = {'class_type': 'KSampler', 'inputs': {
        **deepcopy(base['inputs']), 'latent_image': _link(upscale_id),
        'steps': request.highres_steps, 'denoise': request.highres_denoise,
    }}
    decoder['class_type'] = 'VAEDecodeTiled'
    decoder['inputs'].update(samples=_link(refine_id), tile_size=512, overlap=64,
                             temporal_size=64, temporal_overlap=8)
    return workflow


def _build_newbie_workflow(request: GenerateRequest, settings: Settings, filename_prefix: str) -> dict:
    """Build a ComfyUI API workflow without relying on a saved desktop workflow."""
    positive = ensure_system_prefix(request.prompt)
    negative = _negative(request, DEFAULT_NEGATIVE)

    workflow: dict[str, dict] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": settings.diffusion_model,
                "weight_dtype": request.weight_dtype or "default",
            },
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": settings.gemma_clip,
                "clip_name2": settings.jina_clip,
                "type": "newbie",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": settings.vae},
        },
        "4": {
            "class_type": "ModelSamplingNewbie",
            "inputs": {"model": _link("1"), "shift": request.shift},
        },
        "5": {
            "class_type": "RescaleCFG",
            "inputs": {"model": _link("4"), "multiplier": request.cfg_rescale},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": _link("2"), "text": positive},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": _link("2"), "text": negative},
        },
        "8": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": request.width,
                "height": request.height,
                "batch_size": request.batch_size,
            },
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": _link("5"),
                "positive": _link("6"),
                "negative": _link("7"),
                "latent_image": _link("8"),
                "seed": request.seed,
                "steps": request.steps,
                "cfg": request.cfg,
                "sampler_name": request.sampler,
                "scheduler": request.scheduler,
                "denoise": 1.0,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": _link("9"), "vae": _link("3")},
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {"images": _link("10"), "filename_prefix": filename_prefix},
        },
    }

    if request.lora_name:
        # The NewBie-specific loader maps DiT keys correctly. It is deliberately
        # model-only: Gemma/Jina text encoders should not receive SDXL CLIP LoRA keys.
        workflow["12"] = {
            "class_type": "NewBieLoraModelOnly",
            "inputs": {
                "model": _link("1"),
                "lora_name": request.lora_name,
                "strength": request.lora_strength,
                "enabled": True,
            },
        }
        workflow["4"]["inputs"]["model"] = _link("12")

    if request.highres_enabled:
        # Preserve the 1024px composition, then add detail in latent space. A
        # low denoise value avoids redrawing the character identity and pose.
        del workflow["10"]
        del workflow["11"]
        workflow["13"] = {
            "class_type": "LatentUpscaleBy",
            "inputs": {"samples": _link("9"), "upscale_method": "bislerp", "scale_by": request.highres_scale},
        }
        workflow["14"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": _link("5"), "positive": _link("6"), "negative": _link("7"),
                "latent_image": _link("13"), "seed": request.seed,
                "steps": request.highres_steps, "cfg": request.cfg,
                "sampler_name": request.sampler, "scheduler": request.scheduler,
                "denoise": request.highres_denoise,
            },
        }
        workflow["15"] = {
            "class_type": "VAEDecodeTiled",
            "inputs": {"samples": _link("14"), "vae": _link("3"), "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8},
        }
        workflow["16"] = {"class_type": "SaveImage", "inputs": {"images": _link("15"), "filename_prefix": filename_prefix}}

    if request.face_detail_enabled:
        decoded_node = "15" if request.highres_enabled else "10"
        save_node = "16" if request.highres_enabled else "11"
        del workflow[save_node]
        workflow["17"] = {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": "bbox/face_yolov8m.pt"},
        }
        workflow["18"] = {
            "class_type": "FaceDetailer",
            "inputs": {
                "image": _link(decoded_node), "model": _link("5"), "clip": _link("2"), "vae": _link("3"),
                "guide_size": 768, "guide_size_for": True, "max_size": 1024,
                "seed": request.seed, "steps": request.face_detail_steps, "cfg": request.cfg,
                "sampler_name": request.sampler, "scheduler": request.scheduler,
                "positive": _link("6"), "negative": _link("7"), "denoise": request.face_detail_denoise,
                "feather": 12, "noise_mask": True, "force_inpaint": True,
                "bbox_threshold": 0.35, "bbox_dilation": 10, "bbox_crop_factor": 2.4,
                "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93,
                "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False", "drop_size": 10,
                "bbox_detector": _link("17"), "wildcard": "", "cycle": 1,
                "inpaint_model": False, "noise_mask_feather": 20,
                "tiled_encode": True, "tiled_decode": True,
            },
        }
        workflow["19"] = {"class_type": "SaveImage", "inputs": {"images": _link("18"), "filename_prefix": filename_prefix}}

    return workflow


def _build_z_image_workflow(request: GenerateRequest, settings: Settings, filename_prefix: str) -> dict:
    negative = _negative(request, NEGATIVE_BASELINES['z_image_turbo'])
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": settings.z_diffusion_model, "weight_dtype": request.weight_dtype or "fp8_e5m2"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": settings.z_clip, "type": "lumina2", "device": "cpu"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": settings.z_vae}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": _link("1"), "shift": request.shift}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("2"), "text": request.prompt.strip()}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("2"), "text": negative}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": request.width, "height": request.height, "batch_size": request.batch_size}},
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": _link("4"), "positive": _link("5"), "negative": _link("6"),
                "latent_image": _link("7"), "seed": request.seed, "steps": request.steps,
                "cfg": request.cfg, "sampler_name": request.sampler, "scheduler": request.scheduler,
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": _link("8"), "vae": _link("3")}},
        "10": {"class_type": "SaveImage", "inputs": {"images": _link("9"), "filename_prefix": filename_prefix}},
    }
    return _apply_model_highres(workflow, request, sampler_id='8', decoder_id='9')


def _build_anima_workflow(request: GenerateRequest, settings: Settings, filename_prefix: str) -> dict:
    positive = "You are an assistant designed to generate anime images based on textual prompts.\n<Prompt Start>\n" + request.prompt.strip()
    negative = _negative(request, NEGATIVE_BASELINES['anima'])
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": settings.anima_diffusion_model, "weight_dtype": request.weight_dtype or "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": settings.anima_clip, "type": "stable_diffusion", "device": "cpu"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": settings.anima_vae}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": _link("1"), "shift": request.shift}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("2"), "text": positive}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("2"), "text": negative}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": request.width, "height": request.height, "batch_size": request.batch_size}},
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": _link("4"), "positive": _link("5"), "negative": _link("6"),
                "latent_image": _link("7"), "seed": request.seed, "steps": request.steps,
                "cfg": request.cfg, "sampler_name": request.sampler, "scheduler": request.scheduler,
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": _link("8"), "vae": _link("3")}},
        "10": {"class_type": "SaveImage", "inputs": {"images": _link("9"), "filename_prefix": filename_prefix}},
    }
    return _apply_model_highres(workflow, request, sampler_id='8', decoder_id='9')


def _build_illustrious_workflow(request: GenerateRequest, settings: Settings, filename_prefix: str) -> dict:
    """SDXL-compatible anime baseline used as a deliberately different third opinion."""
    positive = request.prompt.strip()
    negative = _negative(request, NEGATIVE_BASELINES['illustrious'])
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": settings.illustrious_checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("1", 1), "text": positive}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("1", 1), "text": negative}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {
            "width": request.width, "height": request.height, "batch_size": request.batch_size,
        }},
        "5": {"class_type": "KSampler", "inputs": {
            "model": _link("1"), "positive": _link("2"), "negative": _link("3"),
            "latent_image": _link("4"), "seed": request.seed, "steps": request.steps,
            "cfg": request.cfg, "sampler_name": request.sampler, "scheduler": request.scheduler,
            "denoise": 1.0,
        }},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": _link("5"), "vae": _link("1", 2)}},
        "7": {"class_type": "SaveImage", "inputs": {"images": _link("6"), "filename_prefix": filename_prefix}},
    }
    if request.lora_name:
        workflow['8'] = {'class_type': 'LoraLoader', 'inputs': {
            'model': _link('1'), 'clip': _link('1', 1), 'lora_name': request.lora_name,
            'strength_model': request.lora_strength, 'strength_clip': request.lora_strength,
        }}
        workflow['2']['inputs']['clip'] = _link('8', 1)
        workflow['3']['inputs']['clip'] = _link('8', 1)
        workflow['5']['inputs']['model'] = _link('8')
    return _apply_model_highres(workflow, request, sampler_id='5', decoder_id='6')


def build_anima_base_workflow(request: GenerateRequest, settings: Settings, filename_prefix: str) -> dict:
    """Build the Anima source stage without replacing explicit sampling choices."""
    from .bridge_sampling import source_request
    stage = request if request.model_id == ModelProfile.ANIMA_TURBO else source_request(request, profile='anima')
    return _build_anima_workflow(stage, settings, filename_prefix)


def build_newbie_refine_workflow(request: GenerateRequest, settings: Settings, input_name: str, filename_prefix: str) -> dict:
    """Stage 2: preserve Anima geometry while applying NewBie identity, then detail the face."""
    positive = ensure_system_prefix(request.prompt)
    negative = _negative(request, DEFAULT_NEGATIVE)
    workflow: dict[str, dict] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": settings.diffusion_model, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": settings.gemma_clip, "clip_name2": settings.jina_clip, "type": "newbie", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": settings.vae}},
        "4": {"class_type": "ModelSamplingNewbie", "inputs": {"model": _link("1"), "shift": request.shift}},
        "5": {"class_type": "RescaleCFG", "inputs": {"model": _link("4"), "multiplier": request.cfg_rescale}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("2"), "text": positive}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("2"), "text": negative}},
        "8": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "9": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": _link("8"), "vae": _link("3"), "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}},
        "10": {"class_type": "KSampler", "inputs": {"model": _link("5"), "positive": _link("6"), "negative": _link("7"), "latent_image": _link("9"), "seed": request.seed, "steps": request.steps, "cfg": request.cfg, "sampler_name": request.sampler, "scheduler": request.scheduler, "denoise": request.hybrid_denoise}},
    }
    if request.lora_name:
        workflow["16"] = {"class_type": "NewBieLoraModelOnly", "inputs": {"model": _link("1"), "lora_name": request.lora_name, "strength": request.lora_strength, "enabled": True}}
        workflow["4"]["inputs"]["model"] = _link("16")
    latent_out = "10"
    if request.highres_enabled:
        workflow["11"] = {"class_type": "LatentUpscaleBy", "inputs": {"samples": _link("10"), "upscale_method": "bislerp", "scale_by": request.highres_scale}}
        workflow["12"] = {"class_type": "KSampler", "inputs": {"model": _link("5"), "positive": _link("6"), "negative": _link("7"), "latent_image": _link("11"), "seed": request.seed, "steps": request.highres_steps, "cfg": request.cfg, "sampler_name": request.sampler, "scheduler": request.scheduler, "denoise": request.highres_denoise}}
        latent_out = "12"
    workflow["20"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": _link(latent_out), "vae": _link("3"), "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}}
    if request.face_detail_enabled:
        workflow["13"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/face_yolov8m.pt"}}
        workflow["14"] = {"class_type": "FaceDetailer", "inputs": {
            "image": _link("20"), "model": _link("5"), "clip": _link("2"), "vae": _link("3"),
            "guide_size": 768, "guide_size_for": True, "max_size": 1024,
            "seed": request.seed, "steps": request.face_detail_steps, "cfg": request.cfg,
            "sampler_name": request.sampler, "scheduler": request.scheduler,
            "positive": _link("6"), "negative": _link("7"), "denoise": request.face_detail_denoise,
            "feather": 12, "noise_mask": True, "force_inpaint": True,
            "bbox_threshold": 0.35, "bbox_dilation": 10, "bbox_crop_factor": 2.4,
            "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93,
            "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False", "drop_size": 10,
            "bbox_detector": _link("13"), "wildcard": "", "cycle": 1,
            "inpaint_model": False, "noise_mask_feather": 20, "tiled_encode": True, "tiled_decode": True,
        }}
        image_out = "14"
    else:
        image_out = "20"
    workflow["15"] = {"class_type": "SaveImage", "inputs": {"images": _link(image_out), "filename_prefix": filename_prefix}}
    return workflow


def build_text_clean_workflow(request: GenerateRequest, settings: Settings, image_name: str, mask_name: str, filename_prefix: str) -> dict:
    """文字清理：检测在引擎外完成（漫画文字检测器），这里只做遮罩局部重绘。

    与 FaceDetailer 同构：检测器产出遮罩 -> VAEEncodeForInpaint -> 中低步数重绘。
    负向明确压文字，重绘幅度默认较高（0.55），否则伪文字笔画抹不干净。
    """
    positive = ensure_system_prefix(
        "clean anime illustration, coherent detailed surfaces, seamless background textures, "
        "natural lighting, high quality, no text anywhere"
    )
    negative = (
        "text, letters, words, typography, glyph strokes, watermark, signature, logo, "
        "subtitle, caption, speech bubble, " + (request.negative_prompt.strip() or DEFAULT_NEGATIVE)
    )
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": settings.diffusion_model, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": settings.gemma_clip, "clip_name2": settings.jina_clip, "type": "newbie", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": settings.vae}},
        "4": {"class_type": "ModelSamplingNewbie", "inputs": {"model": _link("1"), "shift": request.shift}},
        "5": {"class_type": "RescaleCFG", "inputs": {"model": _link("4"), "multiplier": request.cfg_rescale}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("2"), "text": positive}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("2"), "text": negative}},
        "8": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "9": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "10": {"class_type": "ImageToMask", "inputs": {"image": _link("9"), "channel": "red"}},
        "11": {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": _link("8"), "vae": _link("3"), "mask": _link("10"), "grow_mask_by": 12}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": _link("5"), "positive": _link("6"), "negative": _link("7"), "latent_image": _link("11"),
            "seed": request.seed, "steps": request.text_clean_steps, "cfg": request.cfg,
            "sampler_name": request.sampler, "scheduler": request.scheduler, "denoise": request.text_clean_denoise,
        }},
        "13": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _link("12"), "vae": _link("3"), "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}},
        "14": {"class_type": "SaveImage", "inputs": {"images": _link("13"), "filename_prefix": filename_prefix}},
    }


def build_region_repair_workflow(request: GenerateRequest, settings: Settings, image_name: str, mask_name: str, filename_prefix: str) -> dict:
    """Explicit local development retains source pixels before masked sampling.

    Context richness is separate from the latent strategy. Library/history edits
    must not silently erase their selected detail merely because they lack the
    modern scene-context metadata. Automatic hand/text erasure graphs are separate.
    """
    spec = request.semantic_spec if isinstance(request.semantic_spec, dict) else {}
    context = ", ".join(str(value) for key in ("style", "lighting") for value in (spec.get(key) or []) if str(value).strip())
    if request.repair_context_policy == 'scene_context_with_local_override':
        # A previous costume/pose in the scene is context, not an instruction
        # to undo the requested local change. The final composite still protects
        # every unselected pixel; this wording is not a quality guarantee.
        positive = ensure_system_prefix(
            "<scene_context>\n" + request.repair_context_prompt + "\n</scene_context>\n"
            "<local_edit_override>\n" + request.repair_instruction.strip() + "\n</local_edit_override>\n"
            "Use the full scene as context for identity, space, light and style. "
            "Within the selected region the local edit overrides conflicting earlier scene details. "
            "Do not restore replaced details there. Preserve all unselected pixels and composition."
        )
    else:
        positive = ensure_system_prefix(
            f"localized anime illustration repair, {request.repair_instruction.strip()}, preserve surrounding pixels and composition"
            + (f", match existing design: {context}" if context else "")
        )
    negative = _negative(request, DEFAULT_NEGATIVE)
    if not any(token.strip().casefold() == "visible seam" for token in negative.split(",")):
        negative = "visible seam, " + negative
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": settings.diffusion_model, "weight_dtype": request.weight_dtype or "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": settings.gemma_clip, "clip_name2": settings.jina_clip, "type": "newbie", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": settings.vae}},
        "4": {"class_type": "ModelSamplingNewbie", "inputs": {"model": _link("1"), "shift": request.shift}},
        "5": {"class_type": "RescaleCFG", "inputs": {"model": _link("4"), "multiplier": request.cfg_rescale}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("2"), "text": positive}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("2"), "text": negative}},
        "8": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "9": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "10": {"class_type": "ImageToMask", "inputs": {"image": _link("9"), "channel": "red"}},
        "11": {"class_type": "VAEEncodeTiled", "inputs": {
            "pixels": _link("8"), "vae": _link("3"), "tile_size": 512,
            "overlap": 64, "temporal_size": 64, "temporal_overlap": 8,
        }},
        "17": {"class_type": "SetLatentNoiseMask", "inputs": {
            "samples": _link("11"), "mask": _link("10"),
        }},
        "12": {"class_type": "KSampler", "inputs": {
            "model": _link("5"), "positive": _link("6"), "negative": _link("7"), "latent_image": _link("17"),
            "seed": request.seed, "steps": request.region_repair_steps, "cfg": request.cfg,
            "sampler_name": request.sampler, "scheduler": request.scheduler, "denoise": request.region_repair_denoise,
        }},
        "13": {"class_type": "VAEDecodeTiled", "inputs": {"samples": _link("12"), "vae": _link("3"), "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}},
        "15": {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": _link("8"), "source": _link("13"), "mask": _link("10"),
            "x": 0, "y": 0, "resize_source": True,
        }},
        "14": {"class_type": "SaveImage", "inputs": {"images": _link("15"), "filename_prefix": filename_prefix}},
    }
    if request.lora_name:
        workflow["16"] = {"class_type": "NewBieLoraModelOnly", "inputs": {
            "model": _link("1"), "lora_name": request.lora_name, "strength": request.lora_strength, "enabled": True,
        }}
        workflow["4"]["inputs"]["model"] = _link("16")
    return workflow


def build_hand_repair_workflow(request: GenerateRequest, settings: Settings, image_name: str, mask_name: str, filename_prefix: str) -> dict:
    """手部修复（check 系统的修复 pass）：本地门控检测已产出遮罩，这里只做遮罩区域重绘。

    旧方法（图内 FaceDetailer + hand_yolov8s）证伪的四个根因在此修复：
    - 检测+门控在引擎外完成（hand_yolov9c，conf 0.5，脸重叠丢弃），图内不二次检测
    - 正向用手部专用 prompt，不把整条场景 prompt 灌进手裁剪区
    - guide_size 768（文档：SDXL 级模型 768–1024；旧值 512 放大不足）
    - denoise 默认 0.45（文档：崩坏的手 0.5–0.6 才能重构；旧值 0.3 只抛光不修结构）
    MaskToSEGS + DetailerForEach 与 FaceDetailer 同一套 crop→放大→重绘→贴回机制。

    "部节分明"证伪（e0041aff5712 实测）后的三条修正：
    - prompt 不写 "correct finger joints"：动漫渲染不画关节线，字面要求关节 → 模型画成
      玩偶式分节手指；负向补 doll joints（Danbooru 标签，专指这种关节球分节）
    - 风格/光照标签从语义规格带进修复 prompt：零上下文重绘 = 默认动漫手（硬线稿苍白肤色），
      与原图柔和赛璐璐+环境光断裂，贴回去像拼接部件
    - feather 12→28：1536 图上 12px 羽化遮罩边缘过硬
    """
    spec = request.semantic_spec if isinstance(request.semantic_spec, dict) else {}
    style = ", ".join(str(tag) for tag in spec.get("style") or [] if str(tag).strip())
    lighting = ", ".join(str(tag) for tag in spec.get("lighting") or [] if str(tag).strip())
    context = ", ".join(part for part in (style, lighting) if part)
    positive = ensure_system_prefix(
        "detailed anime hand, five fingers, natural relaxed fingers, natural hand anatomy, "
        + (context + ", " if context else "")
        + "high quality, masterpiece"
    )
    negative = (
        "bad hands, extra fingers, missing fingers, fused fingers, mutated hands, "
        "deformed hands, bad anatomy, extra limbs, doll joints, " + (request.negative_prompt.strip() or DEFAULT_NEGATIVE)
    )
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": settings.diffusion_model, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": settings.gemma_clip, "clip_name2": settings.jina_clip, "type": "newbie", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": settings.vae}},
        "4": {"class_type": "ModelSamplingNewbie", "inputs": {"model": _link("1"), "shift": request.shift}},
        "5": {"class_type": "RescaleCFG", "inputs": {"model": _link("4"), "multiplier": request.cfg_rescale}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("2"), "text": positive}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": _link("2"), "text": negative}},
        "8": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "9": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "10": {"class_type": "ImageToMask", "inputs": {"image": _link("9"), "channel": "red"}},
        "11": {"class_type": "MaskToSEGS", "inputs": {"mask": _link("10"), "combined": False, "crop_factor": 2.6, "bbox_fill": True, "drop_size": 10, "contour_fill": False}},
        "12": {"class_type": "DetailerForEach", "inputs": {
            "image": _link("8"), "segs": _link("11"), "model": _link("5"), "clip": _link("2"), "vae": _link("3"),
            "guide_size": 768, "guide_size_for": True, "max_size": 1024,
            "seed": request.seed, "steps": request.hand_detail_steps, "cfg": request.cfg,
            "sampler_name": request.sampler, "scheduler": request.scheduler,
            "positive": _link("6"), "negative": _link("7"), "denoise": request.hand_detail_denoise,
            "feather": 28, "noise_mask": True, "force_inpaint": True,
            "wildcard": "", "cycle": 1,
        }},
        "13": {"class_type": "SaveImage", "inputs": {"images": _link("12"), "filename_prefix": filename_prefix}},
    }


def build_workflow(request: GenerateRequest, settings: Settings, filename_prefix: str) -> dict:
    """Build a model-specific API workflow without a saved desktop workflow."""
    if request.model_id == ModelProfile.Z_IMAGE_TURBO:
        workflow = _build_z_image_workflow(request, settings, filename_prefix)
    elif request.model_id == ModelProfile.ANIMA_TURBO:
        workflow = _build_anima_workflow(request, settings, filename_prefix)
    elif request.model_id == ModelProfile.ILLUSTRIOUS:
        workflow = _build_illustrious_workflow(request, settings, filename_prefix)
    else:
        workflow = _build_newbie_workflow(request, settings, filename_prefix)
    return apply_regional_conditioning(workflow, request)
