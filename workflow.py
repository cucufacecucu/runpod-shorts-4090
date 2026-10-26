"""Single-stage distilled LTX-2 AV graph using ComfyUI core nodes."""
CHECKPOINT = "ltx-2-19b-distilled-fp8.safetensors"
ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
AUDIO_RULE = (
    " Audio: only environmental ambience and sound effects synchronized with the scene. "
    "No human voices, no speech, no dialogue, no whispers, no singing, no chanting, "
    "no vocalizations, no lyrics. No subtitles or written captions."
)


def build_workflow(prompt, seed):
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    return {
        "1": node("CheckpointLoaderSimple", ckpt_name=CHECKPOINT),
        "2": node("LTXAVTextEncoderLoader", text_encoder=ENCODER,
                  ckpt_name=CHECKPOINT, device="default"),
        "3": node("CLIPTextEncode", clip=["2", 0], text=prompt + AUDIO_RULE),
        "4": node("CLIPTextEncode", clip=["2", 0],
                  text="speech, dialogue, voice, singing, whispers, lyrics, subtitles, watermark"),
        "5": node("LTXVConditioning", positive=["3", 0], negative=["4", 0], frame_rate=24.0),
        "6": node("LTXVAudioVAELoader", ckpt_name=CHECKPOINT),
        "7": node("EmptyLTXVLatentVideo", width=704, height=1280, length=241, batch_size=1),
        "8": node("LTXVEmptyLatentAudio", audio_vae=["6", 0],
                  frames_number=241, frame_rate=24.0, batch_size=1),
        "9": node("LTXVConcatAVLatent", video_latent=["7", 0], audio_latent=["8", 0]),
        "10": node("RandomNoise", noise_seed=seed),
        "11": node("CFGGuider", model=["1", 0], positive=["5", 0], negative=["5", 1], cfg=1.0),
        "12": node("KSamplerSelect", sampler_name="euler_ancestral"),
        "13": node("ManualSigmas", sigmas="1,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0"),
        "14": node("SamplerCustomAdvanced", noise=["10", 0], guider=["11", 0],
                   sampler=["12", 0], sigmas=["13", 0], latent_image=["9", 0]),
        "15": node("LTXVSeparateAVLatent", av_latent=["14", 1]),
        "16": node("VAEDecodeTiled", samples=["15", 0], vae=["1", 2],
                   tile_size=256, overlap=64, temporal_size=64, temporal_overlap=8),
        "17": node("LTXVAudioVAEDecode", samples=["15", 1], audio_vae=["6", 0]),
        "18": node("CreateVideo", images=["16", 0], audio=["17", 0], fps=24.0),
        "19": node("SaveVideo", video=["18", 0], filename_prefix="shorts/generated",
                   **{"format": "mp4", "format.codec": "h264"}),
    }


def validate_input(payload):
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 6000:
        raise ValueError("prompt must contain 1–6000 characters")
    seed = payload.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("seed must be an integer between 0 and 2**63-1")
    return prompt.strip(), seed
