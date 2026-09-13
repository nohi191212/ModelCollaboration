# -*- coding: utf-8 -*-
"""Omni client: thin wrapper around Qwen3-Omni-30B-A3B for probe experiments.

Follows the official Qwen3-Omni transformers recipe:
  text = processor.apply_chat_template(messages, ...)
  audios, images, videos = process_mm_info(messages, use_audio_in_video=...)
  inputs = processor(text=text, audio=audios, images=images, videos=videos, ...)
  text_ids, audio = model.generate(**inputs, speaker=..., ...)
"""
import os
import re
import torch


class OmniClient:
    def __init__(self, model_path, device="cuda:0", dtype=torch.float16,
                 use_audio_in_video=False, max_new_tokens=96, speaker="Ethan"):
        import qwen_omni_utils  # noqa: F401
        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            AutoProcessor,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device
        self.use_audio_in_video = use_audio_in_video
        self.max_new_tokens = max_new_tokens
        self.speaker = speaker
        print(f"[Omni] loaded from {model_path}", flush=True)

    @torch.inference_mode()
    def _generate(self, messages):
        from qwen_omni_utils import process_mm_info
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=self.use_audio_in_video
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids, audio = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            speaker=self.speaker,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=self.use_audio_in_video,
            use_cache=True,
        )
        seq = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
        out = self.processor.batch_decode(
            seq[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return out[0].strip()

    # ---------- modality helpers ----------
    def ask_image(self, image_path, question):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ]}
        ]
        return self._generate(messages)

    def ask_audio(self, audio_path, question):
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": question},
            ]}
        ]
        return self._generate(messages)

    def ask_audio_pair(self, audio1, audio2, question):
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio1},
                {"type": "audio", "audio": audio2},
                {"type": "text", "text": question},
            ]}
        ]
        return self._generate(messages)

    def ask_video(self, video_path, question):
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question},
            ]}
        ]
        return self._generate(messages)

    # ---------- answer parsing ----------
    @staticmethod
    def parse_yes_no(text):
        t = text.strip().lower()
        if re.search(r"\b(yes|true|same|real|benign|normal|genuine)\b", t):
            return 1
        if re.search(r"\b(no|false|different|fake|anomaly|abnormal|forged)\b", t):
            return 0
        return None

    @staticmethod
    def parse_choice(text, choices):
        t = text.strip().lower()
        for c in choices:
            if c.lower() in t:
                return c
        return None
