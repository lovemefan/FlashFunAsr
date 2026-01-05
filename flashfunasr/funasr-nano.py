import json
import logging
import os
import random
import re
import string
import time
import traceback

import torch
import torch.nn as nn

from flashfunasr.config import Config, AudioEncoderConf, AudioAdaptorConf, SamplingParams, FunASRNanoConfig
from flashfunasr.engine.llm_engine import LLMEngine
from flashfunasr.modules.audio_encoder.adaptor import AudioAdaptor
from flashfunasr.modules.audio_encoder.sense_voice_encoder import SenseVoiceEncoderSmall
from flashfunasr.utils.audio import extract_fbank, load_audio_text_image_video
from flashfunasr.utils.frontend import WavFrontend
from flashfunasr.utils.loader import to_device

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class FunASRNano(nn.Module):
    def __init__(
        self, config: Config = None, device='cuda'
    ):
        super().__init__()
        self.config = Config('/data/zlf/pretrained_model/Fun-ASR-Nano-2512') if config is None else config
        # audio encoder
        self.audio_encoder_conf = AudioEncoderConf()
        self.feat_permute = self.audio_encoder_conf.feat_permute

        self.audio_encoder = SenseVoiceEncoderSmall(**self.audio_encoder_conf.__dict__)
        self.audio_encoder.load_weights(self.config, device=device)
        self.audio_encoder.eval()

        # llm
        self.llm = LLMEngine(**self.config.__dict__)

        # adaptor
        self.audio_adaptor_conf = AudioAdaptorConf()
        self.audio_adaptor = AudioAdaptor(**self.audio_adaptor_conf.__dict__)
        self.audio_adaptor.load_weights(self.config, device=device)
        self.audio_adaptor.eval()

    def forward_export(self, speech, speech_lengths, **kwargs):
        x, olens = self.audio_encoder(speech, speech_lengths)
        encoder_out, encoder_out_lens = self.audio_adaptor(x, olens)
        return encoder_out, encoder_out_lens

    def encode(self, speech, speech_lengths):
        # audio encoder
        if self.feat_permute:
            encoder_out, encoder_out_lens = self.audio_encoder(
                speech.permute(0, 2, 1), speech_lengths
            )
        else:
            encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)

        return encoder_out, encoder_out_lens

    def data_template(self, data):
        system, user, assistant = [], [], []
        for i, item in enumerate(data):
            role = item["role"]
            content = item["content"]
            if role == "system":
                system.append(content)
            elif role == "user":
                if "audio" in item:
                    audio = item["audio"]
                    content = [content, audio]
                user.append(content)
            elif role == "assistant":
                assistant.append(content)

        system = system * len(user)

        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
        }

        return contents

    def data_load_speech(
        self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs
    ):
        system = contents["system"]
        user = contents["user"]
        assistant = contents["assistant"]
        pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        do_think = True
        sys_prompt = True
        if "dataset_conf" in kwargs:
            do_think = kwargs["dataset_conf"].get("do_think", True)
            sys_prompt = kwargs["dataset_conf"].get("sys_prompt", True)

        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        input_source_ids = []
        for i, (system_prompt, user_prompt, target_out) in enumerate(
            zip(system, user, assistant)
        ):
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            if len(input_ids) > kwargs.get("max_token_length", 1500):
                break
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            if i == 0:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}"
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            if not do_think:
                source_input += "<think>\n\n</think>\n\n"

            splits = pattern.split(source_input)
            source_ids = []
            fbank_mask_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
            speech, speech_lengths = [], []
            for k, sub_str in enumerate(splits):
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                    fbank_mask_i += [0] * len(sub_token)
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                        "<|endofspeech|>", ""
                    )
                    if sub_str.startswith("!"):
                        sub_str = sub_str[1:]
                        if sub_str.startswith("!"):  # !!: audio sample point
                            sub_str = audio
                        try:
                            time1 = time.perf_counter()
                            data_src = load_audio_text_image_video(
                                sub_str, fs=frontend.fs, **kwargs
                            )
                            time2 = time.perf_counter()
                            meta_data["load_data"] = f"{time2 - time1:0.3f}"
                        except Exception as e:
                            logging.error(
                                f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                            )

                        speech, speech_lengths = extract_fbank(
                            data_src,
                            data_type=kwargs.get("data_type", "sound"),
                            frontend=frontend,

                        )  # speech: [b, T, d]

                        time3 = time.perf_counter()
                        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
                        meta_data["batch_data_time"] = (
                            speech_lengths.sum().item()
                            * frontend.frame_shift
                            * frontend.lfr_n
                            / 1000
                        )

                        if self.feat_permute:
                            speech = speech.permute(0, 2, 1)

                        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                        olens = 1 + (olens - 3 + 2 * 1) // 2
                        fake_token_len_i = (olens - 1) // 2 + 1
                        fake_token = [0] * fake_token_len_i
                        fbank_beg_i = len(source_ids)
                        source_ids += fake_token
                        fbank_mask_i += [1] * len(fake_token)

            fbank_beg += [fbank_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            source_mask = [-100] * len(source_ids)
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            fbank_mask += fbank_mask_i
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)

        input_ids = torch.tensor(
            input_ids, dtype=torch.int64
        )  # [: self.max_token_length]
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

        fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
        fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
        fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        if len(fbank) > 0:
            speech = torch.nn.utils.rnn.pad_sequence(
                fbank, batch_first=True, padding_value=0.0
            )
            speech_lengths = torch.nn.utils.rnn.pad_sequence(
                fbank_lens, batch_first=True, padding_value=-1
            )
        else:
            speech = []
            speech_lengths = []
        output = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "fbank_mask": fbank_mask[None, :],
            "fbank_beg": fbank_beg[None,],
            "fake_token_len": fake_token_len[None, :],
            "input_ids": input_ids[None,],
            "attention_mask": attention_mask[None,],
            "labels_ids": labels,
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }

        return output

    def inference_prepare(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        meta_data = {}

        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        output = self.data_load_speech(
            contents, tokenizer, frontend, meta_data=meta_data, **kwargs
        )
        batch = to_device(output, kwargs["device"])
        self.audio_encoder = self.audio_encoder.to(kwargs["device"])
        self.audio_encoder.eval()
        self.audio_adaptor = self.audio_adaptor.to(kwargs["device"])
        self.audio_adaptor.eval()

        # audio encoder
        speech = batch["speech"]

        if len(speech) > 0:
            if "audio_embedding" in kwargs and "audio_embedding_lens" in kwargs:
                encoder_out = kwargs["audio_embedding"]
                encoder_out_lens = kwargs["audio_embedding_lens"]
            else:
                speech_lengths = batch["speech_lengths"][:, 0]
                # fp16
                if kwargs.get("fp16", False):
                    speech = speech.to(torch.float16)
                elif kwargs.get("bf16", False):
                    speech = speech.to(torch.bfloat16)

                # audio encoder
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

                # audio_adaptor
                encoder_out, encoder_out_lens = self.audio_adaptor(
                    encoder_out, encoder_out_lens
                )
                meta_data["audio_adaptor_out"] = encoder_out
                meta_data["audio_adaptor_out_lens"] = encoder_out_lens

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        fbank_beg = batch["fbank_beg"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("tearchforing", False):
            input_ids = source_ids

        input_ids[input_ids < 0] = 0
        input_ids = to_device(input_ids, kwargs["device"])
        inputs_embeds = self.llm.model_runner.model.model.embed_tokens(input_ids)

        batch_size, token_num, dims = inputs_embeds.shape

        fake_token_len[fake_token_len < 0] = 0
        fbank_beg[fbank_beg < 0] = 0

        speech_idx = 0
        for batch_idx in range(batch_size):
            for turn_id in range(fbank_beg.shape[1]):
                fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                if fbank_beg_idx > 0:
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    speech_token = encoder_out[speech_idx, :speech_token_len, :]

                    try:
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token
                    except Exception as e:
                        #
                        logging.error(f"{str(e)}, {traceback.format_exc()}")
                        logging.info(
                            f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                        )
                        speech_token_len = encoder_out_lens[speech_idx].item()
                        speech_token = encoder_out[speech_idx, :speech_token_len, :]
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token

                    speech_idx += 1
        return inputs_embeds, contents, batch, source_ids, meta_data

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        hotwords = kwargs.get("hotwords", [])
        if len(hotwords) > 0:
            hotwords = ", ".join(hotwords)
            prompt = f"请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            prompt += f"热词列表：[{hotwords}]\n"
        else:
            prompt = ""
        language = kwargs.get("language", "auto")
        if language not in ("auto", "zh", "en", "ja"):
            language = "auto"
        if language == "auto":
            prompt += "语音转写"
        else:
            LANGUAGE_MAP = {"zh": "中文", "en": "英文", "ja": "日文"}
            prompt += f"语音转写成{LANGUAGE_MAP[language]}"
        itn = kwargs.get("itn", True)
        if not itn:
            prompt += "，不进行文本规整"
        prompt += "："

        print('-' * 50)
        print(f"使用提示词为：\n{prompt}")
        print('-' * 50)
        new_data_in = []
        for data in data_in:
            if isinstance(data, str):
                new_data_in.append(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": f"{prompt}<|startofspeech|>!{data}<|endofspeech|>",
                        },
                        {"role": "assistant", "content": "null"},
                    ]
                )
            elif isinstance(data, torch.Tensor):
                new_data_in.append(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": f"{prompt}<|startofspeech|>!!<|endofspeech|>",
                            "audio": data,
                        },
                        {"role": "assistant", "content": "null"},
                    ]
                )
        data_in = new_data_in

        if key is None:
            key = []
            for _ in data_in:
                chars = string.ascii_letters + string.digits
                key.append(
                    "rand_key_" + "".join(random.choice(chars) for _ in range(13))
                )

        return self.inference_llm(
            data_in,
            data_lengths=data_lengths,
            key=key,
            tokenizer=tokenizer or self.llm.tokenizer,
            frontend=frontend,
            **kwargs,
        )

    def inference_llm(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )
        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        with torch.amp.autocast(
            'cuda', enabled=True if llm_dtype != "fp32" else False, dtype=dtype_map[llm_dtype]
        ):
            label = contents["assistant"][-1]

            # self.llm.model_runner.model = self.llm.model_runner.model.to(dtype_map[llm_dtype])
            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])
            llm_kwargs = kwargs.get("llm_kwargs", {})
            if not kwargs.get("teachforing", False):
                generated_tokens = self.llm.generate(
                    source_ids.cpu().numpy().tolist(),
                    inputs_embeds,
                    SamplingParams()
                )
                response = generated_tokens[0]['text']
                loss = None
            else:
                labels_ids = batch["labels_ids"]
                labels_ids[labels_ids == -1] = -100
                attention_mask = batch.get("attention_mask", None)
                model_outputs = self.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels_ids,
                    **llm_kwargs,
                )

                preds = torch.argmax(model_outputs.logits, -1)[:, source_ids.shape[1] :]
                response = tokenizer.batch_decode(
                    preds,
                    add_special_tokens=False,
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]
                loss = model_outputs.loss.item()

        ibest_writer = None

        results = []
        response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        result_i = {
            "key": key[0],
            "text": re.sub(r'\s+', ' ', response.replace("/sil", " ")),
            "text_tn": response_clean,
            "label": label,
        }
        if loss is not None:
            result_i["loss"] = loss
        results.append(result_i)

        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = response.replace("\n", " ")
            ibest_writer["label"][key[0]] = label.replace("\n", " ")
            ibest_writer["text_tn"][key[0]] = response_clean

        return results, meta_data

    @staticmethod
    def from_pretrained(model: str = None, **kwargs):
        from funasr import AutoModel

        model, kwargs = AutoModel.build_model(
            model=model, trust_remote_code=True, **kwargs
        )

        return model, kwargs

if __name__ == '__main__':
    m = FunASRNano()
    wav_path = "/data/zlf/code/Fun-ASR/1765468243.307207.wav"
    frontend = WavFrontend()
    start = time.time()
    res = m.inference(data_in=[wav_path],
                      cache={},
                      batch_size=1,
                      hotwords=["开放时间"],
                      language="auto",  # auto, zh, en, ja
                      itn=True,  # or False
                      frontend=frontend,
                      device='cuda'
                      )
    text = res[0][0]["text"]
    end = time.time()
    print(f"RTF: {(end - start) / 33}")
    print(text)