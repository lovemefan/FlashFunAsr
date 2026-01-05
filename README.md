input 

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。

**上下文信息：**

热词列表：[开放时间]
语音转写成中文：
```
!为占位符，需要使用speech token 替换掉

```json
{
    "speech": "torch.Size([1, 560, 94])",
    "speech_lengths": "torch.Size([1, 1])",
    "fbank_mask": "torch.Size([1, 35])",
    "fbank_beg": "torch.Size([1, 1])",
    "fake_token_len": "torch.Size([1, 1])",
    "input_ids": "torch.Size([1, 37])",
    "attention_mask": "torch.Size([1, 37])",
    "labels_ids": "torch.Size([37])",
    "source_ids": "torch.Size([1, 35])",
    "target_ids": "torch.Size([1, 2])"
}
```