#!/usr/bin/env python3
"""
生成 STT 测试音频文件

生成一个简单的 WAV 文件（16kHz 单声道）用于 STT (Speech-to-Text) 测试。
使用合成的语音或正弦波音调作为假数据。
"""

import sys
from pathlib import Path
import math

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
AUDIO_RAW_DIR = PROJECT_ROOT / "audio" / "raw"
OUTPUT_FILE = AUDIO_RAW_DIR / "test_sample.wav"


def generate_sine_wave_audio(output_path: Path, duration_sec: float = 3.0, frequency: float = 440.0):
    """
    生成正弦波音频（备用方法）
    
    参数:
        output_path: 输出文件路径
        duration_sec: 音频时长（秒）
        frequency: 频率（Hz）
    """
    try:
        import torch
        import torchaudio
        
        print(f"🔊 生成 {duration_sec} 秒正弦波测试音频...")
        print(f"   频率: {frequency}Hz")
        
        sample_rate = 16000
        num_samples = int(sample_rate * duration_sec)
        
        # 生成正弦波
        t = torch.linspace(0, duration_sec, num_samples)
        waveform = (torch.sin(2 * math.pi * frequency * t) * 0.3).unsqueeze(0)
        
        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为 WAV - 使用 soundfile 后端或指定格式
        try:
            # 尝试使用 soundfile 后端（更可靠）
            import soundfile as sf
            # torchaudio 返回的是 (C, T) 格式，soundfile 需要 (T,) 或 (C, T) 转置为 (T, C)
            waveform_np = waveform.squeeze(0).numpy()  # 转换为 numpy array，单声道
            sf.write(str(output_path), waveform_np, sample_rate, subtype='PCM_16')
        except ImportError:
            # 如果没有 soundfile，尝试使用 torchaudio 的默认后端
            torchaudio.save(
                str(output_path),
                waveform,
                sample_rate,
                format="wav",
                encoding="PCM_S",
                bits_per_sample=16
            )
        
        file_size = output_path.stat().st_size / 1024
        print(f"✅ 测试音频已创建: {output_path}")
        print(f"   时长: {duration_sec:.2f}秒")
        print(f"   采样率: {sample_rate}Hz")
        print(f"   声道: 单声道")
        print(f"   大小: {file_size:.1f} KB")
        
        return True
        
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("   请安装: poetry run pip install torch torchaudio")
        return False
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_speech_audio(output_path: Path, text: str = "你好，这是一个语音识别测试。"):
    """
    使用 gTTS 生成语音音频（首选方法，更适用于 STT 测试）
    
    参数:
        output_path: 输出文件路径
        text: 要合成的文本
    """
    try:
        from gtts import gTTS
        import tempfile
        import torchaudio
        
        print(f"📝 从文本生成语音: '{text}'")
        
        # 使用 gTTS 生成语音
        tts = gTTS(text=text, lang='zh')  # 中文
        
        # 保存到临时 MP3 文件
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tts.save(str(tmp_path))
        
        print(f"🎵 转换 MP3 到 WAV 16kHz 单声道...")
        
        # 加载 MP3 并转换为 WAV
        waveform, sample_rate = torchaudio.load(str(tmp_path))
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 重采样到 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            waveform = resampler(waveform)
        
        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为 WAV - 使用 soundfile 后端或指定格式
        try:
            # 尝试使用 soundfile 后端（更可靠）
            import soundfile as sf
            waveform_np = waveform.squeeze(0).numpy()  # 转换为 numpy array，单声道
            sf.write(str(output_path), waveform_np, 16000, subtype='PCM_16')
        except ImportError:
            # 如果没有 soundfile，尝试使用 torchaudio 的默认后端
            torchaudio.save(
                str(output_path),
                waveform,
                16000,
                format="wav",
                encoding="PCM_S",
                bits_per_sample=16
            )
        
        # 清理临时文件
        tmp_path.unlink()
        
        duration = waveform.shape[1] / 16000
        file_size = output_path.stat().st_size / 1024
        print(f"✅ 测试音频已创建: {output_path}")
        print(f"   时长: {duration:.2f}秒")
        print(f"   采样率: 16000Hz")
        print(f"   声道: 单声道")
        print(f"   大小: {file_size:.1f} KB")
        
        return True
        
    except ImportError:
        print("⚠️  gTTS 未安装，将使用正弦波备用方案")
        print("   安装: poetry add gtts")
        return False
    except Exception as e:
        print(f"❌ gTTS 生成失败: {e}")
        return False


def main():
    """生成测试音频文件"""
    
    print("\n" + "="*60)
    print("生成 STT 测试音频文件")
    print("="*60 + "\n")
    print(f"输出路径: {OUTPUT_FILE}\n")
    
    # 优先尝试使用语音合成（更适用于 STT 测试）
    test_texts = [
        "你好，这是一个语音识别测试。",
        "Hello, this is a speech recognition test.",
        "你好，请问我可以帮你什么吗？"
    ]
    
    success = False
    for text in test_texts:
        if generate_speech_audio(OUTPUT_FILE, text=text):
            success = True
            break
    
    # 如果语音合成失败，使用正弦波备用方案
    if not success:
        print("\n⚠️  使用正弦波备用方案（对 STT 测试效果较差）")
        if not generate_sine_wave_audio(OUTPUT_FILE, duration_sec=3.0):
            print("\n❌ 所有音频生成方法均失败！")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ 完成！测试音频文件已生成")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

