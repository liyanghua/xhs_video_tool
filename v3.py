from moviepy.editor import (VideoFileClip, ImageClip, AudioFileClip, 
                          concatenate_videoclips, CompositeAudioClip,
                          CompositeVideoClip)
from gtts import gTTS
import os
from typing import List, Dict, Tuple
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
from PIL import Image

class VideoGenerator:
    def __init__(self):
        """初始化视频生成器"""
        self.base_dir = Path.cwd()
        self.setup_directories()
        self.setup_segments()
        
        # 视频配置
        self.video_size = (1080, 1440)  # 宽 x 高，3:4 比例
        self.video_bitrate = "8000k"    # 高清视频比特率
        
    def setup_directories(self):
        """创建必要的目录结构"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dirs = {
            'debug': self.base_dir / "debug" / timestamp,
            'output': self.base_dir / "output",
            'audio': self.base_dir / "audio" / timestamp,
            'temp': self.base_dir / "temp" / timestamp,
            'logs': self.base_dir / "logs"
        }
        
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            print(f"创建目录 {name}: {path}")
            
        # 设置日志文件
        log_file = self.dirs['logs'] / f"process_{timestamp}.log"
        self.log_file = open(log_file, 'w', encoding='utf-8')
            
    def setup_segments(self):
        """设置片段配置"""
        self.segments = [
            {"name": "intro", "text": "王一博周边定制排行", "start": 0, "target_duration": 3},
            {"name": "bronze", "text": "青铜浴巾毛巾", "start": 3, "target_duration": 3},
            {"name": "silver", "text": "白银马克杯", "start": 6, "target_duration": 3},
            {"name": "gold", "text": "黄金毛巾", "start": 9, "target_duration": 3},
            {"name": "platinum", "text": "铂金带帽卫衣", "start": 12, "target_duration": 3},
            {"name": "diamond", "text": "钻石陶瓷杯具", "start": 15, "target_duration": 3},
            {"name": "king", "text": "王者门帘", "start": 18, "target_duration": 3}
        ]

    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.log_file.write(log_message + "\n")
        self.log_file.flush()

    def debug_clip_info(self, clip, name: str):
        """打印视频/音频片段的调试信息"""
        self.log(f"\n调试信息 - {name}:")
        self.log(f"- 类型: {type(clip)}")
        self.log(f"- 持续时间: {clip.duration:.3f}秒")
        if hasattr(clip, 'size'):
            self.log(f"- 尺寸: {clip.size}")
        if hasattr(clip, 'fps'):
            self.log(f"- FPS: {clip.fps}")
        if hasattr(clip, 'audio') and clip.audio is not None:
            self.log(f"- 包含音频: 是")
            self.log(f"- 音频时长: {clip.audio.duration:.3f}秒")

    def resize_and_pad_video(self, clip: VideoFileClip) -> VideoFileClip:
        """调整视频尺寸并添加填充"""
        self.log(f"\n调整视频尺寸:")
        self.log(f"原始尺寸: {clip.size}")
        
        # 计算目标尺寸
        target_width, target_height = self.video_size
        
        # 计算缩放比例
        width_ratio = target_width / clip.size[0]
        height_ratio = target_height / clip.size[1]
        scale_ratio = max(width_ratio, height_ratio)
        
        # 缩放视频
        scaled_clip = clip.resize(scale_ratio)
        self.log(f"缩放后尺寸: {scaled_clip.size}")
        
        # 如果尺寸不完全匹配，创建黑色背景并将视频居中
        if scaled_clip.size != self.video_size:
            self.log("添加黑色填充以匹配目标尺寸")
            def make_frame(t):
                scaled_frame = scaled_clip.get_frame(t)
                full_frame = np.zeros((target_height, target_width, 3), dtype='uint8')
                
                # 计算居中位置
                y_offset = (target_height - scaled_frame.shape[0]) // 2
                x_offset = (target_width - scaled_frame.shape[1]) // 2
                
                # 将缩放后的帧放入中心位置
                full_frame[y_offset:y_offset+scaled_frame.shape[0], 
                         x_offset:x_offset+scaled_frame.shape[1]] = scaled_frame
                
                return full_frame
            
            final_clip = VideoFileClip(None, make_frame=make_frame)
            final_clip = final_clip.set_duration(clip.duration)
            final_clip = final_clip.set_fps(clip.fps)
            
            if clip.audio is not None:
                final_clip = final_clip.set_audio(clip.audio)
            
            self.log(f"最终尺寸: {final_clip.size}")
            return final_clip
        
        return scaled_clip


    def generate_audio_segments(self) -> List[Tuple[AudioFileClip, float, float]]:
        """生成音频片段并返回实际时长"""
        self.log("\n========== 开始生成音频片段 ==========")
        audio_segments = []
        current_start = 0  # 动态计算开始时间
        
        for segment in self.segments:
            try:
                self.log(f"\n--- 处理音频片段: {segment['name']} ---")
                self.log(f"文本内容: {segment['text']}")
                
                # 生成音频文件
                audio_path = self.dirs['audio'] / f"{segment['name']}_voice.mp3"
                tts = gTTS(text=segment['text'], lang='zh-cn')
                tts.save(str(audio_path))
                
                # 加载音频并获取实际时长
                audio = AudioFileClip(str(audio_path))
                actual_duration = audio.duration
                
                # 设置音频开始时间
                audio = audio.set_start(current_start)
                
                self.log(f"音频信息:")
                self.log(f"- 文件: {audio_path}")
                self.log(f"- 实际时长: {actual_duration:.3f}秒")
                self.log(f"- 开始时间: {current_start:.3f}秒")
                
                # 保存音频信息并更新下一段的开始时间
                audio_segments.append((audio, current_start, actual_duration))
                current_start += actual_duration
                
            except Exception as e:
                self.log(f"错误: 生成音频失败 {segment['name']}: {str(e)}")
                raise
        
        return audio_segments

    def process_video_by_duration(self, video_path: str, start: float, duration: float, 
                                name: str) -> VideoFileClip:
        """处理视频片段"""
        try:
            self.log(f"\n=== 处理视频片段: {name} ===")
            video = VideoFileClip(video_path)
            
            # 调整视频尺寸
            video = self.resize_and_pad_video(video)
            
            # 截取指定时长
            clip = video.subclip(start, start + duration)
            
            # 保存调试视频
            debug_path = self.dirs['debug'] / f"video_{name}_debug.mp4"
            clip.write_videofile(
                str(debug_path),
                codec='libx264',
                audio=False,
                fps=24,
                bitrate=self.video_bitrate
            )
            
            return clip
            
        except Exception as e:
            self.log(f"错误: 处理视频失败: {str(e)}")
            raise

    def create_image_video(self, image_path: str, duration: float, name: str) -> VideoFileClip:
        """创建图片视频"""
        try:
            self.log(f"\n=== 处理图片视频: {name} ===")
            
            # 使用PIL调整图片尺寸
            with Image.open(image_path) as img:
                # 调整图片尺寸以匹配视频尺寸
                img = img.resize(self.video_size, Image.LANCZOS)
                
                # 保存调整后的图片
                temp_image_path = self.dirs['temp'] / f"resized_{name}.jpg"
                img.save(temp_image_path, quality=95)
            
            # 创建视频片段
            image_clip = ImageClip(str(temp_image_path)).set_duration(duration)
            image_clip = image_clip.set_fps(24)
            
            self.log(f"图片视频信息:")
            self.log(f"- 尺寸: {image_clip.size}")
            self.log(f"- 持续时间: {duration:.3f}秒")
            
            return image_clip
            
        except Exception as e:
            self.log(f"错误: 处理图片视频失败: {str(e)}")
            raise

    def process_bgm(self, bgm_path: str, total_duration: float) -> AudioFileClip:
        """处理背景音乐"""
        try:
            self.log(f"\n=== 处理背景音乐 ===")
            self.log(f"BGM文件: {bgm_path}")
            
            # 加载BGM
            bgm = AudioFileClip(bgm_path)
            original_duration = bgm.duration
            self.log(f"原始BGM时长: {original_duration:.3f}秒")
            self.log(f"需要的时长: {total_duration:.3f}秒")
            
            # 如果BGM较短，循环播放
            if original_duration < total_duration:
                self.log("BGM较短，进行循环处理...")
                # 计算需要循环的次数
                repeat_times = int(np.ceil(total_duration / original_duration))
                # 创建循环后的BGM
                bgm = concatenate_audioclips([bgm] * repeat_times)
                
            # 裁剪到所需时长
            bgm = bgm.subclip(0, total_duration)
            
            # 调整音量（可以根据需要调整）
            bgm = bgm.volumex(0.3)  # 降低BGM音量到30%
            
            self.log(f"BGM处理完成，最终时长: {bgm.duration:.3f}秒")
            return bgm
            
        except Exception as e:
            self.log(f"错误: 处理BGM失败: {str(e)}")
            raise
    
    def create_video(self, video_folder: str, image_paths: List[str], bgm_path: str = None) -> str:
        """创建完整视频"""
        try:
            self.log("\n========== 开始生成视频 ==========")
            
            # 1. 生成所有音频片段
            self.log("\n--- 第1步：生成音频片段 ---")
            audio_segments = self.generate_audio_segments()
            
            # 2. 处理视频片段
            self.log("\n--- 第2步：处理视频片段 ---")
            video_clips = []
            
            # 获取视频文件
            video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.MP4'))]
            if not video_files:
                raise Exception("未找到视频文件")
            video_path = os.path.join(video_folder, video_files[0])
            self.log(f"使用视频文件: {video_path}")
            
            # 处理开场视频
            first_audio, first_start, first_duration = audio_segments[0]
            first_clip = self.process_video_by_duration(
                video_path=video_path,
                start=0,
                duration=first_duration,
                name="intro"
            )
            first_clip = first_clip.set_start(first_start)
            video_clips.append(first_clip)
            
            # 处理图片视频
            for i, (audio, start_time, duration) in enumerate(audio_segments[1:], 1):
                if i-1 < len(image_paths):
                    self.log(f"\n处理第 {i} 张图片...")
                    image_clip = self.create_image_video(
                        image_path=image_paths[i-1],
                        duration=duration,
                        name=f"image_{i}"
                    )
                    image_clip = image_clip.set_start(start_time)
                    video_clips.append(image_clip)
            
            # 3. 合并视频片段
            self.log("\n--- 第3步：合并视频片段 ---")
            final_video = CompositeVideoClip(video_clips)
            
            # 4. 合并音频
            self.log("\n--- 第4步：合并音频 ---")
            # 首先合并配音
            voice_audio = CompositeAudioClip([audio for audio, _, _ in audio_segments])
            self.log(f"配音总时长: {voice_audio.duration:.3f}秒")
            
            if bgm_path and os.path.exists(bgm_path):
                # 处理BGM
                bgm_audio = self.process_bgm(bgm_path, voice_audio.duration)
                # 合并配音和BGM
                final_audio = CompositeAudioClip([
                    voice_audio.volumex(1.0),  # 保持配音原音量
                    bgm_audio
                ])
                self.log("已添加BGM")
            else:
                final_audio = voice_audio
                self.log("未找到BGM文件，仅使用配音")
            
            # 5. 添加音频到视频
            self.log("\n--- 第5步：添加音频到视频 ---")
            final_video = final_video.set_audio(final_audio)
            
            # 6. 导出最终视频
            output_path = str(self.dirs['output'] / "final_video.mp4")
            self.log(f"\n--- 第6步：导出最终视频 ---")
            self.log(f"输出路径: {output_path}")
            
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=24,
                bitrate=self.video_bitrate,
                threads=4,
                preset='slow',
                ffmpeg_params=[
                    "-profile:v", "high",
                    "-level", "4.2",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart"
                ]
            )
            
            self.log("\n处理完成!")
            return output_path
            
        except Exception as e:
            self.log(f"错误: 创建视频失败: {str(e)}")
            raise
        
        finally:
            self.log_file.close()

def main():
    try:
        # 创建生成器实例
        generator = VideoGenerator()
        
        # 输入路径
        video_folder = "./input"
        image_paths = [
            "./images/浴巾毛巾.jpg",
            "./images/马克杯.jpg",
            "./images/毛巾.jpg",
            "./images/带帽卫衣.jpg",
            "./images/陶瓷杯具.jpg",
            "./images/门帘.jpg"
        ]
        
        # BGM路径
        bgm_path = "bgm.mp3"  # 替换为你的BGM文件路径
        
        # 生成视频
        output_path = generator.create_video(
            video_folder=video_folder,
            image_paths=image_paths,
            bgm_path=bgm_path
        )
       
        
        print(f"\n视频生成完成: {output_path}")
        print("\n查看日志文件获取详细信息")
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
