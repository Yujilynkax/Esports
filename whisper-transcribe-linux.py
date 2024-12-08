import os
import whisper
from tqdm import tqdm
import logging
import csv

def transcribe_folder(input_folder, output_folder, model_size="large-v2"):
    """转写文件夹中的所有音频文件"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 加载模型
    print(f"正在加载 {model_size} 模型...")
    model = whisper.load_model(model_size)
    print("模型加载成功")

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有MP3文件
    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mp3')]
    total_files = len(audio_files)
    
    print(f"找到 {total_files} 个MP3文件")
    
    # 创建进度条
    for i, filename in enumerate(audio_files, 1):
        try:
            print(f"\n[{i}/{total_files}] 开始处理: {filename}")
            
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0])
            
            # 转写音频
            result = model.transcribe(
                input_path,
                language="Chinese",
                task="transcribe",
                verbose=True  # 显示详细进度
            )
            
            # 保存CSV格式（包含时间戳和文本）
            with open(f"{output_path}.csv", "w", encoding="utf-8", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["序号", "开始时间", "结束时间", "文本内容"])  # 写入表头
                
                for idx, segment in enumerate(result["segments"], start=1):
                    writer.writerow([
                        idx,
                        format_timestamp(segment['start']),
                        format_timestamp(segment['end']),
                        segment['text'].strip()
                    ])
            
            # 保存SRT格式
            with open(f"{output_path}.srt", "w", encoding="utf-8") as srt_file:
                for i, segment in enumerate(result["segments"], start=1):
                    srt_file.write(f"{i}\n")
                    srt_file.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
                    srt_file.write(f"{segment['text'].strip()}\n\n")
            
            print(f"完成: {filename}")
            
        except Exception as e:
            print(f"处理文件失败 {filename}: {str(e)}")
            continue

def format_timestamp(seconds):
    """将秒数转换为SRT时间戳格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")

if __name__ == "__main__":
    input_folder = "/home/yujilynkax/Esports/videos"  # 输入视频文件夹
    output_folder = "/home/yujilynkax/Esports/text"   # 输出文本文件夹
    
    transcribe_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        model_size="large-v2"  # 使用large-v2模型以提高准确性
    )
