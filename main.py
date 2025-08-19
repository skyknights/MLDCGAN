# main.py
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='MLDCGAN - Multimodal Latent Diffusion Conditional GAN')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'inference'], 
                       required=True, help='运行模式')
    parser.add_argument('--config_file', type=str, help='配置文件路径')
    parser.add_argument('--model_path', type=str, help='模型路径（用于测试和推理）')
    parser.add_argument('--data_root', type=str, help='数据根目录')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        from train import main as train_main
        print("开始训练MLDCGAN模型...")
        train_main()
        
    elif args.mode == 'test':
        from evaluate import main as eval_main
        print("开始评估MLDCGAN模型...")
        eval_main()
        
    elif args.mode == 'inference':
        from inference import main as inference_main
        print("开始MLDCGAN推理...")
        inference_main()
    
    print(f"MLDCGAN {args.mode}模式完成!")

if __name__ == '__main__':
    main()
