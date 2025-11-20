#!/usr/bin/env python3
"""
Lokal test için basit bir test script'i
4KAgent sistemini tek bir resim üzerine test etmek için kullanılır
"""

import os
import gc
import sys
import torch
import logging
from pathlib import Path
from argparse import ArgumentParser

# Proje root'unu sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.the4kagent_pipeline import The4KAgent
from utils.logger import get_logger


def setup_logging(output_dir: Path) -> logging.Logger:
    """Test için logging ayarını kur"""
    log_file = output_dir / "test_run.log"
    logger = get_logger(
        logger_name="LocalTest",
        log_file=log_file,
        console_log_level=logging.INFO,
        file_format_str="%(asctime)s - %(levelname)s - %(message)s",
        silent=False
    )
    return logger


def validate_input_image(image_path: Path) -> bool:
    """Resim dosyasının geçerli olup olmadığını kontrol et"""
    if not image_path.exists():
        print(f"❌ Error: Image not found at {image_path}")
        return False
    
    if image_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        print(f"❌ Error: Unsupported image format. Use: png, jpg, jpeg, bmp, webp")
        return False
    
    print(f"✓ Image validated: {image_path.name}")
    return True


def main():
    parser = ArgumentParser(description="4KAgent Local Testing Script")
    parser.add_argument(
        "--image", 
        type=str, 
        default="./assets/profile_test_example/classicsr/test_001.png",
        help="Path to input image for testing"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./outputs/local_test",
        help="Output directory for results"
    )
    parser.add_argument(
        "--profile", 
        type=str, 
        default="LocalTest_P",
        help="Profile name to use (default: LocalTest_P)"
    )
    parser.add_argument(
        "--tool_gpu_id", 
        type=int, 
        default=0,
        help="GPU ID for tool execution"
    )
    parser.add_argument(
        "--perception_gpu_id", 
        type=int, 
        default=None,
        help="GPU ID for perception agent (optional)"
    )
    parser.add_argument(
        "--skip_reflection",
        action="store_true",
        help="Skip reflection step for faster testing"
    )
    parser.add_argument(
        "--skip_retrieval",
        action="store_true",
        help="Skip retrieval step for faster testing"
    )
    
    args = parser.parse_args()
    
    # Paths
    image_path = Path(args.image).resolve()
    output_dir = Path(args.output).resolve()
    
    # Validate input
    if not validate_input_image(image_path):
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    print("\n" + "="*60)
    print("🚀 4KAgent Local Test")
    print("="*60)
    print(f"📸 Input Image: {image_path.name}")
    print(f"📁 Output Dir: {output_dir}")
    print(f"🎯 Profile: {args.profile}")
    print(f"🖥️  Tool GPU ID: {args.tool_gpu_id}")
    if args.perception_gpu_id is not None:
        print(f"🖥️  Perception GPU ID: {args.perception_gpu_id}")
    print(f"⚙️  With Reflection: {not args.skip_reflection}")
    print(f"⚙️  With Retrieval: {not args.skip_retrieval}")
    print("="*60 + "\n")
    
    try:
        # Create 4KAgent instance
        print("📝 Initializing 4KAgent...")
        agent = The4KAgent(
            input_path=image_path,
            output_dir=output_dir,
            with_retrieval=not args.skip_retrieval,
            with_reflection=not args.skip_reflection,
            silent=False,
            tool_run_gpu_id=args.tool_gpu_id,
            perception_agent_run_gpu_id=args.perception_gpu_id,
            profile_name=args.profile
        )
        
        print("✓ Agent initialized successfully\n")
        
        # Run the pipeline
        print("🔄 Starting restoration pipeline...\n")
        agent.run()
        
        print("\n" + "="*60)
        print("✅ Test Completed Successfully!")
        print("="*60)
        print(f"📁 Results saved to: {output_dir}")
        print("\nGenerated files:")
        for item in sorted(output_dir.rglob("*")):
            if item.is_file():
                relative_path = item.relative_to(output_dir)
                size_mb = item.stat().st_size / (1024*1024)
                print(f"   • {relative_path} ({size_mb:.2f} MB)")
        print("="*60 + "\n")
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        return 0
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ Test Failed!")
        print("="*60)
        print(f"Error: {str(e)}")
        print("="*60 + "\n")
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
