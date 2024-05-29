
import os
import gdown
import py3_wget


def download_models():
    os.makedirs('weights', exist_ok=True)
    gdown.download(
        "https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view",
        output="weights/droid.pth",
        fuzzy=True
    )
    
    py3_wget.download_file(
        "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth?download=true",
        output_path="weights/metric_depth_vit_giant2_800k.pth"
    )


if __name__ == "__main__":
    download_models()
