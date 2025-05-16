import subprocess
import sys

def run_command(command):
    try:
        # 执行命令并实时打印输出
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=sys.stdout,  # 输出到当前终端
            stderr=sys.stderr
        )
        print(f"命令执行成功: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")

if __name__ == "__main__":
    # 定义两个需要执行的命令（按顺序执行）
    commands = [
        [
            "python", "detect.py",
            "--source", "data/images/test.jpg",
            "--img", "640",
            "--device", "cpu",
            "--data", "data/coco.yaml",
            "--weights", "gelan-c-converted.pt",
            "--name", "car_detect",
            "--view-img",
            "--save-txt",
            "--save-conf",
            "--save-crop",
            "--exist-ok"
        ],
        [
            "python", "detect.py",
            "--source", "data/images/psd.1.jpg",
            "--img", "640",
            "--device", "cpu",
            "--data", "data/psd.yaml",
            "--weights", "psd-converted.pt",
            "--name", "line_detect",
            "--view-img",
            "--save-txt",
            "--save-conf",
            "--save-crop",
            "--exist-ok"
        ]
    ]

    # 依次执行命令
    for cmd in commands:
        run_command(cmd)