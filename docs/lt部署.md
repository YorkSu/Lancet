lancet环境部署

```cmd
conda create -n lt python=3.6.8
conda activate lt
python -m pip install --upgrade pip
conda install tensorflow-gpu=2.0.0
```

将Anaconda3/envs/lt/Library/bin/libiomp5md.dll复制到Anaconda3/envs/lt/

安装pylint

```
conda install pylint
```

musdb部署

```cmd
conda install ffmpeg
pip install musdb
```

安装librosa

```
conda install librosa
```

