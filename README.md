## EasyOOTD: Pose-Controllable Virtual Try-On via Diffusion Adapter
<a href="README.md">English</a> | <a href="README_CN.md">中文</a>

### Features of EasyOOTD
* Implemented virtual try-on with a new Adapter based on the SD1.5 model, only 198 MB in size, compatible with different SD1.5 base models.
* Seamlessly integrates with pre-trained ControlNet, enabling pose-controllable virtual try-on.
* Trained an LCM-Lora for this Adapter, only 76 MB in size, capable of inference in 4 steps.
* Designed an interactive image segmentation webui based on [SAM-HQ](https://github.com/SysCV/sam-hq), allowing users to easily specify the replacement area for virtual try-on.

**Note: This project was completed in spare time as a hobby. As it only adds one Adapter, the model's capabilities have limitations. If you apply it directly to products, you do so at your own risk.**

**If you find this project useful, please give it a star✨✨**

### Demo
<video src="https://github.com/user-attachments/assets/082ec10c-aa6a-4931-8229-ade11f3a8e2e" controls="controls" width="500" height="300">Your browser does not support playing this video!</video>

### Model Structure
<img src="assets/introductions/model.jpg" alt="Model Structure" width="768" height="500">

### How to Use
* Environment Setup
  * Install Python environment, you can use [conda](https://github.com/conda-forge/miniforge): `conda create -n easy_ootd python=3.10`, then `conda activate easy_ootd`
  * `pip install -e .`
  * If you have a GPU:
    * `conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia`
    * `pip install onnxruntime-gpu`
    * `pip install -r requirements.txt`
  * If using CPU:
    * `conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 cpuonly -c pytorch`
    * `pip install onnxruntime`
    * `pip install -r requirements.txt`
* Model Download: `bash scripts/download_models.sh`
* Run: `python app.py`, open webpage: `http://127.0.0.1:7865/`
* Tutorial:
<video src="https://github.com/user-attachments/assets/fe472102-7b0e-4f35-9292-b442a203ff09" controls="controls" width="500" height="300">Your browser does not support playing this video!</video>

### About Me
Feel free to follow my video channel, where I will continuously share AIGC content I create. For collaboration inquiries, please send a private message.

<img src="assets/introductions/shipinhao.jpg" alt="Video Channel" width="300" height="350">