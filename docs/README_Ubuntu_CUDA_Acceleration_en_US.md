# Ubuntu 22.04 LTS

### 1. Check if NVIDIA Drivers Are Installed

```sh
nvidia-smi
```

If you see information similar to the following, it means that the NVIDIA drivers are already installed, and you can skip Step 2.

> [!NOTE]
> Notice:`CUDA Version` should be >= 12.4, If the displayed version number is less than 12.4, please upgrade the driver.

```plaintext
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.07             Driver Version: 572.83         CUDA Version: 12.8   |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3060 Ti   WDDM  | 00000000:01:00.0  On |                  N/A |
|  0%   51C    P8              12W / 200W |   1489MiB /  8192MiB |      5%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

### 2. Install the Driver

If no driver is installed, use the following command:

```sh
sudo apt-get update
sudo apt-get install nvidia-driver-570-server
```

Install the proprietary driver and restart your computer after installation.

```sh
reboot
```

### 3. Install Anaconda

If Anaconda is already installed, skip this step.

```sh
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh
```

In the final step, enter `yes`, close the terminal, and reopen it.

### 4. Create an Environment Using Conda

```bash
conda create -n mineru 'python=3.12' -y
conda activate mineru
```

### 5. Install Applications

```sh
pip install -U magic-pdf[full]
```
> [!TIP]
> After installation, you can check the version of `magic-pdf` using the following command:
>
> ```sh
> magic-pdf --version
> ```


### 6. Download Models


Refer to detailed instructions on [how to download model files](how_to_download_models_en.md).


## 7. Understand the Location of the Configuration File

After completing the [6. Download Models](#6-download-models) step, the script will automatically generate a `magic-pdf.json` file in the user directory and configure the default model path.
You can find the `magic-pdf.json` file in your user directory.

> [!TIP]
> The user directory for Linux is "/home/username".


### 8. First Run

Download a sample file from the repository and test it.

```sh
wget https://github.com/opendatalab/MinerU/raw/master/demo/pdfs/small_ocr.pdf
magic-pdf -p small_ocr.pdf -o ./output
```

### 9. Test CUDA Acceleration

If your graphics card has at least **6GB** of VRAM, follow these steps to test CUDA acceleration:

1. Modify the value of `"device-mode"` in the `magic-pdf.json` configuration file located in your home directory.
   ```json
   {
     "device-mode": "cuda"
   }
   ```
2. Test CUDA acceleration with the following command:
   ```sh
   magic-pdf -p small_ocr.pdf -o ./output
   ```