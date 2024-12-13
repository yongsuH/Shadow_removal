### Work is Based On
- [Instance-Shadow-Diffusion](https://github.com/MKFMIKU/Instance-Shadow-Diffusion)  
- [SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch/)

---

### How to Run

1. **Download Diffusion Model Checkpoint**  
   Download the diffusion model checkpoint from (https://drive.google.com/file/d/18xXVNpAZ9rhAd7K1iTcSSDerg0soDC-H/view?usp=drive_link) and place it under:  
   `./checkpoints/removal`

2. **Download SAM-Adapter Checkpoint and Config**  
   Download the SAM-adapter checkpoint and configuration file from [provide link here] and place them under:  
   `./checkpoints/sam`

3. **Create conda environment**  
   Work is only tested on torch 2.4.1+cu124.
   Run the following command to create the Conda environment:
```bash
conda env create -f envYML/environment.yaml
```
4.  **start webUI**
```bash
python demoWeb.py
```
