### Work is Based On
- [Instance-Shadow-Diffusion](https://github.com/MKFMIKU/Instance-Shadow-Diffusion)  
- [SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch/)

---

### How to Run

1. **Download Diffusion Model Checkpoint**  
   Download the diffusion model checkpoint from [provide link here] and place it under:  
   `./checkpoints/removal`

2. **Download SAM-Adapter Checkpoint and Config**  
   Download the SAM-adapter checkpoint and configuration file from [provide link here] and place them under:  
   `./checkpoints/sam`

---

### Create Conda Environment
Run the following command to create the Conda environment:
```bash
conda env create -f envYML/environment.yaml
