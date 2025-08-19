Official PyTorch implementation of "MLDCGAN: A Multimodal Latent Diffusion Conditioned GAN Model for Accelerated and High-Fidelity MRI-CT Synthesis in Radiotherapy Planning "
==
Abstract
==
Magnetic resonance imaging (MRI) offers significant advantages in soft tissue contrast. However, it cannot directly provide electron density information for radiotherapy, relying instead on time-consuming and error-prone MRI-CT image registration. Synthetic CT (sCT) technology, which directly generates CT images from MRI, is pivotal for achieving only MRI-based radiotherapy. However, existing synthesis methods based on generative adversarial network (GAN) and diffusion models face challenges such as prolonged inference times and insufficient utilization of multimodal information, which severely hinder the clinical application of synthetic images. In this study, we propose a novel Multimodal Latent Diffusion Conditioned GAN (MLDCGAN) Model. First, we design a non-parametric non-Gaussian complex denoising distribution based on a conditional GAN, employing a multimodal distribution to achieve large-step denoising. This is combined with a pre-trained autoencoder to compress the image into a low-dimensional latent space, significantly reducing inference time. Second, we fully leverage multimodal MRI information by constructing a local refinement conditional generator with multimodal inputs, including T1-Weighted (T1W), T2-Weighted (T2W), and Mask images. The generator is enhanced by an adaptive weighted multi-sequence fusion module and an enhanced cross-attention module, significantly improving the structural consistency and detail fidelity of the generated sCT images. Finally, by jointly optimizing the style loss and content loss, we ensure the perceptual quality and clinical accuracy of the synthetic images. Experimental results demonstrate that MLDCGAN outperforms existing state-of-the-art methods on both public and private datasets, showing significant improvements in both image quality and inference speed. Subjective evaluations from multiple experienced clinicians indicate that the generated sCT images exhibit no significant difference from real CT in terms of key anatomical structure clarity and overall quality (P > 0.05). Further assessments of clinical target delineation and dose distribution confirm that sCT retains anatomical features well and provides dose distributions consistent with real CT, ensuring the reliability of dose calculations in radiotherapy planning. This study provides a more reliable and efficient technical foundation for achieving only MRI-based radiotherapy. It is expected to assist clinicians in developing more precise radiotherapy plans, ultimately improving treatment outcomes in future clinical practice.

How to run
==
* Train model\
python train.py --data_root ./data --batch_size 8 --num_epochs 200
* Evaluate model\
python evaluate.py --model_path checkpoints/best_model.pth --data_root ./data

* Single reasoning\
python inference.py --model_path checkpoints/best_model.pth --single_inference \
    --t1w_path path/to/T1W.nii --t2w_path path/to/T2W.nii --mask_path path/to/mask.nii
* Batch inference\
python inference.py --model_path checkpoints/best_model.pth --data_root ./data

Data preparation
==
data/\
├── train/\
│   ├── patient_001/\
│   │   ├── T1W.nii\
│   │   ├── T2W.nii\
│   │   ├── mask.nii\
│   │   └── CT.nii\
│   └── patient_002/\
│       └── ...\
└── test/\
    └── ...


Acknowledgments
==
Thanks to Xiao et al for releasing their official implementation of the DDGAN paper.

Contacts
==
If you have any problems, please open an issue in this repository or ping an email to caoning@hhu.edu.cn.
