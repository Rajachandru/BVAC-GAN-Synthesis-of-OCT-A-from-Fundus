**Conditional Generative Adversarial Network (C-GAN) for Diabetic Retinopathy (DR) diagnosis**
![WhatsApp Image 2025-05-05 at 5 28 04 PM](https://github.com/user-attachments/assets/83f676f2-4166-45d1-a3cf-327a8a0f6cfb)
![WhatsApp Image 2025-05-05 at 5 28 07 PM (2)](https://github.com/user-attachments/assets/2e45dd15-12f0-48be-9a1c-d9bb885ff9ee)
                                
                                **THE APP TESTED AND DEPUTED IN HOSPITAL**
                                
![APP Screenshot](https://github.com/user-attachments/assets/28ee70b0-40fb-47c7-8c6f-d5ed49bcf8b5)

                                
                                **Screen Shot of the User friendly APP**



**File Descriptions**
Fundus_OCTA_cGAN_v1. ipynb  -  Python code for BVAC GAN
app.py                      -  Flask based Application with user-friendly infterface for Doctors to upload fundus and seek OCTA equivalent.
The Dataset is available in [1] and a similar research work to generate synthetic OCTA images using dataset [1] is available in [2].

**Proposed Methodology:**
The fundus imaging has no depth information of retinal vasculature and OCTA has no direct visualization of the Optic Disc, macula. Both the modalities hold complementary feature maps with respect to DR diagnosis. We propose a first of the kind model, combining the two modalities i. (original) fundus images ii. (GAN synthesized) OCT-A images to diagnose DR. We feed the fundus images and BVAC synthesized OCT-A images to 2 ResNet 50 CNNs with a merged Fully Connected layer for DR diagnosis.

The following steps are executed to generate synthetic OCT-A image pair from Fundus image and diagnose DR.
1. Pre-processing steps
2. Pipeline to create dataset for train, test of Fundus and OCT-A images
3. Generator: Improved U-Net with Squeeze & Excitation (Entropy based) block
Encoder stage of U-Net: customized threshold-based Squeeze and excitaion (SE) block 
4. Loss: Generator loss, discriminator loss
5. Training the model
7. Develop a Web based application (APP) to predict OCT-A pair from Fundus image

   The below fig shows a fundus image sample, OCTA synthesized by conventional GAN, our BVAC GAN and the ground truth OCT-A
![image](https://github.com/user-attachments/assets/946a4a99-937b-449c-ace7-4c8d172f2cfa)

![WhatsApp Image 2025-05-09 at 2 26 27 PM (1)](https://github.com/user-attachments/assets/45058898-d3f1-49ac-b7c5-3b5499269d8b)

**DR Diagnosis**

![WhatsApp Image 2025-05-09 at 2 26 26 PM](https://github.com/user-attachments/assets/35c36d6f-771e-4003-b908-4aeff1e5d1f1)

**Entropy based Squeeze & Excitation**

Reference:
1.	https://zenodo.org/records/6476639 [1]
2.	Coronado I, Pachade S, Trucco E, Abdelkhaleq R, Yan J, Salazar-Marioni S, Jagolino-Cole A, Bahrainian M, Channa R, Sheth SA, Giancardo L. Synthetic OCT-A blood vessel maps using fundus images and generative adversarial networks. Sci Rep 2023;13:15325. https://doi.org/10.1038/s41598-023-42062-9.
