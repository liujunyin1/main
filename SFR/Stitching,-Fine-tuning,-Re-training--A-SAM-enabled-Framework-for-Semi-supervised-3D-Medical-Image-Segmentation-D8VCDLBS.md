# Stitching, Fine-tuning, Re-training: A SAM-enabled Framework for Semi-supervised 3D Medical Image Segmentation

Shumeng Li, Lei Qi, Qian Yu, Jing Huo, Yinghuan Shi\*, Yang Gao

Abstract—Segment Anything Model (SAM) fine-tuning has shown remarkable performance in medical image segmentation in a fully supervised manner, but requires precise annotations. To reduce the annotation cost and maintain satisfactory performance, in this work, we leverage the capabilities of SAM for establishing semi-supervised medical image segmentation models. Rethinking the requirements of effectiveness, efficiency, and compatibility, we propose a three-stage framework, i.e., Stitching, Fine-tuning, and Re-training (SFR). The current fine-tuning approaches mostly involve 2D slice-wise fine-tuning that disregards the contextual information between adjacent slices. Our stitching strategy mitigates the mismatch between natural and 3D medical images. The stitched images are then used for fine-tuning SAM, providing robust initialization of pseudo-labels. Afterwards, we train a 3D semi-supervised segmentation model while maintaining the same parameter size as the conventional segmenter such as V-Net. Our SFR framework is plug-and-play, and easily compatible with various popular semi-supervised methods. We also develop an extended framework $\mathrm{SFR}^{+}$ with selective fine-tuning and re-training through confidence estimation. Extensive experiments validate that our SFR and $\mathrm{SFR}^{+}$ achieve significant improvements in both moderate annotation and scarce annotation across five datasets. In particular, SFR framework improves the Dice score of Mean Teacher from $29.68\%$ to $74.40\%$ with only one labeled data of LA dataset. The code is available at <https://github.com/ShumengLI/SFR>.

Index Terms-3D medical image segmentation, semi-supervised learning, stitching, fine-tuning, re-training, SAM-enabled.

# I. INTRODUCTION

RECENTLY, general foundation models for visual segmentation \[1]–\[4] have attracted widespread attention in the field of medical images owing to their excellent segmentation and generalization capabilities. Although these foundational models have made remarkable progress in medical image analysis, it is sometimes challenging to utilize a unified model to segment all medical images due to the inevitable factors, e.g., specific modalities, complex imaging techniques, and variable tissues. To tackle this issue, several recent works have been proposed to either focus on prompt engineering \[5]–\[7] or design adapters for fine-tuning \[8]–\[14] to borrow the ability of foundation model, e.g., MSA \[9] and SAMed \[10] derived from SAM \[3], to their specific tasks.

This work was supported by the NSFC Project (62222604, 62206052), China Postdoctoral Science Foundation (2024M750424), Fundamental Research Funds for the Central Universities (020214380120, 020214380128), State Key Laboratory Fund (ZZKT2024A14), the Postdoctoral Fellowship Program of CPSF (GZC20240252), Jiangsu Funding Program for Excellent Postdoctoral Talent (2024ZB242), and Jiangsu Science and Technology Major Project (BG2024031).

Shumeng Li, Jing Huo, Yinghuan Shi and Yang Gao are with the State Key Laboratory for Novel Software Technology, Nanjing University, China. They are also with National Institute of Healthcare Data Science, Nanjing University, China. (E-mail: <lism@smail.nju.edu.cn>, <huojing@nju.edu.cn>, <syh@nju.edu.cn>, <gaoy@nju.edu.cn>)

Lei Qi is with the School of Computer Science and Engineering, and the Key Lab of Computer Network and Information Integration (Ministry of Education), Southeast University, China. (E-mail: <qilei@seu.edu.cn>)

Qian Yu is with the School of Data and Computer Science, Shandong Women's University, China. (E-mail: <yuqian@sdwu.edu.cn>)

The corresponding author of this work is Yinghuan Shi.

![\<img alt="" data-attachment-key="8LDM9Q26" width="639" height="475" src="attachments/8LDM9Q26.jpg" ztype="zimage"> | 639](attachments/8LDM9Q26.jpg)\
Fig. 1. Comparison of our SFR framework with extended foundation models and semi-supervised medical segmentation methods on LA dataset \[15] with 16 labeled data.

We notice that most of these works \[5], \[8]–\[11], \[13], \[14] are fully supervised methods with fine-tuning or adaptation techniques employed. However, fully supervised medical image segmentation relies on a great amount of precise annotations delineated by experienced experts, which makes the labeling process tedious, time-consuming, and even subjective.

Recent trends \[16]–\[18] have shown that in some cases, the performance of semi-supervised methods is almost comparable to that of fully supervised methods. For example, with $40\%$ annotation, \[16] outperforms the fully supervised approach on BTCV dataset. And the performance of \[17] on LA dataset with $20\%$ labeled data is only $0.8\%$ lower than the fully supervised result. Therefore, we wonder, whether the current success of the foundation model could drive us to develop an effective and efficient model for semi-supervised medical image segmentation?

With the aforementioned goal, we wish to revisit several important factors before designing our framework for semi-supervised medical image segmentation.

How to initialize effectively? According to previous studies \[16]–\[25], in the semi-supervised scenario, the quality of pseudo-labels in initialization stages plays an important role in the following segmentation. Unlike natural images, inter-slice continuity of 3D medical images is crucial for accurate target segmentation. Also, medical images usually have relatively low resolution. The existing strategies \[8], \[10]–\[12], \[26] of fitting medical images, including directly enlarging 2D slices \[8] and resizing the positional embeddings \[10], \[11], use slice by slice fine-tuning and disregard the inherent interslice correlation that exists in 3D images. Therefore, is there a better way to boost the quality of initial pseudo-labels for medical images using a foundation model?

How to improve efficiency? Foundation model is pretrained on a large-scale dataset whose parameter size is relatively large. Existing fine-tuning methods \[5]–\[9], \[12]–\[14] still preserve the original parameter size as foundation model during inference or even introduce additional parameters. During segmenting medical images, do we really need such a large parameter size? Firstly, the existing model only involving a small-size parameter indeed performs well in segmenting organs \[18], \[25]. Secondly, we notice that very recent works \[27] depict the redundancy issue in the foundation model, revealing the large-scale pre-trained models are over-parameterized. Thirdly, the appearance of medical images often has standardized views and relatively limited texture variants compared with natural images \[28]. Regarding these issues, is it possible to escape from over-parameterized foundation model while maintaining promising results?

How to preserve compatibility? On one hand, in recent years, semi-supervised learning has emerged as an appealing strategy and been widely applied to medical image segmentation tasks \[29], and a lot of semi-supervised medical image segmentation methods have been proposed. Could the foundation models be made to better serve existing semi-supervised methods? On the other hand, the research progress in the field of computer vision and machine learning about semi-supervised learning still helps evolve new semi-supervised medical image segmentation models. Will our framework still be compatible with these new methods in the future?

Being aware of these observations, we believe, that in the era of foundation model, a promising semi-supervised medical image segmentation framework should be performance effectiveness, parameter efficiency, and excellent compatableness. Thus, we propose a straightforward framework of Sti tching, Fine-tuning, and Re-training (SFR) to accomplish our above goals. We first develop a stitching strategy, performing a stitching operation on slices to produce an image that matches the high-resolution input, which better exploits inter-slice relationships and dimensional information. The stitched images are then fed into the SAM for fine-tuning. Afterwards, we train a small-scale 3D segmentation model with the guidance of SAM while maintaining the same parameter size. The fine-tuned SAM provides a favorable initialization and is compatible with various 3D models. In addition, we develop the extended framework $\mathrm{SFR}^+$ , introducing confidence estimation and selective training strategy to enhance the utilization of unlabeled data. We conduct the semi-supervised scenarios of moderate annotation $^{1}$ and scarce annotation $^{2}$ on five datasets. The experiments demonstrate our SFR and $\mathrm{SFR}^{+}$ frameworks achieve extremely close performance to full supervision with moderate annotation and exhibit remarkable improvement with scarce annotation. As shown in Fig. 1, our method in fine-tuning stage outperforms other extended foundation models and achieves a large improvement in re-training stage.

Our contribution could be summarized as follows:

<span style="color: rgb(0, 0, 0)"><span style="background-color: rgb(255, 255, 255)">我们的贡献可以总结如下：</span></span>

*   We propose a novel framework leveraging the ability of the foundation model while ensuring performance under semi-supervised segmentation and further reducing labeling costs, which involves three stages, i.e., stitching, fine-tuning, and re-training.
*   Our stitching strategy is simple yet effective in pseudolabeling initialization, and largely distinct from current resize/directly fine-tuning strategy.
*   Our parameter size during inference maintains the same level as the mainstream segmenter, e.g., V-Net \[30], which is greatly smaller than that of foundation model.
*   Our framework is plug-and-play, which could be easily married to most existing popular semi-supervised segmentation methods.

# II. RELATED WORK

# A. Foundation Models in Medical Images

1.  Visual Foundation Models: Nowadays, visual foundation models have gained significant attention and have shown impressive performance in various computer vision tasks including segmentation. Prominent examples of these models include SAM \[3], SegGPT \[1], SEEM \[2], SLiMe \[4], and SAM 2 \[31], along with their extended applications \[32]–\[34]. These models leverage large-scale image datasets to learn universal visual representations and demonstrate remarkable generalization ability.

In the field of medical images, UniverSeg \[35] achieves universal segmentation for 2D medical images by providing an example set of image-label pairs. STU-Net \[36] is a foundation model specializing in CT modalities, with its largest variant consisting of 1.4 billion parameters. Furthermore, SAM \[3] has emerged as one of the most prevailing models for image segmentation, and many works extend it to medical images. SAM-Med2D \[37] is a 2D model that fine-tunes SAM on 4.6 million medical images. SAM-Med3D \[38] adopts a SAM-like architecture, but it is trained from scratch without utilizing the pre-trained weights of SAM. Due to SAM's impressive performance and broad applicability, it serves as the default foundation model in our proposed framework.

2.  Adapt SAM to 3D Medical Images: SAM's zero-shot capability is insufficient to ensure direct application in medical images \[39], \[40]. To extend the powerful segmentation ability to medical images, many works are devoted to fine-tuning with different image processing and fine-tuning strategies.

For image processing, the disparity in image resolution between 3D medical images and pre-trained natural images poses a challenge, and two strategies have been proposed to tackle this issue. The first is upsampling fine-tuning \[8], which involves upsampling each slice to match the input resolution directly. The second is small-size fine-tuning \[10], \[11], which reduces the input size through bilinear interpolation. However, both of these strategies are based on 2D inputs, and for 3D medical images, the predictions need to be generated by segmenting each slice. 3DSAM-Adapter \[41] and SAM-Med3D \[38] extend SAM to 3D architecture, whereas they increase additional training overhead with a large model size. In contrast, our stitching strategy is designed to accommodate variations in image dimension and resolution, creating large-sized stitched images that capture spatial information across adjacent slices effectively.

The fine-tuning approaches include fine-tuning only subparts parameters and incorporating adapters. MedSAM \[8] fine-tunes the mask decoder of SAM and freezes the encoders, whereas the performance shows a lag behind medical-specific models, particularly in terms of boundary area. MSA \[9] and SAMed \[10] adopt the parameter-efficient fine-tuning techniques, using adapter and low-rank-based strategy (LoRA) \[42] strategies for fine-tuning.

1Moderate refers to a commonly used level of annotation \[16], \[17].\
2Scarce means very few annotations such as 1 labeled data.

![\<img alt="" data-attachment-key="ALSA8VA5" width="1405" height="514" src="attachments/ALSA8VA5.jpg" ztype="zimage"> | 1405](attachments/ALSA8VA5.jpg)\
Fig. 2. Overview of the proposed SFR framework, which includes three modules: Sticking, Fine-tuning and Re-training.

# B. Semi-supervised Medical Image Segmentation

Since the pixel-wise annotations require tremendous delineation time, semi-supervised learning (SSL) for segmentation aims to reduce the annotation burden by leveraging a large number of unlabeled samples along with a limited number of labeled samples \[16]–\[18], \[20]–\[25]. Semi-supervised segmentation methods mainly include pseudo-labeling and consistency regularization. Self-training \[43] gradually generates pseudo-labels for unlabeled data through an iterative process to be jointly trained with labeled data. Mean Teacher \[21] is a classic consistency regularization-based method that effectively reduces over-adaptation. Xu et al. \[17] improves the classical MT model to an ambiguity-consensus mean teacher model, and Chen et al. \[16] develops a data augmentation strategy based on partition-and-recovery $N^3$ cubes. Existing SSL methods have demonstrated promising results with moderate annotation levels. On one hand, our framework ensures seamless integration with various SSL models. On the other hand, we also apply the framework to challenging scenarios, further reducing the annotation requirements.

Some of the latest works \[7], \[26] explore incorporating SAM into semi-supervised training. For example, ASLseg \[26] couples SAM with a specific semi-supervised model to refine pseudo-labels, while CPC-SAM \[7] uses SAM prompts for cross-branch teaching. However, these methods still rely on slice-based processing and preserve a high computational load during inference. Our method could help improve performance while reducing inference costs. It is worth mentioning that our SFR and $\mathrm{SFR}^{+}$ frameworks are compatible with various semi-supervised segmentation methods for medical images.

Remark. Our framework explores the potential of leveraging SAM's capabilities in SSL medical image segmentation models, improving accuracy while reducing annotation costs. It is worth mentioning that our framework is compatible with most SSL methods for medical images.

# III. METHOD

# A. Notations and Framework Overview

Formally, we now provide our notation used in this paper. Given $m$ labeled images and $n$ unlabeled images, the $i$ -th $(1 \leq i \leq m)$ labeled image and its ground truth are denoted as $\mathbf{X}_i^l$ and $\mathbf{Y}_i^l$ . The $j$ -th $(1 \leq j \leq n)$ unlabeled image is denoted as $\mathbf{X}_j^u$ . Here, $\mathbf{X}_i^l$ , $\mathbf{X}_j^u \in \mathbb{R}^{H \times W \times D}$ , $\mathbf{Y}_i^l \in \{0, 1, \dots, K - 1\}^{H \times W \times D}$ , where $H$ , $W$ , $D$ indicate the corresponding dimensionality of 3D medical images and the input patch size for training is square, i.e., $H = W$ . $K$ is the number of different classes to segment.

As illustrated in Fig. 2, we build our SFR framework, which consists of the following three modules: the Stitching Module, the Fine-tuning Module, and the Re-training Module. The stitching module mitigates the mismatch between natural and 3D medical images. The stitched images are input into SAM to fine-tune and provide initial pseudo labels for the semi-supervised module. Afterwards, the 3D semi-supervised segmentation model is trained. Furthermore, we enhance our framework by proposing $\mathrm{SFR}^+$ , which selectively fine-tunes and re-trains through confidence estimation.

![\<img alt="" data-attachment-key="B96KJNNH" width="669" height="436" src="attachments/B96KJNNH.jpg" ztype="zimage"> | 669](attachments/B96KJNNH.jpg)\
Fig. 3. Comparison of different input strategies. Small-size fine-tuning reduces the input size through bilinear interpolation and upsampling fine-tuning directly upsamples each slice. Taking $d = 4$ as an example.

Step 1: Stitching Module. As aforementioned, by reducing the resolution difference between pre-trained samples (i.e., natural image) and fine-tuned samples (i.e., medical image), the stitching module could transform a 3D labeled volume $\mathbf{X}_i^l$ to a large-sized 2D image $\mathbf{M}_i^l \in \mathbb{R}^{Hd \times Wd}$ with a slice stitching function $\mathsf{F}_{\mathbb{C}}(\cdot)$ . Also, the stitched ground truth $\mathbf{N}_i^l$ is obtained similarly. $\mathbf{M}_i^l$ and $\mathbf{N}_i^l$ are arranged in a $d \times d$ grid with $d = \lceil \sqrt{D} \rceil$ , and we stitch zeros after all slices if $d \times d > D$ .

$$
\mathbf {M} _ {i} ^ {l} = \mathrm {F} _ {\mathrm {C}} \left(\mathbf {X} _ {i} ^ {l}\right), \quad \mathbf {N} _ {i} ^ {l} = \mathrm {F} _ {\mathrm {C}} \left(\mathbf {Y} _ {i} ^ {l}\right). \tag {1}
$$

Step 2: Fine-tuning Module. We first utilize the stitched labeled images $\mathbf{M}_i^l$ along with their ground truth $\mathbf{N}_i^l$ to fine-tune a SAM parameterized by $\theta$ via popularly used strategy, e.g., LoRA \[10], \[42]. This aims to narrow the possible distribution shift between natural and medical images. Taking LoRA as an example, we denote the fine-tune function as $\mathbb{F}_{\mathrm{LORA}}(\cdot)$ , which is parameterized by $\theta$ with input as $\mathbf{M}_i^l$ and $\mathbf{N}_i^l$ . The updated SAM with optimal parameter $\theta^{*}$ is obtained as follows:

$$
\theta^ {*} = \arg \min  _ {\theta} \sum_ {i = 1} ^ {m} \mathrm {F} _ {\mathrm {L O R A}} \left(\mathbf {M} _ {i} ^ {l}, \mathbf {N} _ {i} ^ {l}; \theta\right). \tag {2}
$$

Then, we produce high-quality pseudo-labels for unlabeled images by using the prediction function $\mathsf{F}_{\mathsf{FT}}(\cdot)$ of fine-tuned SAM, and generate 3D pseudo-labels by a stitching inverse transform $\mathsf{F}_{\mathsf{C}}^{-1}(\cdot)$ as follows:

$$
\hat {\mathbf {Y}} _ {j} ^ {u} = \mathrm {F} _ {\mathrm {C}} ^ {- 1} \left(\mathrm {F} _ {\mathrm {F T}} \left(\mathrm {F} _ {\mathrm {C}} \left(\mathbf {X} _ {j} ^ {u}\right); \theta^ {*}\right)\right). \tag {3}
$$

![\<img alt="" data-attachment-key="45JM6FWY" width="670" height="353" src="attachments/45JM6FWY.jpg" ztype="zimage"> | 670](attachments/45JM6FWY.jpg)\
Fig. 4. Disrupting slice continuity. Comparison with random slice rotation and flipping, and random shuffling. $d = 3$ as an example.

Step 3: Re-training SSL Module. The recent SSL network, such as the popular self-training, mean teacher, and the most advanced ACMT, MagicNet methods, could be the alternatives in this module. Specifically, the SSL network learns pseudo-labels from fine-tuned SAM. We denote the SSL network as $\mathbb{F}_S(\cdot)$ parameterized by $\omega$ and the re-training module with optimal $\omega^{*}$ is as follows:

$$
\omega^ {*} = \arg \min  _ {\omega} \left(\sum_ {i = 1} ^ {m} \mathrm {F} _ {\mathrm {S}} \left(\mathbf {X} _ {i} ^ {l}, \mathbf {Y} _ {i} ^ {l}; \omega\right) + \lambda \sum_ {j = 1} ^ {n} \mathrm {F} _ {\mathrm {S}} \left(\mathbf {X} _ {j} ^ {u}, \hat {\mathbf {Y}} _ {j} ^ {u}; \omega\right)\right), \tag {4}
$$

where $\lambda$ acts as a tradeoff between two terms.

# B. SFR Framework

1.  Stitching Module: To adapt from 2D natural images to 3D medical images, we recognize that the input resolution and image dimension are crucial factors. The inter-slice spatial information of 3D volumes is relevant for target recognition, and it is difficult for the large model trained at high-resolution images to generalize to low-resolution medical image slices. Inspired by this observation, our stitching strategy, illustrated in Fig. 3, matches medical images to natural image resolution, and supplements the spatial arrangement specific to 3D medical images. The input spatial resolution of the pre-trained SAM model is  $1024 \times 1024$  . Our stitching strategy arranges the 3D volume (either a raw 3D image or a 3D patch) slice by slice into a  $d \times d$  grid, producing a 2D image of size  $1024 \times 1024$  . Regarding the variability in slice sizes across different medical datasets, there is a performance trade-off between the slice resolution and the number of stitched slices. Our method could effectively manage images with different numbers of slices. For the small-scale slices, such as the LA \[15] dataset, we use a raw 3D image as an input volume. For large-scale slices, such as the BTCV \[44] dataset, we follow the common 3D processing way where dividing the volume into patches as an input volume, and then stitch all slices to achieve the final size of  $1024 \times 1024$  , avoiding downsizing a large slice directly. Compared to the small-size input fine-tuning method \[10], \[11] and the direct upsampling fine-tuning methods \[8] in Fig. 3, we find our stitching strategy effectively addresses the challenges of image dimension and resolution differences.

We thoroughly investigate the stitching strategy from slice continuity and contextual integrity.

Slice Continuity. Due to the inherent spatial continuity of 3D medical images, we explore the relationship between slice stitching and slice order. For a stitched 2D image, the segmentation model learns the feature correlation across slices through the self-attention mechanism, so it struggles to capture contextual information and coherence of shape without the slice order. To investigate the importance of slice continuity, we disrupt the continuity in two ways: 1) Randomly shuffling the order of the slices; 2) Randomly rotate and flip each individual slice. The same slicing operation is also applied to the ground truth masks to maintain consistency between the input data and labels. As illustrated in Fig. 4, it has been observed that disrupting the slice order leads to the loss of shape coherence and a decrease in performance, and the results emphasize the importance of slice continuity in 3D organ segmentation.

![\<img alt="" data-attachment-key="RNTITUZZ" width="703" height="233" src="attachments/RNTITUZZ.jpg" ztype="zimage"> | 703](attachments/RNTITUZZ.jpg)\
Fig. 5. Disrupting contextual integrity. Comparison with stitching with natural images. $d = 3$ as an example.

Contextual Integrity. Our stitching module reorganizes a volume (3D raw image or 3D patch) into a $1024 \times 1024$ image, enabling a complete representation of the volume within a single image. To explore the impact of contextual integrity on stitching, we stitch the medical slices with natural images while keeping the resolution of each slice constant. As shown in Fig. 5, we transition from natural images to medical images by progressively increasing the number of medical slices. Specifically, at the beginning of the training process, we incorporate natural images from the PASCAL VOC 2012 dataset \[45], which contains a total of 2,913 images. For each iteration, we randomly select a batch of natural images from this dataset. As training progresses, we incrementally replace parts of the natural images with medical image slices, such that by the final stage, only medical slices are utilized. For testing, we consistently use fully stitched medical image slices. Although this approach seems to gradually adjust from natural image features to medical image features, it actually disrupts the integrity of the anatomical structure in medical images and leads to a decrease in performance.

These observation results reveal the importance of maintaining slice continuity and contextual integrity for our stitching strategy, effectively bridging the domain and spatial dimensional gaps between natural and medical images.

2.  Fine-tuning Module: Our fine-tuning module performs fine-tuning on the vision foundation model. As one of the most popular universal image segmentation models, SAM serves as the default setting of our fine-tuning module, denoted as  $\mathbb{F}_{\mathrm{FT}}(\cdot)$  . We strip away all the prompts and perform automatic segmentation during inference. Our framework is not limited to a specific fine-tuning strategy, which could be used in different strategies. We uniformly denote the fine-tuning loss  $\mathcal{L}_{ft}$  :

$$
\mathcal {L} _ {f t} = \frac {1}{2} \left(\mathcal {L} _ {D i c e} \left(\mathbf {P} _ {i} ^ {l}, \mathbf {N} _ {i} ^ {l}\right) + \mathcal {L} _ {c e} \left(\mathbf {P} _ {i} ^ {l}, \mathbf {N} _ {i} ^ {l}\right)\right), \tag {5}
$$

where $\mathbf{P}_i^l = \mathrm{F}_{\mathrm{FT}}(\mathbf{M}_i^l)$ is the prediction of fine-tuning module.

![\<img alt="" data-attachment-key="C8SIUZQA" width="670" height="403" src="attachments/C8SIUZQA.jpg" ztype="zimage"> | 670](attachments/C8SIUZQA.jpg)\
Fig. 6. Fine-tuning input strategies comparison. Upsampling fine-tuning directly upsamples each slice.

The previous studies mainly involve fine-tuning only subparts parameters \[8] and incorporating adapters \[9], \[10].

Sub-parts Fine-tuning. Sub-parts fine-tuning methods directly modify the model parameters. MedSAM-v1 \[8] freezes the image encoder and prompt encoder by only fine-tuning the mask decoder, and MedSAM-v2 fine-tunes both image encoder and mask decoder. However, the overall performance still lags behind expert models for medical image segmentation, particularly in terms of boundary consensus.

Adapter Tuning. Adapter tuning \[9], \[46] is to insert adapters into the original fundamental model, and only tune adapters while leaving all pre-trained parameters frozen. An adapter consists of a down-projection, ReLU activation, and up-projection layers. The low-rank-based fine-tuning strategy (LoRA) \[42] injects trainable low-rank decomposition matrices into the layers of the pre-trained model. SAMed \[10] freeze the image encoder of SAM, adopt LoRA by adding a bypass, and fine-tune the mask decoder.

Since LoRA can be merged with the original pre-trained weights for inference, we adopt it as our fine-tuning module method $\mathsf{F}_{\mathsf{LORA}}(\cdot)$ . Following \[10], for the classification head of SAM, ambiguity prediction is replaced by the determined prediction output for each semantic category.

We notice that stitching 2D slices may lead to organs appearing in surrounding and similar regions, guiding SAM to capture these similarities. Our stitching strategy preserves the spatial relationship between slices and could leverage the information of the same target from neighboring slices effectively. For example, as shown in Fig. 6, the same semantic class (e.g., spleen) appears in a similar area (upper left) across three adjacent slices. The upsampling fine-tuning method, which predicts slice-by-slice, confuses the entire spleen (red) into the left kidney (blue) in the middle slice. In contrast, our stitching fine-tuning method could correctly identify the spleen in all three slices. By incorporating inter-slice information, our method effectively guides SAM in recognizing the same organ on different slices. In contrast, our stitching strategy preserves the spatial relationship between slices and could leverage the information of the same target from neighboring slices.

![\<img alt="" data-attachment-key="UPQ5DDCB" width="1292" height="553" src="attachments/UPQ5DDCB.jpg" ztype="zimage"> | 1292](attachments/UPQ5DDCB.jpg)\
Fig. 7. Visualization of fine-tuning module on LA \[15] and BTCV \[44] dataset.

To verify the effectiveness of pseudo-labels on different datasets, we visualize the fine-tuning module prediction of single and multiple target datasets, taking LA \[15] and BTCV \[44] datasets as examples, in Fig. 7. The mask decoder outputs a large-sized 2D mask, which is subsequently restored to a 3D volume as a pseudo-label for the SSL module.

3.  Re-training SSL Module: As defined above, the training data consists of labeled dataset  $L = \{(X_i^l,Y_i)\}_{i = 1}^m$  and unlabeled dataset  $U = \{X_j^u\}_{j = 1}^n$  . The training objective of re-training SSL module  $\mathbb{F}_S(\cdot)$  can be formulated as:

$$
\omega^ {*} = \arg \min  _ {\omega} \left(\mathcal {L} _ {s u p} + \lambda \mathcal {L} _ {u n s u p}\right), \tag {6}
$$

where $\mathcal{L}_{sup}$ and $\mathcal{L}_{unsup}$ are supervised and unsupervised terms, respectively, and $\lambda$ acts as a tradeoff between them. Our pseudo-label guidance $\mathcal{L}_{pl}$ is an unsupervised loss.

$$
\mathcal {L} _ {p l} = \frac {1}{2} \left(\mathcal {L} _ {D i c e} \left(\mathbf {P} _ {j} ^ {u}, \hat {\mathbf {Y}} _ {j} ^ {u}\right) + \mathcal {L} _ {c e} \left(\mathbf {P} _ {j} ^ {u}, \hat {\mathbf {Y}} _ {j} ^ {u}\right)\right), \tag {7}
$$

where $\mathbf{P}_i^u = \mathbb{F}_S(\mathbf{X}_i^u)$ is the prediction of SSL module.

We investigate four 3D medical image semi-supervised methods for our re-training module, including two classical methods (i.e., self-training and mean teacher), and two advanced methods (i.e., ACMT and MagicNet).

Self-training: Self-training \[43] involves three iteratively steps: Firstly, a teacher network is initially trained on $L$ . Secondly, it makes predictions on $U$ to obtain $\hat{U}$ . Thirdly, a student network retrained on the union set $L \cup \hat{U}$ . Among them, steps 2 and 3 are iteratively performed in alternation.

Mean Teacher: Mean teacher (MT) \[21] consists of a student network and a teacher network, and teacher network weights are updated with the exponential moving average (EMA) of student network weights. The unsupervised loss is the consistency regularization between the two networks.

ACMT: ACMT \[17] improves MT to the ambiguity-consensus mean teacher model, encouraging consistency between the student's and the teacher's predictions at the identified ambiguous regions.

![\<img alt="" data-attachment-key="KV9CUY7Y" width="702" height="256" src="attachments/KV9CUY7Y.jpg" ztype="zimage"> | 702](attachments/KV9CUY7Y.jpg)\
Fig. 8. Overview of the proposed $\mathrm{SFR}^+$ framework.

MagicNet: MagicNet \[16] introduces a data augmentation strategy based on MT. First, a pair of labeled and unlabeled samples are mixed into two shuffled cubes. Next, small cubes and mixed cubes are fed into the segmentation network, and finally recovery the mixed cubes.

# C. SFR^{+} Framework

To further enhance our framework, we developed the extension version of SFR, named $\mathrm{SFR}^+$ , as illustrated in Fig. 8. In the SFR framework, the fine-tuning module trained with labeled samples provides pseudo-labels of unlabeled samples uni-directionally to the re-training module. To more effectively extract and leverage information of unlabeled data from the two modules, our $\mathrm{SFR}^+$ introduces a confidence estimation strategy to distinguish between confident and uncertain samples. This approach enables selective optimization of both the fine-tuning and re-training modules.

1.  Confidence Estimation:  $\mathrm{SFR}^+$  introduces confidence estimation to determine how to handle each unlabeled sample. We calculate the voxel-level average confidence for each unlabeled sample and classify them based on a threshold. For each unlabeled sample  $\mathbf{X}_j^u$  , the voxel-level average confidence is defined as:

$$
\mathbf {C} _ {j} ^ {u} = \frac {1}{N _ {v}} \sum_ {v} \max  _ {c} \operatorname {s o f t m a x} _ {c} \left(\mathrm {F} _ {\mathrm {S}} \left(\mathbf {X} _ {j, v} ^ {u}\right)\right), \tag {8}
$$

where $\mathbb{F}_S(\mathbf{X}_{j,v}^u)$ is the model prediction for voxel $v$ and $c$ indexes over the classes. $N_{v}$ is the number of voxels in $\mathbf{X}_j^u$ . Samples are classified based on a threshold $\tau$ :

*   High-confident samples: If  $\mathbf{C}_j^u\geq \tau$  , the sample is considered confident and to update the fine-tuning module.

*   Low-confident samples: If  $\mathbf{C}_j^u < \tau$  , the sample is classified as uncertain and sent to the re-training module.

2.  Selective Training Strategy: In  $\mathrm{SFR}^+$  , selective learning in the fine-tuning and re-training modules allows for more effective handling of unlabeled samples. On one hand, high-confident samples are employed to update the fine-tuning module, ensuring that only reliable information from unlabeled data contributes to further refinement. On the other hand, low-confidence samples benefit from the pseudo-labels, which enable further improvements of the re-training module.

By selectively alternating updates between the two modules, $\mathrm{SFR}^+$ mitigates the risk of error propagation from inaccurate predictions that may arise due to the uni-directional transfer of pseudo-labels.

# D. Summary

The training procedure of our SFR framework is summarized in Fig. 2. During fine-tuning, the 3D labeled volumes are first transformed into larger-sized 2D images by the stitching module. Then fine-tuning module is updated by minimizing the supervised fine-tuning loss. During re-training, the fine-tuning module annotates unlabeled samples and provides pseudo-labels to re-training module. The SSL re-training module is trained by incorporating the supervised information from labeled images and the pseudo-label consistency from unlabeled images.

Our SFR and $\mathrm{SFR}^+$ framework ensures computational efficiency. The fine-tuning module $\mathrm{SFR}_{\mathrm{FT}}$ with the LoRA strategy has the same parameter size as the foundation model (SAM). The re-training module has a parameter scale to the mainstream segmenters like V-Net \[30]. During inference, we discard fine-tuning module and retain only re-training module. The results of new samples are directly predicted using $\mathbb{F}_S(\cdot)$ .

# IV. EXPERIMENTS

The experiments are conducted on two single target datasets (i.e., LA \[15] and BraTS \[47]) and three multiple target datasets (i.e., BTCV \[44], MACT \[48], and AbdomenCT-1K \[49]). We perform experiments of semi-supervised segmentation with moderate annotations and scarce annotations on each dataset. Subsequently, we further analyze the pseudolabels generated by the fine-tuning module, as well as the effectiveness and compatibility of the retraining module, and conduct experiments on different labeled samples.

# A. Datasets

LA Dataset. The LA dataset \[15] in the MICCAI 2018 Atrium Segmentation Challenge is for left atrium segmentation in 3D gadolinium-enhanced MR image scans (GE-MRIs). It contains 100 scans with an isotropic resolution of $0.625 \times 0.625 \times 0.625 \mathrm{~mm}^3$ , and ground truth masks segmented by expert radiologists. Fairly, we follow the same data split and pre-processing procedures as the existing work \[17], \[20], \[50].

BraTS Dataset. The dataset contains preoperative MRI (with T1, T1Gd, T2 and T2-FLAIR modalities) of 335 glioma patients from the BraTS 2019 challenge \[47], \[51], \[52], where 259 patients with high-grade glioma and 76 with low-grade glioma. Following \[17], we only use T2-FLAIR images with the same data split and pre-processing procedures for fair comparison.

BTCV Dataset. The BTCV multiorgan dataset \[44] from the MICCAI Multi-Atlas Labeling Beyond Cranial Vault-Workshop Challenge contains 30 subjects with 3779 axial abdominal CT slices. It consists of 13 organ annotations, including 8 organs of Synapse. We strictly follow the same data split and pre-processing procedures as the existing work \[16], where the volume is divided into $96 \times 96 \times 96$ patches.

MACT Dataset. The MACT dataset \[48] is a public multi-organ abdominal CT reference standard segmentation dataset, containing 90 CT volumes with 8 organs annotation. The original data is from the Cancer Image Archive (TCIA) Pancreas-CT dataset and the BTCV dataset. We follow the same pre-processing procedure as \[16], and we divide 70 cases for training and 20 cases for testing. Following \[16], the volume is divided into $96 \times 96 \times 96$ patches as input volumes.

AbdomenCT-1K Dataset. The AbdomenCT-1K dataset \[49] is a diverse abdominal CT organ segmentation dataset, with more than 1,000 CT scans from 12 medical centers, including multi-phase, multi-vendor, and multidisease cases. We follow the same pre-processing procedure as nnU-Net \[53]. Similar to BTCV and MACT datasets, we divide the image into $96 \times 96 \times 96$ patches as input volumes.

# B. Experimental Settings

In this paper, all the experiments are implemented in Pytorch on the NVIDIA GeForce RTX 3090/4090TI GPU. For foundation model SAM, we conduct all the experiments based on the "ViT-B" version. We adopt LoRA finetuning and the rank of LoRA is set to 4 for efficiency and performance optimization. In our experiments, we follow the current popular setting \[10] in designing mask decoder, which modifies the segmentation head to generate masks of each class in a deterministic manner and aggregate to final segmentation map.

For the semi-supervised network, it is trained by the SGD optimizer with an initial learning rate of 0.01. For LA and BraTS datasets, we follow the training strategy of \[17], \[20]. And we employ four measurements to quantitatively evaluate the segmentation performance, including Dice, Jaccard, the average surface distance (ASD), and the $95\%$ Hausdorff Distance (HD). For BTCV, MACT, and AbdomenCT-1K datasets, we follow the implementation details of Magicnet \[16] and Dice as an evaluation metric for multi-organ segmentation. For fair comparison, we use official reported $\lambda$ of various baselines \[16], \[17], \[20]. In $\mathrm{SFR}^+$ , we set the $\tau$ as 0.985.

During inference, we discard fine-tuning module and new samples are directly predicted by re-training module.

![\<img alt="" data-attachment-key="BJT7FXQI" width="558" height="348" src="attachments/BJT7FXQI.jpg" ztype="zimage"> | 558](attachments/BJT7FXQI.jpg)\
Fig. 9. Fine-tuning with different stitching scales on LA dataset with 16 labeled data. When the stitch scale is set to $9 \times 9$ , the large-sized 2D image represents the complete 3D volume.

TABLEI RESULTS OF LA DATASET WITH MODERATE ANNOTATIONS. $"L / U"$ INDICATES THE NUMBER OF LABLED AND UNLABELED VOLUMES. $\uparrow$ MEANS HIGHER VALUES ARE BETTER AND $\downarrow$ MEANS LOWER VALUES ARE PREFERABLE. LB AND UB ARE THE LOWER AND UPPER BOUND, RESPECTIVELY.METRICS ARE DICE $(\%)$ JACCARD $(\%)$ ASD (VOXEL), AND 95HD (VOXEL).

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ------- | ------------------------------ | ----- | -------- | ---- | ----- |
| L / U   | Method                         | Dice↑ | Jaccard↑ | ASD↓ | 95HD↓ |
| 16 / 0  | V-Net \[30] (LB)               | 86.03 | 76.06    | 3.51 | 14.26 |
| 16 / 64 | UA-MT \[MICCAI'19] \[20]       | 88.88 | 80.21    | 2.26 | 7.32  |
|         | CCT \[CVPR'20] \[22]           | 88.01 | 80.95    | 2.37 | 8.25  |
|         | CPS \[CVPR'21] \[23]           | 87.87 | 78.61    | 2.16 | 12.87 |
|         | DTC \[AAAI'21] \[50]           | 89.42 | 80.98    | 2.10 | 7.32  |
|         | ICT \[NN'22] \[54]             | 89.02 | 80.34    | 1.97 | 10.38 |
|         | CPCL \[JBHI'22] \[55]          | 88.32 | 81.02    | 2.02 | 8.01  |
|         | URPC \[MedIA'22] \[56]         | 88.43 | 81.15    | 2.23 | 8.21  |
|         | BCP \[CVPR'23] \[18]           | 90.10 | 82.11    | 2.51 | 7.62  |
|         | CauSSL \[ICCV'23] \[25]        | 89.48 | 81.20    | 1.75 | 7.55  |
|         | MT \[NIPS'17] \[21]            | 88.12 | 79.03    | 2.65 | 10.92 |
|         | \\( SFR\_{MT} \\) (Ours)       | 90.86 | 83.34    | 1.45 | 6.15  |
|         | \\( SFR\_{MT}^{+} \\) (Ours)   | 90.91 | 83.42    | 1.72 | 5.80  |
|         | ACMT \[MedIA'23] \[17]         | 90.31 | 82.43    | 1.76 | 6.21  |
|         | \\( SFR\_{ACMT} \\) (Ours)     | 90.95 | 83.47    | 1.43 | 6.11  |
|         | \\( SFR\_{ACMT}^{+} \\) (Ours) | 91.00 | 83.53    | 1.61 | 6.13  |
| 80 / 0  | V-Net \[30] (UB)               | 91.14 | 83.82    | 1.52 | 5.75  |


# C. Stitching Strategy Analysis

There is a performance trade-off between the resolution of individual slices and the number of slices stitched together into a single $1024 \times 1024$ image. Increasing the number of slices provides more context and spatial continuity. However, it also requires resizing the slices to fit into the fixed resolution, potentially resulting in a loss of fine-grained details. Therefore, how to consider both the slice resolution and the stitching scale becomes a critical issue.

In Fig. 9, we explored different stitching scales ( $d \times d$ grid) to analyze this trade-off. On the LA dataset \[15], when the stitch scale is set to $9 \times 9$ , the large-sized 2D image represents the complete 3D volume, with each slice maintaining its original resolution without any downsampling. The results show that when the stitch scale is small, performance tends to be inferior. With a small scale, it is possible that the representation may not fully capture the volume context, limiting the model's ability to leverage inter-slice information. By increasing the number of slices, the model benefits from more context, resulting in improved segmentation performance, particularly when the stitched image encompasses the entire 3D volume. When the resolution of individual slices is maintained, the large-sized 2D image represents the 3D volume without any downsampling, and the best results are achieved. We supplement this figure by further increasing the stitching scale, that is, reducing the resolution of each slice. However, it becomes counterproductive when further increasing the stitching scale—excessive downsampling of individual slices begins to degrade performance as the resolution becomes too low to capture critical anatomical details. In our method, each slice retains its resolution throughout the process, ensuring that no upsampling or downsampling occurs, which balances the need for spatial context with the preservation of resolution. The stitching process does not affect the resolution of the individual slices; rather, it combines them into a large-sized 2D image. As a result, the final 2D segmentation predictions are also aligned with the original slice resolution, preserving the fine-grained details essential for high-quality segmentation.

TABLE II RESULTS OF BRATS DATASET WITH MODERATE ANNOTATIONS.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| -------- | ------------------------------ | ----- | -------- | ---- | ----- |
| L / U    | Method                         | Dice↑ | Jaccard↑ | ASD↓ | 95HD↓ |
| 50 / 0   | 3D U-Net \[57] (LB)            | 80.16 | 71.55    | 3.43 | 22.68 |
| 50 / 200 | UA-MT \[MICCAI'19] \[20]       | 83.12 | 73.01    | 2.30 | 9.87  |
|          | CCT \[CVPR'20] \[22]           | 82.53 | 72.36    | 2.21 | 15.87 |
|          | CPS \[CVPR'21] \[23]           | 84.01 | 74.02    | 2.18 | 12.16 |
|          | DTC \[AAAI'21] \[50]           | 83.43 | 73.56    | 2.34 | 14.77 |
|          | ICT \[NN'22] \[54]             | 81.76 | 72.01    | 2.82 | 9.66  |
|          | CPCL \[JBHI'22] \[55]          | 83.48 | 74.08    | 2.08 | 9.53  |
|          | URPC \[MedLA'22] \[56]         | 82.93 | 72.57    | 4.19 | 15.93 |
|          | BCP \[CVPR'23] \[18]           | 84.17 | 74.37    | 3.24 | 11.69 |
|          | CauSSL \[ICCV'23] \[25]        | 81.09 | 71.01    | 3.76 | 11.90 |
|          | MT \[NIPS'17] \[21]            | 82.96 | 72.95    | 2.32 | 9.85  |
|          | \\( SFR\_{MT} \\) (Ours)       | 85.19 | 75.77    | 2.92 | 11.04 |
|          | \\( SFR\_{MT}^{+} \\) (Ours)   | 86.08 | 76.79    | 1.93 | 8.50  |
|          | ACMT \[MedLA'23] \[17]         | 84.63 | 74.39    | 2.11 | 9.50  |
|          | \\( SFR\_{ACMT} \\) (Ours)     | 85.81 | 76.66    | 1.79 | 7.75  |
|          | \\( SFR\_{ACMT}^{+} \\) (Ours) | 86.09 | 76.80    | 2.47 | 8.51  |
| 250 / 0  | 3D U-Net \[57] (UB)            | 85.93 | 76.81    | 1.93 | 9.85  |


TABLE III RESULTS OF ABDOMENCT-1K DATASET WITH MODERATE ANNOTATIONS AND THE EVALUATION METRIC IS THE DICE SCORE $(\%)$ .\* MEANS ALL THE Pixels ARE PREDICTED AS BACKGROUND OR ANOTHER REGION.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| --------- | ---------------------------------- | ----- | ----- | ------ | ------ | -------- |
| L / U     | Method                             | AVG ↑ | Liver | Kidney | Spleen | Pancreas |
| 180 / 0   | V-Net \[30] (LB)                   | 67.17 | 96.03 | 93.64  | 78.99  | 0\*      |
| 180 / 720 | MT \[NIPS'17] \[21]                | 75.16 | 93.78 | 91.54  | 76.55  | 38.79    |
|           | \\( SFR\_{MT} \\) (Ours)           | 88.73 | 96.10 | 94.55  | 93.63  | 70.65    |
|           | \\( SFR\_{MT}^{+} \\) (Ours)       | 90.00 | 96.11 | 94.50  | 95.19  | 74.19    |
|           | MagicNet \[CVPR'23] \[16]          | 90.54 | 96.38 | 94.54  | 95.17  | 76.08    |
|           | \\( SFR\_{MagicNet} \\) (Ours)     | 91.41 | 96.70 | 95.01  | 96.02  | 77.90    |
|           | \\( SFR\_{MagicNet}^{+} \\) (Ours) | 91.70 | 96.66 | 95.20  | 96.06  | 78.88    |
| 900 / 0   | V-Net \[30] (UB)                   | 88.26 | 95.31 | 94.49  | 92.30  | 70.94    |


TABLE IV RESULTS OF BTCV DATASET WITH MODERATE ANNOTATIONS AND THE EVALUATION METRIC IS THE DICE SCORE $(\%)$ .NOTE: SPL: SPLEEN, R.KID: RIGHT KIDNEY, L.KID: LEFT KIDNEY, GALL: GALLBLADDER, ESO: ESOPHAGUS, LIV: LIVER, STO: STOMACH, AOR: AORTA, IVC: INFERIOR VENA CAVA, VEINS: PORTAL AND SPLenic VEINS, PAN: PANCREAS, LG/RG: LEFT/RIGHT ADRENAL GLANDS.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ------ | ---------------------------------- | ------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| L / U  | Method                             | AVG ↑         | Spl   | R.Kid | L.Kid | Gall  | Eso   | Liv   | Sto   | Aor   | IVC   | Veins | Pan   | RG    | LG    |
| 7 / 0  | V-Net \[30] (LB)                   | 67.17         | 84.98 | 82.72 | 82.07 | 36.64 | 63.48 | 93.54 | 57.49 | 89.74 | 78.63 | 60.42 | 49.39 | 55.60 | 38.49 |
| 7 / 11 | UA-MT \[MICCAI'19] \[20]           | 67.75         | 88.74 | 75.88 | 78.91 | 54.25 | 58.55 | 93.46 | 58.90 | 89.23 | 76.15 | 62.30 | 47.91 | 51.53 | 44.92 |
|        | CPS \[CVPR'21] \[23]               | 65.81         | 87.56 | 72.99 | 77.59 | 53.31 | 54.08 | 92.41 | 54.58 | 87.75 | 74.32 | 58.68 | 48.02 | 50.39 | 43.86 |
|        | ICT \[NN'22] \[54]                 | 73.69         | 90.31 | 84.41 | 86.96 | 49.22 | 65.65 | 94.29 | 65.95 | 90.23 | 81.44 | 69.56 | 66.61 | 57.35 | 56.01 |
|        | SS-Net \[MICCAI'22] \[58]          | 58.26         | 84.74 | 76.37 | 74.19 | 43.42 | 57.05 | 92.90 | 14.37 | 83.14 | 69.77 | 52.45 | 27.08 | 54.29 | 27.66 |
|        | SLC-Net \[MICCAI'22] \[59]         | 70.40         | 90.05 | 84.00 | 86.43 | 56.16 | 58.91 | 94.68 | 70.72 | 89.93 | 79.45 | 60.59 | 54.22 | 51.03 | 39.08 |
|        | MT \[NIPS'17] \[21]                | 65.68         | 85.70 | 78.93 | 79.08 | 42.80 | 61.09 | 93.45 | 57.57 | 89.70 | 80.30 | 63.95 | 41.14 | 50.46 | 29.69 |
|        | \\( SFR\_{MT} \\) (Ours)           | 70.09 (↑4.41) | 87.92 | 82.86 | 81.94 | 53.02 | 60.50 | 95.06 | 72.82 | 89.71 | 81.23 | 67.46 | 59.29 | 41.77 | 37.58 |
|        | \\( SFR\_{MT}^{+} \\) (Ours)       | 71.81 (↑6.13) | 88.78 | 84.08 | 86.48 | 55.04 | 64.61 | 95.08 | 74.60 | 89.89 | 81.62 | 63.38 | 63.47 | 41.99 | 44.57 |
|        | MagicNet \[CVPR'23] \[16]          | 76.74         | 91.61 | 85.02 | 88.13 | 58.16 | 66.72 | 94.07 | 74.46 | 90.77 | 84.31 | 71.56 | 68.90 | 63.48 | 60.47 |
|        | \\( SFR\_{MagicNet} \\) (Ours)     | 77.06 (↑0.32) | 92.09 | 85.36 | 83.70 | 63.38 | 69.97 | 94.15 | 74.69 | 91.14 | 84.45 | 70.68 | 67.22 | 64.39 | 60.53 |
|        | \\( SFR\_{MagicNet}^{+} \\) (Ours) | 77.07 (↑0.33) | 90.43 | 86.03 | 86.15 | 60.32 | 70.72 | 94.61 | 74.83 | 91.15 | 84.21 | 71.57 | 70.39 | 61.55 | 59.90 |
| 18 / 0 | V-Net \[30] (UB)                   | 76.28         | 84.00 | 84.82 | 86.38 | 67.42 | 65.02 | 94.83 | 73.75 | 90.27 | 84.19 | 69.85 | 63.54 | 62.60 | 65.02 |


TABLE V RESULTS OF MACT DATASET WITH MODERATE ANNOTATIONS AND THE EVALUATION METRIC IS THE DICE SCORE $(\%)$ .L.KIDEY: LEFT KIDNEY. \* MEANS ALL THE Pixels ARE PREDICTED AS BACKGROUND OR ANOTHER REGION.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ------- | ---------------------------------------------------- | ------------- | ------ | -------- | ----------- | --------- | ----- | ------- | -------- | -------- |
| L / U   | Method                                               | AVG ↑         | Spleen | L.Kedney | Gallbladder | Esophagus | Liver | Stomach | Pancreas | Doudenum |
| 14 / 0  | V-Net \[30] (LB)                                     | 69.05         | 93.94  | 94.36    | 60.43       | 0\*       | 95.57 | 78.04   | 72.19    | 57.83    |
| 14 / 56 | UA-MT \[MICCAI'19] \[20]                             | 78.33         | 93.76  | 92.07    | 75.01       | 65.53     | 95.42 | 77.91   | 72.40    | 54.57    |
|         | CPS \[CVPR'21] \[23]                                 | 65.17         | 93.84  | 79.80    | 64.62       | 0\*       | 93.66 | 81.49   | 62.25    | 45.70    |
|         | ICT \[NN'22] \[54]                                   | 77.52         | 93.12  | 92.05    | 71.51       | 67.54     | 94.38 | 77.40   | 68.03    | 56.16    |
|         | SS-Net \[MICCAI'22] \[58]                            | 69.69         | 93.15  | 92.89    | 71.75       | 0\*       | 93.08 | 73.99   | 73.11    | 59.56    |
|         | MT \[NIPS'17] \[21]                                  | 77.10         | 93.52  | 93.06    | 75.19       | 67.61     | 94.44 | 67.99   | 73.39    | 51.56    |
|         | \\( \mathbf{SFR}\_{\mathbf{MT}} \\) (Ours)           | 82.50 (↑5.40) | 95.77  | 94.42    | 83.61       | 68.92     | 96.05 | 82.81   | 76.30    | 62.10    |
|         | \\( \mathbf{SFR}\_{\mathbf{MT}}^{+} \\) (Ours)       | 83.05 (↑5.59) | 95.35  | 92.84    | 80.67       | 69.93     | 95.44 | 87.64   | 77.32    | 65.18    |
|         | MagicNet \[CVPR'23] \[16]                            | 81.04         | 94.19  | 94.38    | 76.07       | 74.08     | 95.04 | 78.55   | 73.36    | 62.61    |
|         | \\( \mathbf{SFR}\_{\mathbf{MagicNet}} \\) (Ours)     | 82.87 (↑1.83) | 95.54  | 94.45    | 82.67       | 72.89     | 95.93 | 81.49   | 75.90    | 64.06    |
|         | \\( \mathbf{SFR}\_{\mathbf{MagicNet}}^{+} \\) (Ours) | 83.47 (↑2.43) | 94.65  | 94.39    | 81.63       | 74.46     | 95.15 | 84.41   | 74.85    | 68.21    |
| 70 / 0  | V-Net \[30] (UB)                                     | 85.85         | 96.07  | 95.05    | 84.09       | 72.28     | 96.31 | 89.45   | 82.05    | 71.47    |


In this work, our stitching module reorganizes a volume (3D raw image or 3D patch) into a $1024 \times 1024$ image, enabling a complete representation of the volume within a single image. For the size of slices, different slices might vary a lot in different datasets. For large-scale medical datasets, such as multi-organ abdominal scans, the full 3D volume often has a very large voxel size (e.g., $512 \times 512 \times 512$ ). In 3D medical image segmentation approaches like 3D-UNet \[57], V-Net \[30], and nnU-Net \[53], it is common practice to divide the volume into smaller 3D patches (e.g., $96 \times 96 \times 96$ ) for training and inference. In our approach, we follow a similar strategy during both the fine-tuning and retraining phases, using patches of consistent size. Before fine-tuning SAM, we stitch the 3D patches into a $1024 \times 1024$ image, ensuring that the patch structure is preserved throughout the process.

Specifically, we divide the full 3D volume into consistent 3D patches, each maintaining an original resolution to avoid any loss of fine anatomical details. Each patch is then systematically stitched into a $1024 \times 1024$ image to capture full spatial context. For the LA segmentation \[15], we adopted the same strategy as works \[17], \[20], \[50], where the patch size is set to $H \times W \times D$ where $W = H = 112$ and $D = 80$ . On the BraTS dataset \[47], we followed the approach used in ACMT \[17], using patch dimensions of $W = H = D = 96$ . For the BTCV \[44], MACT \[48] and AbdomenCT-1K \[49] datasets, following MagicNet \[16], the patch size is set to $W = H = D = 96$ . After fine-tuning, the predictions are restored to their 3D form through an inverse stitching transform, ensuring that the resolution and spatial integrity of the original 3D image are preserved. By carefully adjusting the patch size and stitching scale according to each dataset's needs, our approach ensures robust performance across both single-target and multi-target segmentation tasks.

TABLE VI RESULTS OF LA DATASET WITH $96^{3}$ PATCH SIZE.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ------- | -------------------------- | -------------- | -------- | ----- | ----- |
| L / U   | Method                     | Dice↑          | Jaccard↑ | ASD↓  | 95HD↓ |
| 16 / 64 | MT \[NIPS'17] \[21]        | 87.47          | 78.03    | 3.92  | 14.13 |
|         | \\( SFR\_{MT} \\) (Ours)   | 89.09 (↑1.62)  | 80.73    | 2.58  | 10.33 |
|         | ACMT \[Medla'23] \[17]     | 88.50          | 79.56    | 3.58  | 14.71 |
|         | \\( SFR\_{ACMT} \\) (Ours) | 89.49 (↑0.99)  | 81.09    | 1.94  | 7.88  |
| 1 / 79  | MT \[NIPS'17] \[21]        | 44.89          | 29.85    | 18.86 | 44.78 |
|         | \\( SFR\_{MT} \\) (Ours)   | 65.65 (↑20.76) | 53.64    | 9.82  | 29.43 |
|         | ACMT \[Medla'23] \[17]     | 58.40          | 42.61    | 20.55 | 50.54 |
|         | \\( SFR\_{ACMT} \\) (Ours) | 68.24 (↑9.84)  | 57.40    | 12.38 | 30.76 |


In addition, we investigate a simplified strategy to ensure uniformity across different datasets and further validate the robustness of our framework. In this alternative approach, we processed all 3D medical images, regardless of their original size, by first cropping into $96 \times 96 \times 96$ patches. These patches are then directly stitched together to form a $10 \times 10$ 2D image. Typically, the resolution of slices (both in height and width) in 3D medical images exceeds 96, though the number of slices (depth) may vary. For datasets where the number of slices is fewer than 96, we employ interpolation to achieve the desired depth. This approach provides a consistent standard for stitching, ensuring that the slices are uniformly processed before being fed into the model, without manual intervention. We have conducted experiments using this approach on the LA dataset with both moderate and scarce annotations, and the results are shown in Table VI. Notably, our method achieves over $89\%$ Dice score on moderate annotations and outperforms SSL methods by a large margin on scarce annotations. This demonstrates that even with this simplified and fixed patch-size mechanism, the performance remains effective.

![\<img alt="" data-attachment-key="IWP4BJVQ" width="506" height="408" src="attachments/IWP4BJVQ.jpg" ztype="zimage"> | 506](attachments/IWP4BJVQ.jpg)\
LA Dataset

![\<img alt="" data-attachment-key="ZJ8MY238" width="464" height="403" src="attachments/ZJ8MY238.jpg" ztype="zimage"> | 464](attachments/ZJ8MY238.jpg)\
Fig. 10. Kernel dense estimations of MT and $\mathrm{SFR}_{\mathrm{MT}}$ on LA and BraTS datasets with Moderate Annotation (20% labeled data) and Scarce Annotation (1 labeled data).\
BraTS Dataset

TABLE VII RESULTS OF LA DATASET WITH SCARCE ANNOTATIONS.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ------ | ------------------------------------------------ | -------------- | -------- | ----- | ----- |
| L / U  | Method                                           | Dice↑          | Jaccard↑ | ASD↓  | 95HD↓ |
| 1 / 0  | V-Net \[30] (LB)                                 | 17.99          | 12.93    | 19.66 | 44.58 |
| 1 / 79 | MT \[NIPS'17] \[21]                              | 29.68          | 18.19    | 18.63 | 42.58 |
|        | \\( \mathbf{SFR}\_{\mathbf{MT}} \\) (Ours)       | 74.40 (↑44.72) | 61.47    | 6.25  | 25.90 |
|        | \\( \mathbf{SFR}\_{\mathbf{MT}}^{+} \\) (Ours)   | 81.03 (↑51.35) | 69.44    | 5.12  | 19.77 |
|        | ACMT \[MedIA'23] \[17]                           | 72.64          | 58.33    | 10.42 | 33.22 |
|        | \\( \mathbf{SFR}\_{\mathbf{ACMT}} \\) (Ours)     | 76.76 (↑4.12)  | 64.65    | 8.54  | 28.07 |
|        | \\( \mathbf{SFR}\_{\mathbf{ACMT}}^{+} \\) (Ours) | 83.72 (↑11.08) | 72.80    | 4.54  | 18.60 |


TABLE VIII RESULTS OF BRATS DATASET WITH SCARCE ANNOTATIONS.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ------- | ------------------------------ | -------------- | -------- | ----- | ----- |
| L / U   | Method                         | Dice↑          | Jaccard↑ | ASD↓  | 95HD↓ |
| 1 / 0   | 3D U-Net \[57] (LB)            | 73.74          | 61.44    | 13.81 | 37.07 |
| 1 / 249 | MT \[NIPS'17] \[21]            | 63.52          | 50.59    | 20.59 | 47.54 |
|         | \\( SFR\_{MT} \\) (Ours)       | 78.58 (↑15.06) | 66.56    | 7.16  | 23.43 |
|         | \\( SFR\_{MT}^{+} \\) (Ours)   | 79.04 (↑15.52) | 67.55    | 6.06  | 21.11 |
|         | ACMT \[Medla'23] \[17]         | 63.32          | 51.41    | 9.28  | 31.71 |
|         | \\( SFR\_{ACMT} \\) (Ours)     | 78.47 (↑15.15) | 66.33    | 8.17  | 26.64 |
|         | \\( SFR\_{ACMT}^{+} \\) (Ours) | 79.24 (↑15.92) | 67.56    | 6.72  | 22.88 |


# D. Segmentation Results with Moderate Annotations

1.  Single Target Segmentation: We compare with current SOTA methods on LA dataset and BraTS dataset, including V-Net \[30], MT \[21], UA-MT \[20], CCT \[22], CPS \[23], DTC \[50], ICT \[54], CPCL \[55], URPC \[56], BCP \[18], MCCauSSL \[25] and ACMT \[17], and the results are shown in Table I and Table II. We follow the same data partitioning and training strategies as in previous works \[16]-\[18], \[20], ensuring that the comparison results are fair. In all experiments, we apply data augmentation techniques such as random cropping and flipping. It could be seen that BCP \[18], CauSSL \[25], and ACMT \[17] perform well, which may be attributed to these methods are often well-designed for small datasets, where their sophisticated architecture may yield promising results. We select the basic MT \[21] network and method ACMT \[17] with the current highest performance, and respectively provide pseudo-labels of adapted SAM to assist training. Our SFR framework with MT and ACMT as the retraining module is denoted as  $\mathrm{SFR}_{\mathrm{MT}}$  and  $\mathrm{SFR}_{\mathrm{ACMT}}$  , respectively. The results show that our SFR framework brings substantial improvements in both MT and ACMT methods, approaching the performance of fully-supervised segmentation. Additionally, it is worth noting that SFR shows improvements in the Average Surface Distance (ASD) metric. In both datasets, our semi-supervised method  $\mathrm{SFR}_{\mathrm{ACMT}}$  not only reduces the ASD but even surpasses the fully-supervised results. This improvement in ASD indicates a meaningful boost in boundary precision, which is critical for accurate medical image segmentation. Our  $\mathrm{SFR}^{+}$  framework achieves a further improvement over SFR on both datasets. In Table I, it is clear that  $\mathrm{SFR}^{+}$  improves the Dice score of ACMT from  $90.31\%$  to  $91.00\%$  , which is only  $0.14\%$  lower than the fully supervised result of  $91.14\%$  .

TABLE IX RESULTS OF ABDOMENCT-1K DATASET WITH SCARCE ANNOTATIONS AND THE EVALUATION METRIC IS THE DICE SCORE $(\%)$ . \* MEANS ALL THE Pixels ARE PREDICTED AS BACKGROUND OR ANOTHER REGION.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ------- | ---------------------------------------------------- | ----- | ----- | ------ | ------ | -------- |
| L / U   | Method                                               | AVG ↑ | Liver | Kidney | Spleen | Pancreas |
| 1 / 0   | V-Net \[30] (LB)                                     | 22.72 | 56.04 | 33.20  | 0\*    | 0.02     |
| 1 / 899 | MT \[NIPS'17] \[21]                                  | 33.34 | 70.62 | 45.09  | 0.90   | 16.75    |
|         | \\( \mathsf{SFR}\_{\mathsf{MT}} \\) (Ours)           | 57.89 | 72.09 | 66.06  | 71.33  | 22.10    |
|         | \\( \mathsf{SFR}\_{\mathsf{MT}}^{+} \\) (Ours)       | 65.12 | 77.40 | 71.35  | 77.27  | 34.45    |
|         | MagicNet \[CVPR'23] \[16]                            | 54.19 | 75.26 | 62.44  | 48.82  | 30.23    |
|         | \\( \mathsf{SFR}\_{\mathsf{MagicNet}} \\) (Ours)     | 64.32 | 75.04 | 68.08  | 76.12  | 38.03    |
|         | \\( \mathsf{SFR}\_{\mathsf{MagicNet}}^{+} \\) (Ours) | 66.16 | 79.91 | 71.63  | 70.91  | 42.18    |


2\) Multiple Target Segmentation: Table III, Table IV, and Table V show the results on AbdomenCT-1K, BTCV, and MACV datasets, respectively. We compare our method with V-Net \[30], MT \[21], UA-MT \[20], CPS \[23], ICT \[54],SS-Net \[58], SLC-Net \[59], \[60], and MagicNet \[16]. We employ SFR framework to guide the classical MT \[21] and the SOTA MagicNet \[16] methods, i.e., $\mathrm{SFR}_{\mathrm{MT}}$ and $\mathrm{SFR}_{\mathrm{MagicNet}}$ . By leveraging the SFR framework, both methods achieve significant performance improvements. On the BTCV dataset, $\mathrm{SFR}_{\mathrm{MagicNet}}$ even surpasses the results obtained with fully-supervised learning. For the AbdomenCT-1K dataset, it could be observed that $\mathrm{SFR}_{\mathrm{MT}}$ improves performance across all four organs, boosting the average Dice score by $13.57\%$ (88.73% vs. 75.16%) with $20\%$ labeled data. In addition, our $\mathrm{SFR}^{+}$ framework demonstrates improvements across the three different datasets for moderate annotation scenarios, highlighting its robustness and effectiveness in segmentation.

TABLE X RESULTS OF BTCV DATASET WITH SCARCE ANNOTATIONS AND THE EVALUATION METRIC IS THE DICE SCORE $(\%)$ .NOTE THAT THE FULL NAME OF THE ORGAN IS THE SAME AS IN TABLE IV. \* MEANS ALL THE Pixels ARE PREDICTED AS BACKGROUND OR ANOTHER REGION.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ------ | -------------------------------------------------- | -------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| L / U  | Method                                             | AVG ↑          | Spl   | R.Kid | L.Kid | Gall  | Eso   | Liv   | Sto   | Aor   | IVC   | Veins | Pan   | RG    | LG    |
| 1 / 0  | V-Net \[30] (LB)                                   | 14.84          | 52.97 | 29.59 | 2.34  | 27.63 | 0\*   | 75.25 | 3.33  | 0\*   | 0\*   | 0\*   | 1.87  | 0\*   | 0\*   |
| 1 / 17 | MT \[NIPS'17] \[21]                                | 29.47          | 67.12 | 41.92 | 41.01 | 0.82  | 0\*   | 81.67 | 3.98  | 53.30 | 46.81 | 30.60 | 15.02 | 0\*   | 0.82  |
|        | \\( \mathbf{SFR}\_{\text{MT}} \\) (Ours)           | 39.59 (↑10.12) | 82.87 | 63.52 | 61.20 | 0\*   | 0\*   | 87.90 | 19.17 | 81.68 | 61.43 | 33.01 | 23.86 | 0\*   | 0\*   |
|        | \\( \mathbf{SFR}\_{\text{MT}}^{+} \\) (Ours)       | 43.51 (↑14.04) | 89.24 | 46.81 | 30.39 | 50.74 | 0\*   | 88.97 | 38.99 | 82.45 | 55.19 | 42.96 | 39.90 | 0\*   | 0\*   |
|        | MagicNet \[CVPR'23] \[16]                          | 40.02          | 52.97 | 50.22 | 44.17 | 10.04 | 35.55 | 76.02 | 29.30 | 56.69 | 49.65 | 37.59 | 48.58 | 16.31 | 13.15 |
|        | \\( \mathbf{SFR}\_{\text{MagicNet}} \\) (Ours)     | 53.59 (↑13.57) | 85.01 | 67.43 | 56.74 | 34.77 | 35.90 | 86.65 | 38.07 | 84.58 | 57.45 | 56.90 | 43.35 | 20.94 | 28.84 |
|        | \\( \mathbf{SFR}\_{\text{MagicNet}}^{+} \\) (Ours) | 58.84 (↑18.82) | 89.02 | 64.19 | 55.70 | 44.27 | 46.84 | 89.50 | 39.41 | 87.24 | 66.02 | 60.92 | 49.12 | 35.98 | 36.75 |


TABLE XI RESULTS OF MACT DATASET WITH SCARCE ANNOTATIONS AND THE EVALUATION METRIC IS THE DICE SCORE $(\%)$ .L.KIDEY: LEFT KIDNEY. \* MEANS ALL THE Pixels ARE PREDICTED AS BACKGROUND OR ANOTHER REGION.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ------ | ---------------------------------------------------- | -------------- | ------ | -------- | ----------- | --------- | ----- | ------- | -------- | -------- |
| L / U  | Method                                               | AVG ↑          | Spleen | L.Kedney | Gallbladder | Esophagus | Liver | Stomach | Pancreas | Doudenum |
| 1 / 0  | V-Net \[30] (LB)                                     | 18.73          | 41.76  | 5.19     | 3.70        | 0.23      | 85.87 | 11.09   | 1.98     | 0.05     |
| 1 / 69 | MT \[NIPS'17] \[21]                                  | 23.09          | 37.08  | 29.58    | 6.74        | 0\*       | 77.41 | 27.65   | 4.78     | 1.49     |
|        | \\( \mathbf{SFR}\_{\mathbf{MT}} \\) (Ours)           | 36.23 (↑13.14) | 74.41  | 64.11    | 7.37        | 0\*       | 91.01 | 36.94   | 13.37    | 2.65     |
|        | \\( \mathbf{SFR}\_{\mathbf{MT}}^{+} \\) (Ours)       | 52.54 (↑29.45) | 76.27  | 86.97    | 65.94       | 15.70     | 86.85 | 21.85   | 35.31    | 31.42    |
|        | MagicNet \[CVPR'23] \[16]                            | 42.90          | 79.32  | 62.32    | 21.30       | 20.87     | 89.60 | 44.96   | 12.83    | 12.01    |
|        | \\( \mathbf{SFR}\_{\mathbf{MagicNet}} \\) (Ours)     | 49.08 (↑6.18)  | 89.57  | 74.00    | 33.21       | 16.55     | 91.05 | 46.91   | 25.84    | 15.47    |
|        | \\( \mathbf{SFR}\_{\mathbf{MagicNet}}^{+} \\) (Ours) | 54.05 (↑11.15) | 84.74  | 85.03    | 36.13       | 25.01     | 90.34 | 52.22   | 41.92    | 17.03    |


![\<img alt="" data-attachment-key="847IQDLF" width="1433" height="578" src="attachments/847IQDLF.jpg" ztype="zimage"> | 1433](attachments/847IQDLF.jpg)\
Fig. 11. Segmentation results on LA and MACT datasets with moderate annotation, and on BraTS and BTCV datasets with scarce annotation.

The single target segmentation examples on LA dataset with 16 labeled data and multiple target segmentation examples on MACT dataset with 14 labeled data are shown in Fig. 11. Our $\mathrm{SFR}_{\mathrm{MT}}$ and $\mathrm{SFR}_{\mathrm{ACMT}}$ predictions align more accurately with ground truth masks, further validating the effectiveness. Our framework demonstrates its ability to adapt different data.

# E. Segmentation Results with Scarce Annotations

1.  Single Target Segmentation: We evaluate the impact of SFR with scarce annotations on the LA and BraTS datasets. In Table VII, when only one labeled volume is available, supervised learning performs poorly, achieving only  $17.99\%$  accuracy. SFR takes advantage of medical image stitching and SAM's powerful feature capture capabilities to guide MT to increase from  $29.68\%$  to  $74.40\%$  . The results of the BraTS are shown in Table VIII and SFR helps MT improve by  $15.06\%$  and MagicNet by  $15.15\%$  . In addition, one phenomenon observed is that some semi-supervised methods perform worse than the lower bound. This could be attributed to the incomplete similarity of the data distribution between labeled and unlabeled samples, and the inaccurate pseudo-labels from the unlabeled samples have a negative impact on the model's learning process. Furthermore, our  $\mathrm{SFR}^{+}$  framework exploits a confidence-based selective strategy and achieves better results than SFR on both MT and ACMT methods.

We recognize that there is a large performance difference between MT with moderate and scarce annotations on the LA dataset compared to the BraTS dataset. To better understand this performance improvement with the increase of labeled samples, we perform kernel dense estimation on different settings, as shown in Fig. 10. In the LA dataset, the results show that with scarce annotation, the distribution gap between labeled and unlabeled data is quite large. This may cause MT to struggle in feature extraction because it relies on this labeled set to learn the overall data distribution. As the number of labeled samples increases, the distribution gap between labeled and unlabeled data becomes small, so the effect of MT improves rapidly. For BraTS dataset, the difference between the feature distributions of labeled and unlabeled data is not as pronounced as in LA dataset with scarce annotation, which may lead to a better effect on BraTS than on LA dataset.

2.  Multiple Target Segmentation: The results of scarce annotations on three multi-organ datasets are presented in Table IX, Table X, and Table XI. On BTCV dataset with only one labeled sample, the model struggles to identify esophagus, aorta, inferior vena cava, portal and splenic veins, left and right adrenal glands, with Dice scores below  $5\%$  for left kidney, stomach, and pancreas. Semi-supervised methods with unlabeled samples alleviate the poor performance of most organs. Our SFR framework achieves improvements of  $10.12\%$  and  $13.57\%$  on MT and MagicNet models, respectively, particularly with a nearly  $30\%$  improvement on aorta region. Similarly, on MACT dataset, SFR helps identify challenging classes, with significant improvements of  $37.33\%$  for spleen and  $34.53\%$  for right kidney on the Mean Teacher model. Also,  $\mathrm{SFR}_{\mathrm{MagicNet}}$  achieves a performance improvement of up to  $10.13\%$  (64.32% vs. 66.16%) Dice score on the AbdomenCT-1K dataset with one annotation. Our  $\mathrm{SFR}^{+}$  framework further improves performance, achieving more than  $10\%$  improvement in the Dice metric on all three datasets.

Fig. 11 shows the segmentation examples of single target dataset BraTS and multiple target dataset BTCV with scarce annotation. The effectiveness of our proposed framework can be shown in some challenging examples. In these examples, the baselines (i.e., MT \[21], ACMT \[17] and MagicNet \[16]) tend to generate false predictions or incomplete structures, whereas the introduction of SFR framework can mitigate these problems and obtain a more plausible segmentation result. Our framework demonstrates its ability to adapt flexibly to different task requirements. This versatility highlights the potential of our framework to handle a diverse range of medical imaging tasks with varying levels of annotation availability.

TABLE XII COMPARISON OF FINE-TUNING AND RE-TRAINING SEGMENTATION RESULTS OF DICE ON FOUR DATSETS. "# L" IS THE NUMBER OF Labeled DATA."M" MEANS MODERATE AND "S" MEANS SCARCE.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| -- | ---------------- | ------- | ------------ | ------------ | ------------- | ------------- |
| #L | Method           | Params↓ | Dice↑        |              |               |               |
|    |                  |         | LA           | BraTS        | BTCV          | MACT          |
| M  | SFRFT            | 91M     | 90.39        | 84.94        | 65.26         | 79.59         |
|    | SFRMT            | 10M     | 90.86(↑0.47) | 85.19(↑0.25) | 70.09(↑4.83)  | 82.50(↑2.91)  |
|    | SFRACMT/MagicNet | 10/18M  | 90.95(↑0.56) | 85.81(↑0.87) | 77.06(↑11.80) | 82.87(↑3.28)  |
| S  | SFRFT            | 91M     | 74.78        | 78.14        | 40.98         | 35.13         |
|    | SFRMT            | 10M     | 74.40(↓0.38) | 78.58(↑0.44) | 39.59(↓1.39)  | 36.23(↑1.10)  |
|    | SFRACMT/MagicNet | 10/18M  | 76.76(↑1.98) | 78.47(↑0.33) | 53.59(↑12.61) | 49.08(↑13.95) |


![\<img alt="" data-attachment-key="929VVWWM" width="336" height="355" src="attachments/929VVWWM.jpg" ztype="zimage"> | 336](attachments/929VVWWM.jpg)\
Fig. 12. Results of SFR on self-training (ST in short), MT, and ACMT with different numbers of labeled samples.

![\<img alt="" data-attachment-key="C2W7MN7S" width="331" height="355" src="attachments/C2W7MN7S.jpg" ztype="zimage"> | 331](attachments/C2W7MN7S.jpg)

# F. Effectiveness of Re-training Module

We report the results of fine-tuning step omitting re-training in all datasets in Table XII, and compare the model parameters and performance after fine-tuning and re-training steps. By comparing fine-tuning results with re-training results in more detail, we observe that there is a slight performance decrease in $\mathrm{SFR}_{\mathrm{MT}}$ with scarce labels on LA and BTCV datasets, but overall there is a clear improvement in effect. On the other hand, considering the practical deployment requirements, retraining model has a smaller number of parameters than fine-tuning model. Our fine-tuning module $\mathrm{SFR}_{\mathrm{FT}}$ has the same parameter size as the foundation model (SAM) and the retraining module reduces the parameter scale to the mainstream segmenters. The results validate that the re-training module contributes to an overall performance boost for semi-supervised segmentation.

# G. Further Analysis

TABLE XIII COMPARISON OF 3D-BASED FINE-TUNING METHOD ON LA DATASET.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ------------------- | ------- | ----- | -------- | ---- | ----- |
| Method              | Params↓ | Dice↑ | Jaccard↑ | ASD↓ | 95HD↓ |
| 3DSAM-Adapter \[41] | 114M    | 82.39 | 70.85    | 3.74 | 15.49 |
| SFRFT               | 91M     | 90.39 | 82.54    | 1.83 | 6.71  |


TABLE XIV COMPARISON WITH SAM-MED3D ON LA DATASET.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| --------------------------------------- | ------ | ------ | ----- | -------- | ----- | ----- |
| Method                                  | Prompt | Number | Dice↑ | Jaccard↑ | ASD↓  | 95HD↓ |
| SAM-Med3D \[38]                         | click  | 2      | 57.33 | 42.31    | 8.58  | 28.88 |
|                                         | click  | 5      | 65.12 | 50.29    | 10.65 | 31.17 |
|                                         | click  | 10     | 67.17 | 52.54    | 11.05 | 33.37 |
| SFR (Ours) - scarce annotations (1)     |        |        | 76.76 | 64.65    | 8.54  | 28.07 |
| SFR (Ours) - moderate annotations (20%) |        |        | 90.95 | 83.47    | 1.43  | 6.11  |


1.  Analysis of Initial Pseudo-labels: We compare our fine-tuning module of SFR framework with four foundation model fine-tuning methods, including MedSAM-v1 \[8], MedSAM-v2 \[8], SAMed \[10] and SAMUS \[11]. MedSAM adopts a sub-parts fine-tuning approach and SAMed is a LoRA-based adapter tuning method, which maintains the same parameter scale as the original SAM model. SAMUS introduces a parallel CNN branch and thus has more parameters. As shown in Fig. 1, it demonstrates that our stitching fine-tuning strategy (denoted as  $\mathrm{SFR}_{\mathrm{FT}}$  ) achieves a significant performance improvement without introducing additional parameters, and therefore could provide more reliable pseudo-labels for subsequent retraining module.

<!---->

2.  Analysis of Pseudo-label Error Propagation: In our work, we use fine-tuned foundation model to generate pseudolabels, and the errors in pseudo-labels on unlabeled data could propagate and impact the model's performance. We have considered the potential challenge and implemented some measures to address it effectively. Firstly, within our SFR framework, the semi-supervised retraining module does not solely rely on the pseudo-labels generated by the fine-tuned SAM models. Instead, we incorporate a semi-supervised consistency loss that encourages stable predictions across various perturbations of the input. This helps the model learn consistent and robust features, reducing its sensitivity to minor inaccuracies or noise in the pseudo-labels. Secondly, in the extended version of our framework,  $\mathrm{SFR}^+$  , we introduce confidence estimation to better handle each unlabeled sample. This approach ensures that the model's learning process is guided primarily by more reliable pseudo-labels, reducing the impact of erroneous labels on the overall performance.

3.  Compatibility under Different Number of Labeled Data: With its plug-and-play nature, our re-training module could apply different SSL methods. We conduct re-training experiments on self-training, MT, and ACMT methods with different numbers of labeled samples in Fig. 12. Compared to the baseline, our SFR yields excellent segmentation performance with 1, 2, 4, or 16 labeled samples. Especially in the context of extreme annotation conditions, i.e., 1 labeled sample, our framework produces impressive improvements.

4.  Compared with 3D-based SAM methods: We compare the results of our fine-tuning  $\mathrm{SFR}_{\mathrm{FT}}$  and the existing 3D-based SAM fine-tuning method (3DSAM-Adapter) in Table XIII, which reveals that our method achieves better accuracy without additional computational overhead. Our method outperforms the 3DSAM-Adapter by  $8\%$  . A possible explanation for this performance difference lies in the size of the training parameters. For small-scale medical imaging datasets, particularly in scenarios where training data is limited, a model with a larger number of parameters, which has high complexity, may be prone to overfitting. In contrast, our method effectively utilizes 2D slice stitching input with the low-rank-based fine-tuning strategy. It allows the model to learn spatial relationships while managing computational complexity, which results in better generalization and segmentation performance.

TABLE XV COMPARISON WITH FOUNDATION MODELS EVALUATED BY DICE SCORE.

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| --------------------------------- | ------ | --------- | ----- | ----- | ----- | ----- |
| Method                            | Prompt | Frequency | LA    | BraTS | BTCV  | MACT  |
| SAM 2 \[31]                       | click  | 2         | 31.13 | 25.68 | 55.64 | 77.09 |
|                                   | click  | 5         | 30.73 | 25.08 | 54.07 | 73.12 |
|                                   | click  | 10        | 29.41 | 23.88 | 53.48 | 72.40 |
| Medical SAM 2 \[34]               | click  | 2         | 31.64 | 28.91 | 59.47 | 78.77 |
|                                   | click  | 5         | 33.56 | 28.06 | 56.78 | 76.67 |
|                                   | click  | 10        | 30.38 | 25.85 | 56.27 | 75.42 |
| SFR (Ours) - scarce annotations   |        |           | 76.76 | 78.58 | 53.59 | 49.08 |
| SFR (Ours) - moderate annotations |        |           | 90.95 | 85.81 | 77.06 | 82.87 |


In addition, we conduct a comparison with the SAM-Med3D method \[38], which modifies the SAM paradigm to 3D architecture with training from scratch on 3D medical image datasets. We compare the results of our framework with the prompt-based SAM-Med3D method on the same test data in Table XIV. Notably, on the LA dataset, our SFR framework, using just a single annotation, outperforms SAM-Med3D with 2, 5, or 10-point prompts per sample.

5.  Compared with SAM 2-based methods: We evaluated both SAM 2 \[31] and Medical SAM 2 \[34] across multiple datasets, including LA, BraTS, BTCV, and MACT, under different prompt conditions (2, 5, and 10 clicks), as shown in Table XV. While these models perform well in certain datasets, such as BTCV and MACT, our SFR framework demonstrates significantly superior performance when moderate annotations are provided. Moreover, for the LA and BraTS datasets, the Dice scores of our method, even with scarce annotations (1 annotation), are noticeably higher than the results obtained by both SAM 2 and Medical SAM 2 under any prompt frequency condition. This further underscores the robustness of our approach, particularly when dealing with limited annotations.
6.  Compared with Concatenating along Channel Dimension: By concatenating three slices along the channel dimension to create a pseudo-RGB image, SAM could potentially segment this combined representation. On the other hand, as the SAM model is pre-trained on natural color images, it might interpret the concatenated channels more as color information than as meaningful anatomical context. This presents an intriguing yet challenging question, that is worth exploring further in future research to better evaluate its applicability and uncover potential benefits.
7.  Spatial Structure of Medical Images: During fine-tuning, regarding the distribution gap and dimensionality mismatch, we believe the global scope of all the consecutive slices could be a bridge to link the SAM (designed for 2D natural images) and 3D medical images. As we demonstrated in Section III-B1, our stitching method outperforms the listed alternatives, indicating that in the fine-tuning step, SAM benefits from accurately locating the regions to segment due to its generalization ability in various segmentation tasks. During retraining, we still use 3D structure-based SSL methods, to retain spatial information. In a nutshell, SAM in fine-tuning and SSL in re-training separately treat 3D information in different ways: SAM to solve stitched large-sized 2D slices, and SSL to further segment the 3D volumes.

# V. CONCLUSION

In this work, we present the SFR framework, which consists of the stitching, fine-tuning, and re-training modules, to achieve higher improvements in semi-supervised segmentation tasks by leveraging the foundation model. The stitching module copes with the resolution difference between medical and natural images, and fine-tuning module provides reliable initial pseudo-labels for re-training module. Our framework maintains the same parameter size as the mainstream segmenter, e.g., V-Net \[30], and could be compatible with most popular SSL methods, e.g., Mean Teacher \[21]. In addition, we develop the $\mathrm{SFR}^+$ , which further enhances the framework by introducing confidence estimation and selective training strategy. Extensive experiments demonstrate that the SFR and $\mathrm{SFR}^+$ frameworks improve performance remarkably in both moderate and scarce annotation scenarios.
