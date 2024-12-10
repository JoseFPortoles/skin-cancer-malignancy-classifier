### Important Disclaimer

**This project is for educational and research purposes only.**  
The code provided in this repository is not validated for clinical or diagnostic use. It is intended solely for exploring image classification and segmentation techniques, and should **NOT** be used to make medical decisions or diagnoses.  

The author(s) take no responsibility for any misuse of this code, and it is strictly prohibited to use it in any clinical or real-world medical setting.

If you are looking for tools for clinical or diagnostic purposes, please consult certified medical software and professionals.


# skin-cancer-malignancy-classifier

**First of all**, this project is an exercise inspired on [1] and [2], intended primarily to help me explore deep learning techniques for image classification and segmentation and their interplay.

# Description

The basic architecture is a binary classifier network for malignancy prediction. It consists in two subnets that work in sequence:

* __Segmenter__: This is a classic U-Net semantic segmentation network widely used in medical imaging, as previously implemented here: https://github.com/JoseFPortoles/U-Net.
* __Classifier__: For this part the network of choice was a ResNet-50.

A similar (although more sofisticated) architecture can be seen for instance in [2], with some important differences, including that the authors train on datasets that combine the predicted label with segmentation annotations which allow them to introduce a segmentation term to the total loss, and that they use a categorical network approach on top of their classifier.  

The approach adopted here starts by pretraining the segmenter with HAM10000, an annotated public dataset for skin injury binary segmentation [3]. As such HAM10000 provide binary masks (injury vs healthy skin) as annotations. In the first instance, the role of the segmenter in the combined network is to filter out areas of the image containing normal, healthy skin, allowing only affected areas to pass through to the classifier. Under this scheme, it would make sense to freeze the weights of the pretrained segmenter during training and simply train the classifier on top of it. However, I chose not to freeze the weights, effectively allowing the segmenter and classifier to learn together. The reasoning behind this approach is that the segmenter learns to highlight the areas most relevant for the classifier to perform its diagnostic task, effectively becoming more of an attention mechanism than a purely binary segmenter.

This is still work in progress and the preferred approach could still change at a later stage.

# Bibliography

[1] Nicholas Kurtansky, Veronica Rotemberg, Maura Gillis, Kivanc Kose, Walter Reade, and Ashley Chow. ISIC 2024 - Skin Cancer Detection with 3D-TBP. https://kaggle.com/competitions/isic-2024-challenge, 2024. Kaggle.

[2] M.Khurshid et al. "Multi-task Explainable Skin Lesion Classification", arXiv:2310.07209v1.

[3] Tschandl, P., Rinner, C., Apalla, Z. et al. Human–computer collaboration for skin cancer recognition. Nat Med 26, 1229–1234 (2020). https://doi.org/10.1038/s41591-020-0942-0