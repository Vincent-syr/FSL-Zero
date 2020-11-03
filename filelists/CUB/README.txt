attr_list : 属性的word vector， 来自19CVPR CADA-VAE提供的att_splits.mat文件；
    顺序已经按照.json中重拍
    shape： (200, 312)


attribute file: 属性的0,1表示，来自19ICCV Learning Compositional Representations for Few-Shot Recognition
    file: cub_class_attr_05thresh_freq.dat
    shape: (200, 134) # 从312个attribute中选择了134个，选择方法见19ICCV paper
    # Q：检查attr_file的label顺序是否与 CUB的原始label顺序对齐；
      A：是对齐的