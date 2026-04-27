import torch
class HeadArgs:
    def __init__(self, number_of_heads=4,  enc_layers=6, dec_layers=6):
        # Learning parameters
        self.lr = 1e-4
        self.lr_backbone = 1e-5
        self.batch_size = 2
        self.weight_decay = 1e-4
        self.epochs = 300
        self.lr_drop = 200
        self.clip_max_norm = 0.1
        
        # Model parameters
        self.frozen_weights = None
        self.backbone = 'resnet50'
        self.dilation = False
        self.position_embedding = 'sine'
        
        # Transformer parameters - Your specified values
        self.head_dim = 32
        self.hidden_dim = number_of_heads * self.head_dim  # 128
        self.dropout = 0.1
        self.nheads = number_of_heads
        #self.effective_heads = 4  # Custom parameter
        self.head_dim = self.hidden_dim // self.nheads  # 32
        self.dim_feedforward = 8*self.hidden_dim  # 1024
        self.activation = 'relu'
        
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.num_queries = 100
        self.pre_norm = False
        
        # Segmentation
        self.masks = False
        
        # Loss parameters
        self.aux_loss = True
        self.set_cost_class = 1
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.mask_loss_coef = 1
        self.dice_loss_coef = 1
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.eos_coef = 0.1
        
        # Dataset parameters
        self.dataset_file = 'coco'
        self.coco_path = 'C:\\workspace\\ml\\data\\coco'
        self.coco_panoptic_path = None
        self.remove_difficult = False
        
        # System parameters
        self.output_dir = ''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 42
        self.resume = ''
        self.start_epoch = 0
        self.eval = True
        self.num_workers = 2
        
        # Distributed training
        self.world_size = 1
        self.dist_url = 'env://'
        self.distributed = False
        self.gpu = None


import torch
class LayerArgs:
    def __init__(self, number_of_heads=4,  n_layer=6):
        # Learning parameters
        self.lr = 1e-4
        self.lr_backbone = 1e-5
        self.batch_size = 2
        self.weight_decay = 1e-4
        self.epochs = 300
        self.lr_drop = 200
        self.clip_max_norm = 0.1
        
        # Model parameters
        self.frozen_weights = None
        self.backbone = 'resnet50'
        self.dilation = False
        self.position_embedding = 'sine'
        
        # Transformer parameters - Your specified values
        self.head_dim = 32
        self.hidden_dim = number_of_heads * self.head_dim  # 128
        self.dropout = 0.1
        self.nheads = number_of_heads
        #self.effective_heads = 4  # Custom parameter
        self.head_dim = self.hidden_dim // self.nheads  # 32
        self.dim_feedforward = 8*self.hidden_dim  # 1024
        self.activation = 'relu'
        
        self.enc_layers = n_layer
        self.dec_layers = n_layer
        self.num_queries = 100
        self.pre_norm = False
        
        # Segmentation
        self.masks = False
        
        # Loss parameters
        self.aux_loss = True
        self.set_cost_class = 1
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.mask_loss_coef = 1
        self.dice_loss_coef = 1
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.eos_coef = 0.1
        
        # Dataset parameters
        self.dataset_file = 'coco'
        self.coco_path = 'C:\\workspace\\ml\\data\\coco'
        self.coco_panoptic_path = None
        self.remove_difficult = False
        
        # System parameters
        self.output_dir = ''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 42
        self.resume = ''
        self.start_epoch = 0
        self.eval = True
        self.num_workers = 2
        
        # Distributed training
        self.world_size = 1
        self.dist_url = 'env://'
        self.distributed = False
        self.gpu = None
