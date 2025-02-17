class Configs:
    def __init__(self):
        self.device = "cuda:0"             # device
        self.train_batch_size = 32         # Batch size
        self.task_name = 'reconstruction'  # Example task name
        self.seq_len = 100                 # Sequence length
        self.freq_seq_len = 50             # Frequency sequence length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 256                 # Model final dimension
        self.patch_d_model = 64            # Patch encode dimension
        self.embed = 'timeF'               # Time encoding method
        self.freq = 'h'                    # Time frequency
        self.joint_train = False           # Joint training or not
        self.num_subjects = 10             # num_subjects
        self.dropout = 0.25                # Dropout rate
        self.factor = 1                    # Attention scaling factor
        self.n_heads = 4                   # Number of attention heads
        self.encode_layers = 2             # Number of encoder layers
        self.d_ff = 256                    # Dimension of the feedforward network
        self.activation = 'gelu'           # Activation function
        self.enc_in = 63                   # Encoder input dimension (example value)
        self.padding_patch = 'end'         # Padding Path help to be modified to general case
        self.patch_len = 10                # Patch len
        self.patch_stride = 4              # Patch stride
        self.num_train_epochs = 2          # Train epochs number
        self.learning_rate = 1e-4          # Learning rate to use