def load_parameters():
    """
        Loads the defined parameters
    """

    # GPU params
    GPU_ID = '0'
    GPU_MEMORY_FRACTION = 0.9

    # Image pre-processingparameters
    NORMALIZE_IMAGES = True
    MEAN_SUBSTRACTION = False
    DATA_AUGMENTATION = True  # only applied on training set

    # Evaluation params
    EVAL_ON_SETS = ['train', 'val', 'test']  # Possible values: 'train', 'val' and 'test'
    START_EVAL_ON_EPOCH = 1  # First epoch where the model will be evaluated

    # Optimizer parameters (see model.compile() function)
    #OPTIMIZER = 'adadelta'
    #OPTIMIZER = 'sgd'
    OPTIMIZER = 'adam'

    #LR = 1. #0.001  # general LR (0.001 recommended for adam optimizer)
    LR = 0.0001
    PRE_TRAINED_LR_MULTIPLIER = 0.001  # LR multiplier for pre-trained network (LR x PRE_TRAINED_LR_MULTIPLIER)
    NEW_LAST_LR_MULTIPLIER = 1.0  # LR multiplier for the newly added layers (LR x NEW_LAST_LR_MULTIPLIER)
    #LR_DECAY = None
    LR_DECAY = 8
    # Training parameters
    PARALLEL_LOADERS = 1  # parallel data batch loaders
    EPOCHS_FOR_SAVE = 1  # number of epochs between model saves
    WRITE_VALID_SAMPLES = True  # Write valid samples in file


    VERBOSE = 1  # Verbosity
    MODE = 'training'  # 'training' or 'predict' (if 'predict' then RELOAD must be greater than 0 and EVAL_ON_SETS will be used)

    RELOAD = 0  # If 0 start training from scratch, otherwise the model saved on epoch 'RELOAD' will be used


    # ============================================
    parameters = locals().copy()
    return parameters
