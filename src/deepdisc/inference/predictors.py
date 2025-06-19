import deepdisc.astrodet.astrodet as toolkit


class AstroPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """
    def __init__(self, cfg, lazy=False, cfglazy=None, checkpoint=None):
        self.cfg = copy.deepcopy(cfg) # cfg can be modified by model
        
        if "model" in self.cfg: # This is when were using a LazyConfig-style model in the solo config
            self.model = instantiate(self.cfg.model)
            self.model.to(self.cfg.train.device)
            self.model = create_ddp_model(self.model)
            
        else: # This is when we're using a yacs-style model in the solo config (will be specified as "MODEL")
            self.model = build_model(self.cfg)

        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)

        # If we provide AstroPredictor with a checkpoint already loaded in memory
        # just simply load the weights into the model.
        if checkpoint:
            checkpointer._load_model(checkpoint)
        else:
            checkpointer.load(cfg.train.init_checkpoint)
        
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
      
    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions



def get_predictions(dataset_dict, imreader, key_mapper, predictor):
    """Returns indices for matched pairs of ground truth and detected objects in an image

    Parameters
    ----------
    dataset_dict : dictionary
        The dictionary metadata for a single image
    imreader: ImageReader object
        An object derived from ImageReader base class to read the images.
    key_mapper: function
        The key_mapper should take a dataset_dict as input and return the key used by imreader
    predictor: AstroPredictor
        The predictor object used to make predictions on the test set

    Returns
    -------
        matched_gts: list(int)
            The indices of matched objects in the ground truth list
        matched_dts: list(int)
            The indices of matched objects in the detections list
        outputs: list(Intances)
            The list of detected object Instances
    """
    key = key_mapper(dataset_dict)
    img = imreader(key)
    outputs = predictor(img)
    return outputs


def get_predictions_new(dataset_dict, predictor):
    """Returns indices for matched pairs of ground truth and detected objects in an image

    Parameters
    ----------
    dataset_dict : dictionary
        The dictionary metadata for a single image
    predictor: AstroPredictor
        The predictor object used to make predictions on the test set

    Returns
    -------
        outputs: list(Intances)
            The list of detected object Instances
    """

    img = dataset_dict["image_shaped"]
    outputs = predictor(img)
    return outputs
