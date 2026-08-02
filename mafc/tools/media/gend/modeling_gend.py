"""GenD deepfake classifier: a frozen vision encoder plus a linear probe.

Vendored from the GenD repository (https://github.com/yermandy/deepfake-detection,
MIT, see LICENSE.txt) at src/hf/modeling_gend.py.

Two changes were needed to run against transformers 5.x, which the upstream
requirements predate (they pin 4.56.2); both are marked COMPAT below. Neither
alters the architecture or the weights — verified via output_loading_info,
which reports no missing, unexpected, or mismatched keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import PretrainedConfig, PreTrainedModel


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes, normalize_inputs=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.normalize_inputs = normalize_inputs

    def forward(self, x: torch.Tensor, **kwargs):
        if self.normalize_inputs:
            x = F.normalize(x, p=2, dim=1)

        return self.linear(x)


class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super().__init__()

        from transformers import CLIPModel, CLIPProcessor

        try:
            self._preprocess = CLIPProcessor.from_pretrained(model_name)
        except Exception:
            self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        # COMPAT(transformers>=5): the outer GenD.from_pretrained builds the
        # model under a meta-device context, and 5.x refuses a nested
        # from_pretrained there. Pin this inner load to CPU; GenDDetector moves
        # the assembled model to its target device afterwards.
        with torch.device("cpu"):
            clip: CLIPModel = CLIPModel.from_pretrained(model_name)

        # take vision model from CLIP, maps image to vision_embed_dim
        self.vision_model = clip.vision_model

        self.model_name = model_name

        self.features_dim = self.vision_model.config.hidden_size

        # take visual_projection, maps vision_embed_dim to projection_dim
        self.visual_projection = clip.visual_projection

    def preprocess(self, image: Image) -> torch.Tensor:
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, preprocessed_images: torch.Tensor) -> torch.Tensor:
        return self.vision_model(preprocessed_images).pooler_output

    def get_features_dim(self):
        return self.features_dim


class DINOEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-with-registers-base"):
        super().__init__()

        from transformers import AutoImageProcessor, AutoModel, Dinov2Model, Dinov2WithRegistersModel

        self._preprocess = AutoImageProcessor.from_pretrained(model_name)
        self.backbone: Dinov2Model | Dinov2WithRegistersModel = AutoModel.from_pretrained(model_name)

        self.features_dim = self.backbone.config.hidden_size

    def preprocess(self, image: Image) -> torch.Tensor:
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs).last_hidden_state[:, 0]

    def get_features_dim(self) -> int:
        return self.features_dim


class PerceptionEncoder(nn.Module):
    def __init__(self, model_name="vit_pe_core_large_patch14_336"):
        super().__init__()

        import timm
        from timm.models.eva import Eva

        self.backbone: Eva = timm.create_model(
            model_name,
            pretrained=True,
            dynamic_img_size=True,
        )

        # Get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.backbone)
        data_config["input_size"] = (3, 224, 224)

        self._preprocess = timm.data.create_transform(**data_config, is_training=False)

        # Remove head
        self.backbone.head = nn.Identity()

        self.features_dim = self.backbone.num_features

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def get_features_dim(self) -> int:
        return self.features_dim


def _repair_position_ids(module: nn.Module) -> None:
    """Rebuild every `position_ids` buffer as a proper arange.

    COMPAT(transformers>=5): `position_ids` is a *non-persistent* buffer — it is
    not in the checkpoint, and is meant to be reconstructed by torch.arange at
    construction time. Because GenD builds CLIP through a nested
    from_pretrained, 5.x's loader materialises these buffers from uninitialised
    memory instead of recomputing them. The result is silently corrupt indices
    like [0, 0, 71, 2, 49527317536, ...].

    This is a genuinely nasty failure: when the garbage happens to land within
    the embedding table it does not raise, it just gathers the wrong positional
    vectors and returns confident nonsense. Only out-of-range garbage surfaces
    as an IndexError. So this must be repaired, never merely caught.
    """
    for submodule in module.modules():
        position_ids = getattr(submodule, "position_ids", None)
        if position_ids is None or not isinstance(position_ids, torch.Tensor):
            continue
        length = position_ids.shape[-1]
        submodule.position_ids = torch.arange(length, device=position_ids.device).expand((1, -1))


class GenDConfig(PretrainedConfig):
    model_type = "GenD"

    def __init__(self, backbone: str = "openai/clip-vit-large-patch14", head: str = "linear", **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.head = head


class GenD(PreTrainedModel):
    config_class = GenDConfig

    def __init__(self, config):
        super().__init__(config)

        self.head = config.head
        self.backbone = config.backbone
        self.config = config

        self._init_feature_extractor()
        self._init_head()

        # COMPAT(transformers>=5): post_init populates all_tied_weights_keys,
        # which 5.x's loader now requires. Optional in 4.x, so upstream omits it.
        self.post_init()

    def _init_feature_extractor(self):
        backbone = self.backbone
        backbone_lowercase = backbone.lower()

        if "clip" in backbone_lowercase:
            self.feature_extractor = CLIPEncoder(backbone)

        elif "vit_pe" in backbone_lowercase:
            self.feature_extractor = PerceptionEncoder(backbone)

        elif "dino" in backbone_lowercase:
            self.feature_extractor = DINOEncoder(backbone)

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _init_head(self):
        features_dim = self.feature_extractor.get_features_dim()

        match self.head:
            case "linear":
                self.model = LinearProbe(features_dim, 2)

            case "LinearNorm":
                self.model = LinearProbe(features_dim, 2, True)

            case _:
                raise ValueError(f"Unknown head: {self.head}")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # COMPAT(transformers>=5): buffers are materialised after __init__ runs,
        # so the position_ids repair has to happen here, once loading is done.
        model = super().from_pretrained(*args, **kwargs)
        _repair_position_ids(model)
        return model

    def forward(self, inputs: torch.Tensor):
        features = self.feature_extractor(inputs)
        outputs = self.model.forward(features)
        return outputs
