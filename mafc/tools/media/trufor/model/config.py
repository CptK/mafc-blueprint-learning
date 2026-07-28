"""The fixed TruFor test configuration, as a plain object.

Upstream loads `trufor.yaml` through `yacs`. The values never change for
inference, so they are inlined here and `yacs` is dropped as a dependency.
Mirrors test_docker/src/trufor.yaml and the defaults in test_docker/src/config.py.
"""


class Config:
    """Attribute-access config node. Supports `in` because the network checks
    `if 'CONF_BACKBONE' in cfg.MODEL.EXTRA`."""

    def __init__(self, **entries):
        for key, value in entries.items():
            setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"


def default_config() -> Config:
    return Config(
        MODEL=Config(
            NAME="detconfcmx",
            PRETRAINED="",  # empty: weights come from the checkpoint, not from a backbone init
            MODS=("RGB", "NP++"),
            EXTRA=Config(
                BACKBONE="mit_b2",
                DECODER="MLPDecoder",
                DECODER_EMBED_DIM=512,
                PREPRC="imagenet",
                BN_EPS=0.001,
                BN_MOMENTUM=0.1,
                DETECTION="confpool",
                CONF=True,
            ),
        ),
        DATASET=Config(NUM_CLASSES=2),
    )
