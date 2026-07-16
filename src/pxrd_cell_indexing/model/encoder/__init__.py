"""Vendored RealPXRD BertModel encoder and checkpoint loading."""

from pxrd_cell_indexing.model.encoder.bert import BertModel
from pxrd_cell_indexing.model.encoder.histogram import InverseD2HistogramEncoder
from pxrd_cell_indexing.model.encoder.loader import (
    DEFAULT_ENCODER_CONFIG,
    REALPXRD_ENCODER_CHECKPOINT,
    extract_xrd_encoder_state_dict,
    load_xrd_encoder_from_checkpoint,
)
from pxrd_cell_indexing.model.encoder.peak_transformer import PeakGeometryTransformerEncoder
from pxrd_cell_indexing.model.encoder.peak_hist_fusion import PeakTransformerHistogramFusionEncoder
from pxrd_cell_indexing.model.encoder.spectrum_fusion import (
    HistogramSpectrumFusionEncoder,
    SpectrumOnlyEncoder,
)

__all__ = [
    "BertModel",
    "InverseD2HistogramEncoder",
    "PeakGeometryTransformerEncoder",
    "PeakTransformerHistogramFusionEncoder",
    "HistogramSpectrumFusionEncoder",
    "SpectrumOnlyEncoder",
    "DEFAULT_ENCODER_CONFIG",
    "REALPXRD_ENCODER_CHECKPOINT",
    "extract_xrd_encoder_state_dict",
    "load_xrd_encoder_from_checkpoint",
]
