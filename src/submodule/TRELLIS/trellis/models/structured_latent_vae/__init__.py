from .encoder import SLatEncoder
from .decoder_gs import SLatGaussianDecoder
from .decoder_rf import SLatRadianceFieldDecoder
from .decoder_mesh import SLatMeshDecoder

# Aliases for FlexiDualGridVae models
FlexiDualGridVaeEncoder = SLatEncoder
FlexiDualGridVaeDecoder = SLatMeshDecoder
