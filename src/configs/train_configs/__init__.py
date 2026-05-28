from .personal_configs_part import personal_configs_part
from .personal_configs_part_stage2 import personal_configs_part_s2
from .personal_configs_part_trellis2 import personal_configs_part_trellis2
from .personal_configs_global_trellis2_adapter import personal_configs_global_trellis2_adapter


train_configs = {
    **personal_configs_part,
    **personal_configs_part_s2,
    **personal_configs_part_trellis2,
    **personal_configs_global_trellis2_adapter,
}
