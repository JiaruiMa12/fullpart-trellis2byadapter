from .base_pipeline import BasePipelineConfig, EDMTrainConfig
from .jointdit_single_3d_pipeline import JointDiTSingle3DPipeline, JointDiTSingle3DPipelineConfig

# Stage2 pipeline requires nvdiffrast, make it optional
try:
    from .jointdit_single_3d_pipeline_stage2 import JointDiTSingle3DPipelineStage2, JointDiTSingle3DPipelineConfigStage2
except ImportError as e:
    print(f"[WARNING] Failed to import Stage2 pipeline (requires nvdiffrast): {e}")
    JointDiTSingle3DPipelineStage2 = None
    JointDiTSingle3DPipelineConfigStage2 = None

try:
    from .jointdit_single_3d_pipeline_stage2_trellis2 import JointDiTSingle3DPipelineStage2Trellis2, JointDiTSingle3DPipelineConfigStage2Trellis2
except ImportError as e:
    print(f"[WARNING] Failed to import Stage2 Trellis2 pipeline: {e}")
    JointDiTSingle3DPipelineStage2Trellis2 = None
    JointDiTSingle3DPipelineConfigStage2Trellis2 = None
