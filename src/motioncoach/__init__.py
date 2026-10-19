"""motioncoach: motion comparison engine + application layers (Baduanjin MVP).

Layering (see docs/ARCHITECTURE.md):
    engine/      platform-independent numpy code: data contract, features, alignment, comparison
    perception/  pose-model adapters that output engine.PoseSequence
    media/       video probing, normalisation (browser proxy), sampled decoding
    apps/        application rules (profiles, references, report wording)
    service/     FastAPI job service + static web client
"""

__version__ = "0.1.0"
