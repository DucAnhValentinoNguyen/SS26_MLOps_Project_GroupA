r"""BentoML service for PaliGemma2 ScienceQA (M25, specialized ML deployment).

A thin BentoML wrapper around the same model code the FastAPI app uses
(``load_model`` + ``predict_single``). Chosen over ONNX because exporting a 3B
multimodal ``generate()`` graph is impractical, whereas BentoML serves the
PyTorch model directly and adds batching/packaging on top.

Serve locally:
    CHECKPOINT_PATH=checkpoints/adapter-production \\
      uv run --group serving bentoml serve project_name.bento_service:ScienceQAService

Build a bento (for containerizing / deploying):
    uv run --group serving bentoml build
"""

import os

import bentoml
from PIL import Image as PILImage

# Same env contract as the FastAPI app: a local adapter dir, a .ckpt, or a
# gs:// path is resolved by load_model (the gs:// case is handled in api.py).
_CHECKPOINT = os.environ.get("CHECKPOINT_PATH", "checkpoints/adapter-production")


@bentoml.service(
    name="scienceqa-paligemma2",
    resources={"cpu": "4", "memory": "16Gi"},
    traffic={"timeout": 600},  # cold model load + slow CPU generate
)
class ScienceQAService:
    """Serve single-sample ScienceQA predictions with the fine-tuned adapter."""

    def __init__(self) -> None:
        """Load the model once when the service worker starts."""
        from project_name.predict import load_model

        self.module = load_model(_CHECKPOINT)  # type: ignore[arg-type]

    @bentoml.api
    def predict(
        self,
        image: PILImage.Image,
        question: str,
        choices: list[str],
        hint: str = "",
        lecture: str = "",
        max_new_tokens: int = 10,
    ) -> dict[str, str]:
        """Return the predicted answer letter for one ScienceQA item.

        Args:
            image: The question image (PaliGemma is image-conditioned).
            question: Question text.
            choices: Answer choices (>= 2).
            hint: Optional hint forwarded to the prompt.
            lecture: Optional lecture forwarded to the prompt.
            max_new_tokens: Max tokens to generate.

        Returns:
            A dict ``{"prediction": "A"|"B"|...}``.
        """
        from project_name.predict import predict_single

        prompt_kwargs: dict[str, str] = {}
        if hint:
            prompt_kwargs["hint"] = hint
        if lecture:
            prompt_kwargs["lecture"] = lecture

        letter = predict_single(
            module=self.module,
            image=image.convert("RGB"),
            question=question,
            choices=choices,
            max_new_tokens=max_new_tokens,
            **prompt_kwargs,
        )
        return {"prediction": letter}
