"""ESM2 ties lm_head.weight to embed_tokens.weight; ttt_reset() must preserve that.

_ttt_set_state() deep-copies each child module separately, which unties parameters
shared across children.  ESM2 freezes `embed_tokens` during customization
(_ttt_get_frozen_modules), and because `lm_head.weight` *is* that same tensor, the
output projection is frozen too.  If a reset unties them, every ttt() call after the
first trains an extra 33 x embed_dim parameter matrix that the first call did not --
so protein #1 is customized with a different parameter set than proteins #2..N.
"""

import copy

import pytest

torch = pytest.importorskip("torch")
esm = pytest.importorskip("esm")

from proteinttt.models.esm2 import ESM2TTT, DEFAULT_ESM2_35M_TTT_CFG

SEQ_A = (
    "GIHLGELGLLPSTVLAIGYFENLVNIICESLNMLPKLEVSGKEYKKFKFTIVIPKDLDANIKKRAKIYFKQKS"
    "LIEIEIPTSSRNYPIHIQFDENSTDDILHLYDMPTTIGGIDKAIEMFMRKGHIGKTDQQKLLEERELRNFKTT"
    "LENLIATDAFAKEMVEVIIEE"
)
SEQ_B = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSL"
    "AKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSHANVKSAVTRYNDDKLPGLRSFLLDAQT"
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]


def test_ttt_reset_preserves_weight_tying():
    model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().cuda()

    cfg = copy.deepcopy(DEFAULT_ESM2_35M_TTT_CFG)
    cfg.seed = 0
    cfg.steps = 2
    ttt_model = ESM2TTT.ttt_from_pretrained(model, ttt_cfg=cfg)

    assert ttt_model.embed_tokens.weight is ttt_model.lm_head.weight, (
        "pretrained ESM2 is expected to tie the LM head to the token embedding"
    )
    n_params_before = sum(1 for _ in ttt_model.parameters())

    ttt_model.ttt(SEQ_A)
    ttt_model.ttt_reset()

    assert ttt_model.embed_tokens.weight is ttt_model.lm_head.weight, (
        "ttt_reset() untied lm_head.weight from embed_tokens.weight"
    )
    assert sum(1 for _ in ttt_model.parameters()) == n_params_before

    # The consequence the tying exists to prevent: a frozen output projection.
    lm_head_before = ttt_model.lm_head.weight.detach().clone()
    ttt_model.ttt(SEQ_B)
    assert torch.equal(ttt_model.lm_head.weight.detach(), lm_head_before), (
        "lm_head.weight moved during ttt(); it is tied to the frozen embedding and "
        "must stay frozen on every protein, not just the first"
    )
