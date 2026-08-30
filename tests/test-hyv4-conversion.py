"""Unit tests for HYV4 (hy_v4_internal) conversion helpers.

Run:  python3 tests/test-hyv4-conversion.py
(pytest is not required; uses unittest.)
"""

import os
import sys
import unittest

# make `conversion` and `gguf` importable when run from the repo root
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "gguf-py"))

import torch  # noqa: E402

from conversion.hyv4 import (  # noqa: E402
    split_kv_b_proj,
    split_gate_up,
)


class TestSplitKVB(unittest.TestCase):
    def test_matches_reference_split(self):
        torch.manual_seed(3)
        n_head, qk_nope, v_head_dim, kv_lora = 4, 6, 10, 7
        W = torch.randn(n_head * (qk_nope + v_head_dim), kv_lora, dtype=torch.float64)

        k_b, v_b = split_kv_b_proj(W, n_head, qk_nope, v_head_dim)
        self.assertEqual(tuple(k_b.shape), (n_head, kv_lora, qk_nope))
        self.assertEqual(tuple(v_b.shape), (n_head, v_head_dim, kv_lora))

        # reference (from conversion/deepseek.py)
        kv_b = W.view(n_head, v_head_dim + qk_nope if False else qk_nope + v_head_dim, kv_lora)
        ref_k, ref_v = torch.split(kv_b, [qk_nope, v_head_dim], dim=1)
        ref_k = ref_k.transpose(1, 2)
        self.assertTrue(torch.equal(k_b, ref_k.contiguous()))
        self.assertTrue(torch.equal(v_b, ref_v.contiguous()))

    def test_bad_shape_asserts(self):
        with self.assertRaises(AssertionError):
            split_kv_b_proj(torch.randn(99, 7), 4, 6, 10)


class TestSplitGateUp(unittest.TestCase):
    def test_gate_first_up_second(self):
        torch.manual_seed(4)
        n_expert, inter, hidden = 3, 5, 4
        gate_ref = torch.randn(n_expert, inter, hidden, dtype=torch.float64)
        up_ref = torch.randn(n_expert, inter, hidden, dtype=torch.float64)
        fused = torch.cat([gate_ref, up_ref], dim=1)  # [E, 2I, H], gate first

        gate, up = split_gate_up(fused, inter)
        self.assertTrue(torch.equal(gate, gate_ref))
        self.assertTrue(torch.equal(up, up_ref))
        self.assertTrue(gate.is_contiguous() and up.is_contiguous())

    def test_bad_shape_asserts(self):
        with self.assertRaises(AssertionError):
            split_gate_up(torch.randn(3, 7, 4), 5)  # 7 != 2*5


class TestModifyTensorsMapping(unittest.TestCase):
    """Exercise HYV4Model.modify_tensors name mapping without a full model init.

    A bare instance (object.__new__) is enough because format_tensor_name only needs the
    class-level model_arch and the gguf tables; modify_tensors otherwise reads self.hparams.
    """

    def _inst(self):
        from conversion.hyv4 import HYV4Model
        inst = object.__new__(HYV4Model)
        inst.hparams = {
            "num_attention_heads": 2,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 2,
            "v_head_dim": 3,
            "kv_lora_rank": 5,
            "moe_intermediate_size": 3,
        }
        return inst

    def _names(self, inst, tensor, name, bid):
        return [nm for nm, _ in inst.modify_tensors(tensor, name, bid)]

    def test_global_names(self):
        inst = self._inst()
        cases = {
            "model.embed_tokens.weight": "token_embd.weight",
            "model.norm.weight": "output_norm.weight",
            "lm_head.weight": "output.weight",
            "model.hc_head.hc_head_fn": "output_hc_fn.weight",
            "model.hc_head.hc_head_base": "output_hc_base.weight",
            "model.hc_head.hc_head_scale": "output_hc_scale.weight",
        }
        for src, want in cases.items():
            self.assertEqual(self._names(inst, torch.zeros(2, 2), src, None), [want], src)

    def test_simple_layer_names(self):
        inst = self._inst()
        cases = {
            "model.layers.3.input_layernorm.weight": "blk.3.attn_norm.weight",
            "model.layers.3.post_attention_layernorm.weight": "blk.3.ffn_norm.weight",
            "model.layers.3.self_attn.q_a_proj.weight": "blk.3.attn_q_a.weight",
            "model.layers.3.self_attn.q_a_layernorm.weight": "blk.3.attn_q_a_norm.weight",
            "model.layers.3.self_attn.kv_a_layernorm.weight": "blk.3.attn_kv_a_norm.weight",
            "model.layers.3.self_attn.o_proj.weight": "blk.3.attn_output.weight",
            "model.layers.3.self_attn.linear_gate.weight": "blk.3.attn_gate.weight",
            "model.layers.3.self_attn.learnable_sink_param": "blk.3.attn_sinks.weight",
            "model.layers.3.hc_attn_layer.hc_pre.hc_fn": "blk.3.hc_attn_fn.weight",
            "model.layers.3.hc_attn_layer.hc_pre.hc_base": "blk.3.hc_attn_base.weight",
            "model.layers.3.hc_attn_layer.hc_pre.hc_scale": "blk.3.hc_attn_scale.weight",
            "model.layers.3.hc_mlp_layer.hc_pre.hc_fn": "blk.3.hc_ffn_fn.weight",
            "model.layers.3.mlp.gate.weight": "blk.3.ffn_gate_inp.weight",
            # base.py renames e_score_correction_bias -> e_score_correction.bias before modify_tensors
            "model.layers.3.mlp.gate.e_score_correction.bias": "blk.3.exp_probs_b.bias",
            "model.layers.3.mlp.gate_proj.weight": "blk.3.ffn_gate.weight",
            "model.layers.3.mlp.up_proj.weight": "blk.3.ffn_up.weight",
            "model.layers.3.mlp.down_proj.weight": "blk.3.ffn_down.weight",
            "model.layers.3.mlp.shared_experts.gate_proj.weight": "blk.3.ffn_gate_shexp.weight",
            "model.layers.3.mlp.shared_experts.up_proj.weight": "blk.3.ffn_up_shexp.weight",
            "model.layers.3.mlp.shared_experts.down_proj.weight": "blk.3.ffn_down_shexp.weight",
        }
        for src, want in cases.items():
            self.assertEqual(self._names(inst, torch.zeros(2, 2), src, 3), [want], src)

    def test_q_b_and_kv_a_names_and_shapes(self):
        inst = self._inst()
        qk_head = 4 + 2
        qb = torch.randn(2 * qk_head, 8)
        out = inst.modify_tensors(qb, "model.layers.1.self_attn.q_b_proj.weight", 1)
        self.assertEqual([nm for nm, _ in out], ["blk.1.attn_q_b.weight"])
        self.assertEqual(tuple(out[0][1].shape), (2 * qk_head, 8))  # shape preserved

        kva = torch.randn(5 + 2, 7)
        out = inst.modify_tensors(kva, "model.layers.1.self_attn.kv_a_proj_with_mqa.weight", 1)
        self.assertEqual([nm for nm, _ in out], ["blk.1.attn_kv_a_mqa.weight"])
        self.assertEqual(tuple(out[0][1].shape), (7, 7))

    def test_kv_b_splits_into_two(self):
        inst = self._inst()
        kvb = torch.randn(2 * (4 + 3), 5)
        out = inst.modify_tensors(kvb, "model.layers.2.self_attn.kv_b_proj.weight", 2)
        self.assertEqual([nm for nm, _ in out], ["blk.2.attn_k_b.weight", "blk.2.attn_v_b.weight"])
        self.assertEqual(tuple(out[0][1].shape), (2, 5, 4))  # k_b [n_head, kv_lora, qk_nope]
        self.assertEqual(tuple(out[1][1].shape), (2, 3, 5))  # v_b [n_head, v_head_dim, kv_lora]

    def test_experts_gate_up_splits_and_down(self):
        inst = self._inst()
        gate_up = torch.randn(4, 2 * 3, 7)  # [n_expert, 2*moe_inter, hidden]
        out = inst.modify_tensors(gate_up, "model.layers.5.mlp.experts.gate_up_proj", 5)
        self.assertEqual([nm for nm, _ in out], ["blk.5.ffn_gate_exps.weight", "blk.5.ffn_up_exps.weight"])
        self.assertEqual(tuple(out[0][1].shape), (4, 3, 7))
        down = torch.randn(4, 7, 3)
        out = inst.modify_tensors(down, "model.layers.5.mlp.experts.down_proj", 5)
        self.assertEqual([nm for nm, _ in out], ["blk.5.ffn_down_exps.weight"])

    def test_unsupported_name_raises(self):
        inst = self._inst()
        with self.assertRaises(ValueError):
            list(inst.modify_tensors(torch.zeros(2, 2), "model.layers.0.self_attn.bogus.weight", 0))


class TestForceQuant(unittest.TestCase):
    """tensor_force_quant: HC *_fn matrices always F32; output F32 only when enable_lm_head_fp32."""

    def _inst(self, enable_lm_head_fp32):
        from conversion.hyv4 import HYV4Model
        inst = object.__new__(HYV4Model)
        inst.hparams = {"enable_lm_head_fp32": enable_lm_head_fp32}
        # attrs the DeepseekV2Model base tensor_force_quant reads when we delegate
        inst._fp8_as_q8 = False
        inst._fp8_dequantized = set()
        return inst

    def test_hc_fn_forced_f32(self):
        import gguf
        inst = self._inst(False)
        for nm in ("blk.0.hc_attn_fn.weight", "blk.0.hc_ffn_fn.weight", "output_hc_fn.weight"):
            self.assertEqual(inst.tensor_force_quant("x", nm, 0, 2), gguf.GGMLQuantizationType.F32, nm)

    def test_output_forced_f32_only_when_flag(self):
        import gguf
        self.assertEqual(
            self._inst(True).tensor_force_quant("lm_head.weight", "output.weight", None, 2),
            gguf.GGMLQuantizationType.F32,
        )
        # flag off -> our branch does not force; base returns the False "no-force" sentinel
        self.assertIs(
            self._inst(False).tensor_force_quant("lm_head.weight", "output.weight", None, 2),
            False,
        )


class TestIndexerTypes(unittest.TestCase):
    """indexer_is_full(): map indexer_types -> per-layer indexer ownership."""

    DSA_HPARAMS = {"index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048}

    def _inst(self, hparams):
        from conversion.hyv4 import HYV4Model
        inst = object.__new__(HYV4Model)
        # DSA hparams are required whenever sparse layers exist; tests focus on indexer_types
        inst.hparams = {**self.DSA_HPARAMS, **hparams}
        return inst

    def test_none_when_no_dsa(self):
        inst = self._inst({"num_hidden_layers": 4, "layer_types": ["full_attention"] * 4})
        self.assertIsNone(inst.indexer_is_full())

    def test_full_shared_pattern(self):
        inst = self._inst({
            "num_hidden_layers": 6,
            "layer_types": ["sparse_attention"] * 6,
            "indexer_types": ["full", "full", "shared", "shared", "full", "shared"],
        })
        self.assertEqual(inst.indexer_is_full(), [True, True, False, False, True, False])

    def test_real_ckpt_pattern(self):
        # the shipped new checkpoint: 78 layers, full at [0,1,5,9,...,77]
        types = ["shared"] * 78
        for i in [0, 1] + list(range(5, 78, 4)):
            types[i] = "full"
        inst = self._inst({
            "num_hidden_layers": 78,
            "layer_types": ["sparse_attention"] * 78,
            "indexer_types": types,
        })
        isf = inst.indexer_is_full()
        self.assertEqual(sum(isf), 21)
        self.assertEqual([i for i, v in enumerate(isf) if v],
                         [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77])

    def test_missing_indexer_types_means_every_sparse_layer_owns_one(self):
        inst = self._inst({
            "num_hidden_layers": 3,
            "layer_types": ["sparse_attention", "full_attention", "sparse_attention"],
        })
        self.assertEqual(inst.indexer_is_full(), [True, False, True])

    def test_rejects_short_indexer_types(self):
        inst = self._inst({
            "num_hidden_layers": 4,
            "layer_types": ["sparse_attention"] * 4,
            "indexer_types": ["full", "shared"],
        })
        with self.assertRaises(ValueError):
            inst.indexer_is_full()

    def test_rejects_unknown_value(self):
        inst = self._inst({
            "num_hidden_layers": 2,
            "layer_types": ["sparse_attention"] * 2,
            "indexer_types": ["full", "bogus"],
        })
        with self.assertRaises(ValueError):
            inst.indexer_is_full()

    def test_rejects_missing_dsa_hparams(self):
        from conversion.hyv4 import HYV4Model
        inst = object.__new__(HYV4Model)
        inst.hparams = {  # sparse layers but no index_* keys
            "num_hidden_layers": 2,
            "layer_types": ["sparse_attention"] * 2,
        }
        with self.assertRaises(ValueError):
            inst.indexer_is_full()

    def test_rejects_short_layer_types(self):
        inst = self._inst({
            "num_hidden_layers": 8,
            "layer_types": ["sparse_attention"] * 3,
        })
        with self.assertRaises(ValueError):
            inst.indexer_is_full()

    def test_rejects_shared_first_layer(self):
        # nothing precedes layer 0, so it cannot share
        inst = self._inst({
            "num_hidden_layers": 2,
            "layer_types": ["sparse_attention"] * 2,
            "indexer_types": ["shared", "full"],
        })
        with self.assertRaises(ValueError):
            inst.indexer_is_full()


class TestIndexerAndMtpTensors(unittest.TestCase):
    def _inst(self):
        from conversion.hyv4 import HYV4Model
        inst = object.__new__(HYV4Model)
        inst.hparams = {
            "num_attention_heads": 2, "qk_nope_head_dim": 4, "qk_rope_head_dim": 2,
            "v_head_dim": 3, "kv_lora_rank": 5, "moe_intermediate_size": 3,
        }
        return inst

    def test_indexer_tensor_names(self):
        inst = self._inst()
        cases = {
            "self_attn.indexer.wq_b.weight": "blk.7.indexer.attn_q_b.weight",
            "self_attn.indexer.wk.weight": "blk.7.indexer.attn_k.weight",
            "self_attn.indexer.k_norm.weight": "blk.7.indexer.k_norm.weight",
            "self_attn.indexer.k_norm.bias": "blk.7.indexer.k_norm.bias",
            "self_attn.indexer.weights_proj.weight": "blk.7.indexer.proj.weight",
        }
        for suffix, expect in cases.items():
            got = [nm for nm, _ in inst.modify_tensors(torch.zeros(2, 2), f"model.layers.7.{suffix}", 7)]
            self.assertEqual(got, [expect], suffix)

    def test_mtp_tensors_dropped_before_load(self):
        # filter_tensors runs before the weight is materialized, so MTP is never read
        from conversion.hyv4 import HYV4Model
        for nm in ("model.mtp_layers.0.eh_proj.weight",
                   "model.mtp_layers.0.self_attn.q_a_proj.weight",
                   "model.mtp_layers.0.self_attn.indexer.wk.weight",
                   "model.mtp_layers.0.mlp.experts.gate_up_proj"):
            self.assertIsNone(HYV4Model.filter_tensors((nm, lambda: None)), nm)

    def test_backbone_tensors_survive_filter(self):
        from conversion.hyv4 import HYV4Model
        for nm in ("model.layers.0.self_attn.q_a_proj.weight",
                   "model.layers.5.self_attn.indexer.wk.weight",
                   "model.embed_tokens.weight"):
            self.assertIsNotNone(HYV4Model.filter_tensors((nm, lambda: None)), nm)

    def test_e_score_bias_still_normalized_by_filter(self):
        # the base filter rewrites ..._bias -> ....bias; our override must not break that
        from conversion.hyv4 import HYV4Model
        out = HYV4Model.filter_tensors(("model.layers.3.mlp.gate.e_score_correction_bias", lambda: None))
        self.assertEqual(out[0], "model.layers.3.mlp.gate.e_score_correction.bias")

    def test_indexer_k_norm_bias_forced_f32(self):
        # k_norm.bias needs our override; k_norm.weight (*_norm.weight) and INDEXER_PROJ are
        # already forced F32 by the base rules in conversion/base.py.
        import gguf
        from conversion.hyv4 import HYV4Model
        inst = object.__new__(HYV4Model)
        inst.hparams = {}
        inst._fp8_as_q8 = False
        inst._fp8_dequantized = set()
        self.assertEqual(inst.tensor_force_quant("x", "blk.0.indexer.k_norm.bias", 0, 1),
                         gguf.GGMLQuantizationType.F32)



if __name__ == "__main__":
    unittest.main(verbosity=2)
