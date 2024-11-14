import ctypes
from ctypes import c_void_p, c_uint, POINTER, c_float
import sys
import os


if len(sys.argv) < 3:
    print("Usage: python test_llama.py [--cpu | --cuda | --cambricon | --ascend] <path/to/model_dir>")
    sys.exit(1)
model_path =  sys.argv[2]
lib_path = os.path.join(os.environ.get("INFINI_ROOT"), "lib", "libinfiniinfer.so")
lib = ctypes.CDLL(lib_path)


class DataType(ctypes.c_int):
    DATA_TYPE_F32 = 0
    DATA_TYPE_F16 = 1
    DATA_TYPE_U64 = 2


class DeviceType(ctypes.c_int):
    DEVICE_TYPE_CPU = 0
    DEVICE_TYPE_CUDA = 1
    DEVICE_TYPE_CAMBRICON = 2
    DEVICE_TYPE_ASCEND = 3

device_type = DeviceType.DEVICE_TYPE_CPU
if sys.argv[1] == "--cpu":
    device_type = DeviceType.DEVICE_TYPE_CPU
elif sys.argv[1] == "--cuda":
    device_type = DeviceType.DEVICE_TYPE_CUDA
elif sys.argv[1] == "--cambricon":
    device_type = DeviceType.DEVICE_TYPE_CAMBRICON
elif sys.argv[1] == "--ascend":
    device_type = DeviceType.DEVICE_TYPE_ASCEND
else:
    print("Usage: python test_llama.py [--cpu | --cuda | --cambricon | --ascend] <path/to/model_dir>")
    sys.exit(1)


class LlamaMeta(ctypes.Structure):
    _fields_ = [
        ("dt_logits", DataType),
        ("dt_norm", DataType),
        ("dt_mat", DataType),
        ("nlayer", c_uint),
        ("d", c_uint),
        ("nh", c_uint),
        ("nkvh", c_uint),
        ("dh", c_uint),
        ("di", c_uint),
        ("dctx", c_uint),
        ("dvoc", c_uint),
        ("epsilon", c_float),
        ("theta", c_float),
    ]


# Define the LlamaWeights struct
class LlamaWeights(ctypes.Structure):
    _fields_ = [
        ("nlayer", c_uint),
        ("input_embd", c_void_p),
        ("output_norm", c_void_p),
        ("output_embd", c_void_p),
        ("attn_norm", POINTER(c_void_p)),
        ("attn_qkv", POINTER(c_void_p)),
        ("attn_o", POINTER(c_void_p)),
        ("ffn_norm", POINTER(c_void_p)),
        ("ffn_gate_up", POINTER(c_void_p)),
        ("ffn_down", POINTER(c_void_p)),
    ]

    def __init__(self, llama):
        import torch

        self.nlayer = llama.config.num_hidden_layers
        state_dict = llama.state_dict()
        self.input_embd = state_dict["model.embed_tokens.weight"].data_ptr()
        self.output_norm = state_dict["model.norm.weight"].data_ptr()
        self.output_embd = state_dict["lm_head.weight"].data_ptr()
        self.attn_norm = (c_void_p * self.nlayer)(
            *[
                state_dict[f"model.layers.{i}.input_layernorm.weight"].data_ptr()
                for i in range(self.nlayer)
            ]
        )
        nh = llama.config.num_attention_heads
        nkvh = llama.config.num_key_value_heads
        dh = llama.config.hidden_size // llama.config.num_attention_heads
        d = llama.config.hidden_size
        self.qkv_tensor = [
            torch.concat(
                [
                    state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
                    .reshape([nh, 2, dh // 2, d])
                    .transpose(1, 2)
                    .reshape(nh * dh, d)
                    .clone(),
                    state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
                    .reshape([nkvh, 2, dh // 2, d])
                    .transpose(1, 2)
                    .reshape(nkvh * dh, d)
                    .clone(),
                    state_dict[f"model.layers.{i}.self_attn.v_proj.weight"],
                ]
            )
            for i in range(self.nlayer)
        ]
        self.attn_qkv = (c_void_p * self.nlayer)(
            *[self.qkv_tensor[i].data_ptr() for i in range(self.nlayer)]
        )
        self.attn_o = (c_void_p * self.nlayer)(
            *[
                state_dict[f"model.layers.{i}.self_attn.o_proj.weight"].data_ptr()
                for i in range(self.nlayer)
            ]
        )
        self.ffn_norm = (c_void_p * self.nlayer)(
            *[
                state_dict[
                    f"model.layers.{i}.post_attention_layernorm.weight"
                ].data_ptr()
                for i in range(self.nlayer)
            ]
        )
        self.gate_up_tensor = [
            torch.concat(
                [
                    state_dict[f"model.layers.{i}.mlp.gate_proj.weight"],
                    state_dict[f"model.layers.{i}.mlp.up_proj.weight"],
                ]
            )
            for i in range(self.nlayer)
        ]

        self.ffn_gate_up = (c_void_p * self.nlayer)(
            *[self.gate_up_tensor[i].data_ptr() for i in range(self.nlayer)]
        )
        self.ffn_down = (c_void_p * self.nlayer)(
            *[
                state_dict[f"model.layers.{i}.mlp.down_proj.weight"].data_ptr()
                for i in range(self.nlayer)
            ]
        )


class Model(ctypes.Structure):
    pass


class KVCache(ctypes.Structure):
    pass


lib.create_model.restype = POINTER(Model)
lib.create_model.argtypes = [
    POINTER(LlamaMeta),  # LlamaMeta const *
    POINTER(LlamaWeights),  # LlamaWeights const *
    DeviceType,  # DeviceType
    c_uint,  # unsigned int ndev
    POINTER(c_uint),  # unsigned int const *dev_ids
]

lib.create_kv_cache.restype = POINTER(KVCache)

lib.infer.restype = None
lib.infer.argtypes = [
    ctypes.POINTER(Model),  # struct Model const *
    c_uint,  # unsigned int ntok
    POINTER(c_uint),  # unsigned int const *tokens
    c_uint,  # unsigned int nreq
    POINTER(c_uint),  # unsigned int const *req_lens
    POINTER(c_uint),  # unsigned int const *req_pos
    POINTER(POINTER(KVCache)),  # struct KVCache **kv_caches
    POINTER(c_uint),  # unsigned int *ans
    c_float,  # float temperature
    c_uint,  # unsigned int topk
    c_float,  # float topp
]


def main():
    ndev = 1
    dev_ids = (c_uint * ndev)(0)

    import torch
    import transformers
    import time

    llama = transformers.LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path
    )

    temperature = 0.7
    topk = 1
    topp = 0.8

    meta = LlamaMeta(
        dt_logits=DataType.DATA_TYPE_F16,
        dt_norm=DataType.DATA_TYPE_F16,
        dt_mat=DataType.DATA_TYPE_F16,
        nlayer=llama.config.num_hidden_layers,
        d=llama.config.hidden_size,
        nh=llama.config.num_attention_heads,
        nkvh=(
            llama.config.num_key_value_heads
            if llama.config.num_key_value_heads
            else llama.config.num_attention_heads
        ),
        dh=llama.config.hidden_size // llama.config.num_attention_heads,
        di=llama.config.intermediate_size,
        dctx=llama.config.max_position_embeddings,
        dvoc=llama.config.vocab_size,
        epsilon=llama.config.rms_norm_eps,
        theta=llama.config.rope_theta,
    )

    weights = LlamaWeights(llama)

    model_instance = lib.create_model(
        ctypes.byref(meta),
        ctypes.byref(weights),
        device_type,
        ndev,
        dev_ids,
    )

    kv_cache = lib.create_kv_cache(model_instance)
    tokens = tokenizer.encode("Once upon a time,")
    ntok = len(tokens)
    nreq = 1

    tokens = (c_uint * ntok)(*tokens)
    req_lens = (c_uint * nreq)(*[ntok])
    req_pos = (c_uint * nreq)(*[0])
    kv_caches = (POINTER(KVCache) * nreq)(*[kv_cache])
    ans = (c_uint * nreq)()

    steps = 500
    start_time = time.time()
    for _step in range(steps):
        lib.infer(
            model_instance,
            ntok,
            tokens,
            nreq,
            req_lens,
            req_pos,
            kv_caches,
            ans,
            temperature,
            topk,
            topp,
        )

        output_tokens = list(ans)
        print(
            tokenizer._tokenizer.id_to_token(output_tokens[0])
            .replace("▁", " ")
            .replace("<0x0A>", "\n"),
            end="",
        )
        req_pos[0] = req_pos[0] + ntok
        ntok = 1
        tokens = (c_uint * ntok)(*output_tokens)
        req_lens = (c_uint * nreq)(*[ntok])

    print("\n")
    end_time = time.time()
    print(f"Time per step: {(end_time - start_time) *  1000 / steps:.3f}ms")

if __name__ == "__main__":
    main()
