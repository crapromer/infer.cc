#include "infini_infer.h"
#include "infinirt.h"
#include <vector>

struct DeviceResource{
    // Device
    DeviceType device;
    unsigned int id;
    infiniopHandle_t handle;
    // Weights
    Tensor w_in_embd, w_out_norm, w_out_embd;
    std::vector<Tensor>
        w_attn_norm,
        w_attn_qkv,
        w_attn_out,
        w_ffn_norm,
        w_ffn_gate_up,
        w_ffn_down;
    // Streams
    infiniStream_t stream_compute, stream_data;
};

struct Model
{
    LlamaMeta meta;
    std::vector<DeviceResource> const dev;
};

__C struct Model *
create_model(LlamaMeta const *,
             LlamaWeights const *,
             DeviceType device,
             unsigned int ndev,
             unsigned int const *dev_ids)
{
    // TODO: Implement this function
    return nullptr;
}

__C struct KVCache *
create_kv_cache(struct Model const *)
{
    // TODO: Implement this function
    return nullptr;
}

__C struct KVCache *
duplicate_kv_cache(struct Model const *,
                   struct KVCache const *, unsigned int seq_len)
{
    // TODO: Implement this function
    return nullptr;
}

__C void drop_kv_cache(struct Model const *,
                       struct KVCache *)
{
    // TODO: Implement this function
}

__C void infer(struct Model const *,
               unsigned int ntok, unsigned int const *tokens,
               unsigned int nreq, unsigned int const *req_lens, unsigned int const *req_pos,
               struct KVCache *kv_caches, unsigned int *ans,
               float temperature, unsigned int topk, float topp)
{
    // TODO: Implement this function
}

__C void destroy_model(struct Model *){
    // TODO: Implement this function
}
