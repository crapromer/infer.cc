#include "../tensor.h"
#include "../utils.h"
#include "infini_infer.h"
#include "infinirt.h"
#include "llama_weights.h"
#include <thread>
#include <vector>

struct DeviceResource
{
    // Device
    DeviceType device;
    unsigned int device_id;
    infiniopHandle_t handle;
    // Weights
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd;
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    // Streams
    infinirtStream_t stream_compute, stream_data, stream_cache;
};

inline DeviceResource
create_device_resource(LlamaMeta const *meta, LlamaWeights const *weights,
                       DeviceType device, unsigned int idev, unsigned int ndev,
                       unsigned int dev_id) {
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle, (Device)device, dev_id);
    infinirtStream_t stream_compute, stream_data, stream_cache;
    infinirtStreamCreate(&stream_compute, device, dev_id);
    infinirtStreamCreate(&stream_data, device, dev_id);
    infinirtStreamCreate(&stream_cache, device, dev_id);
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        w_attn_norm.push_back(
            get_attn_norm(meta, weights, layer, device, dev_id));
        w_attn_qkv.push_back(
            get_attn_qkv(meta, weights, layer, idev, ndev, device, dev_id));
        w_attn_out.push_back(
            get_attn_o(meta, weights, layer, idev, ndev, device, dev_id));
        w_ffn_norm.push_back(
            get_ffn_norm(meta, weights, layer, device, dev_id));
        w_ffn_gate_up.push_back(
            get_ffn_gate_up(meta, weights, layer, idev, ndev, device, dev_id));
        w_ffn_down.push_back(
            get_ffn_down(meta, weights, layer, idev, ndev, device, dev_id));
    }
    return DeviceResource{device,
                          dev_id,
                          handle,
                          get_in_embd(meta, weights, device, dev_id),
                          get_out_norm(meta, weights, device, dev_id),
                          get_out_embd(meta, weights, device, dev_id),
                          w_attn_norm,
                          w_attn_qkv,
                          w_attn_out,
                          w_ffn_norm,
                          w_ffn_gate_up,
                          w_ffn_down,
                          stream_compute,
                          stream_data,
                          stream_cache};
}

struct Model
{
    LlamaMeta meta;
    std::vector<DeviceResource> const dev;
    Model(LlamaMeta const &_meta, std::vector<DeviceResource> const &&_dev)
        : meta(_meta), dev(std::move(_dev)) {}
};

__C struct Model *create_model(LlamaMeta const *meta,
                               LlamaWeights const *weights, DeviceType device,
                               unsigned int ndev, unsigned int const *dev_ids) {
    ASSERT_EQ(meta->nh % ndev, 0);
    ASSERT_EQ(meta->nkvh % ndev, 0);
    ASSERT_EQ(meta->di % ndev, 0);
    auto dev = std::vector<DeviceResource>();
    for (unsigned int i = 0; i < ndev; i++) {
        dev.push_back(
            create_device_resource(meta, weights, device, i, ndev, dev_ids[i]));
    }
    auto model = new Model(*meta, std::move(dev));
    return model;
}

struct KVCache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> k, v;
};

__C struct KVCache *create_kv_cache(struct Model const *model) {
    KVCache *cache = new KVCache();
    auto ndev = model->dev.size();
    auto nkvh = model->meta.nkvh / ndev;
    auto max_len = model->meta.dctx;
    auto dh = model->meta.dh;
    auto shape = std::vector<index_t>{nkvh, max_len, dh};
    for (unsigned int idev = 0; idev < ndev; idev++) {
        auto kcache = std::vector<std::shared_ptr<Tensor>>(model->meta.nlayer);
        auto vcache = std::vector<std::shared_ptr<Tensor>>(model->meta.nlayer);
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            kcache.emplace_back(Tensor::buffer(
                model->meta.dt_mat, shape, model->dev[idev].device,
                model->dev[idev].device_id, model->dev[idev].stream_cache));
            vcache.emplace_back(Tensor::buffer(
                model->meta.dt_mat, shape, model->dev[idev].device,
                model->dev[idev].device_id, model->dev[idev].stream_cache));
        }
        cache->k.push_back(kcache);
        cache->v.push_back(vcache);
    }

    return cache;
}

__C struct KVCache *duplicate_kv_cache(struct Model const *model,
                                       struct KVCache const *kv_cache,
                                       unsigned int seq_len) {
    auto new_kv_cache = create_kv_cache(model);
    auto ndev = model->dev.size();
    for (unsigned int idev = 0; idev < ndev; idev++) {
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            new_kv_cache->k[idev][layer]
                ->slice(1, 0, seq_len)
                ->copy_from(kv_cache->k[idev][layer]->slice(1, 0, seq_len),
                            model->dev[idev].handle,
                            model->dev[idev].stream_cache);

            new_kv_cache->v[idev][layer]
                ->slice(1, 0, seq_len)
                ->copy_from(kv_cache->v[idev][layer]->slice(1, 0, seq_len),
                            model->dev[idev].handle,
                            model->dev[idev].stream_cache);
        }
    }
    return new_kv_cache;
}

__C void drop_kv_cache(struct Model const *, struct KVCache *kv_cache) {
    delete kv_cache;
}

void infer_device(LlamaMeta const &meta, DeviceResource const &rsrc,
                  unsigned int idev, unsigned int ndev, unsigned int ntok,
                  unsigned int const *tokens, unsigned int nreq,
                  unsigned int const *req_lens, unsigned int const *req_pos,
                  std::vector<std::shared_ptr<Tensor>> k_cache,
                  std::vector<std::shared_ptr<Tensor>> v_cache,
                  unsigned int *ans, float temperature, unsigned int topk,
                  float topp) {
    auto nlayer = meta.nlayer;
    auto nkvh = meta.nkvh / ndev;
    auto nh = meta.nh / ndev;
    auto dctx = meta.dctx;
    auto dh = meta.dh;
    auto d = meta.d;
    auto dt_logits = meta.dt_logits;
    auto di = meta.di / ndev;
    auto device = rsrc.device;
    auto device_id = rsrc.device_id;
    auto stream_compute = rsrc.stream_compute;
    auto stream_data = rsrc.stream_data;
    auto stream_cache = rsrc.stream_cache;

    // Allocate buffers
    auto logits_in =
        Tensor::buffer(dt_logits, {ntok, d}, device, device_id, stream_data);
    auto logits_out =
        Tensor::buffer(dt_logits, {ntok, d}, device, device_id, stream_data);
    auto q_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, device, device_id,
                                stream_data);
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, device,
                                      device_id, stream_data);
    auto batch_pos_ids = std::vector<index_t>(ntok);
    index_t max_len = 0;
    index_t req_start = 0;
    for (unsigned int req = 0; req < nreq; req++) {
        for (unsigned int i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
        max_len = std::max(max_len, (index_t)req_lens[req]);
    }
    auto att_scores_buf = Tensor::buffer(dt_logits, {nh * max_len * max_len},
                                         device, device_id, stream_data);

    // Prepare inputs
    if (rsrc.device == DEVICE_CPU) {
        auto pos_ids_buf = Tensor::weight(batch_pos_ids.data(), DATA_TYPE_U64,
                                          {ntok}, rsrc.device, rsrc.device_id);
    } else {
        auto pos_ids_buf = Tensor::buffer(DATA_TYPE_U64, {ntok}, rsrc.device,
                                          rsrc.device_id, rsrc.stream_compute);
        infinirtMemcpyH2DAsync(pos_ids_buf->data(rsrc.stream_compute), device,
                               device_id, batch_pos_ids.data(),
                               sizeof(uint64_t) * ntok, rsrc.stream_compute);
    }

    for (unsigned int layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm

        // qkv_proj

        for (unsigned int req = 0; req < nreq; req++) {
            // self attention
        }

        // o_proj and all_reduce

        // 2. FFN
        // rms_norm

        // mlp

        // all_reduce
    }

    if (idev == 0) {
        // Sample and Output
    }
}

__C void infer(struct Model const *model, unsigned int ntok,
               unsigned int const *tokens, unsigned int nreq,
               unsigned int const *req_lens, unsigned int const *req_pos,
               struct KVCache *kv_caches, unsigned int *ans, float temperature,
               unsigned int topk, float topp) {
    auto ndev = model->dev.size();
    auto threads = std::vector<std::thread>(ndev);
    for (unsigned int idev = 0; idev < ndev; idev++) {
        threads[idev] = std::thread(
            infer_device, model->meta, model->dev[idev], idev, ndev, ntok,
            tokens, nreq, req_lens, req_pos, kv_caches->k[idev],
            kv_caches->v[idev], ans, temperature, topk, topp);
    }
    for (unsigned int idev = 0; idev < ndev; idev++) {
        threads[idev].join();
    }
}

__C void destroy_model(struct Model *model) {
    auto ndev = model->dev.size();
    for (unsigned int i = 0; i < ndev; i++) {
        infiniopDestroyHandle(model->dev[i].handle);
        infinirtStreamDestroy(model->dev[i].stream_compute);
        infinirtStreamDestroy(model->dev[i].stream_data);
        infinirtStreamDestroy(model->dev[i].stream_cache);
    }
    delete model;
}
