#include "../tensor.h"
#include "infini_infer.h"
#include <cmath>
inline std::shared_ptr<Tensor> get_in_embd(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    DeviceType device, unsigned int device_id)
{
    auto shape = std::vector<index_t>({meta->dvoc, meta->d});
    return Tensor::weight((char *)w->input_embd, meta->dt_logits, shape, device,
                          device_id);
}

inline std::shared_ptr<Tensor> get_out_norm(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    DeviceType device, unsigned int device_id)
{
    auto shape = std::vector<index_t>({meta->d});
    return Tensor::weight((char *)w->output_norm, meta->dt_norm, shape, device,
                          device_id);
}

inline std::shared_ptr<Tensor> get_out_embd(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    DeviceType device, unsigned int device_id)
{
    auto shape = std::vector<index_t>({meta->dvoc, meta->d});
    return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape,
                          device, device_id)
        ->permute({1, 0});
}

inline std::shared_ptr<Tensor> get_attn_norm(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    size_t layer,
    DeviceType device, unsigned int device_id)
{
    auto shape = std::vector<index_t>({meta->d});
    return Tensor::weight((char *)(w->attn_norm[layer]), meta->dt_norm, shape, device, device_id);
}

inline std::shared_ptr<Tensor> get_attn_qkv(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    size_t layer, size_t idev, size_t ndev,
    DeviceType device, unsigned int device_id)
{
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * d;
    auto shape = std::vector<index_t>({(nh + 2 * nkvh) / ndev * dh, d});
    return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, meta->dt_mat,
                          shape, device, device_id)
        ->permute({1, 0});
}

inline std::shared_ptr<Tensor> get_attn_o(LlamaMeta const *meta,
                                          LlamaWeights const *w, size_t layer,
                                          size_t idev, size_t ndev,
                                          DeviceType device,
                                          unsigned int device_id) {
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * d * (nh / ndev * dh);
    auto shape = std::vector<index_t>({d, nh / ndev * dh});
    return Tensor::weight((char *)(w->attn_o[layer]) + offset, meta->dt_mat,
                          shape, device, device_id)
        ->permute({1, 0});
}

inline std::shared_ptr<Tensor> get_ffn_norm(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    size_t layer,
    DeviceType device, unsigned int device_id)
{
    auto shape = std::vector<index_t>({meta->d});
    return Tensor::weight((char *)(w->ffn_norm[layer]), meta->dt_norm, shape, device, device_id);
}

inline std::shared_ptr<Tensor> get_ffn_gate_up(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    size_t layer, size_t idev, size_t ndev,
    DeviceType device, unsigned int device_id)
{
    auto di = meta->di;
    auto d = meta->d;
    size_t offset = idev * (2 * di / ndev) * d;
    auto shape = std::vector<index_t>({2 * di / ndev, d});
    return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset,
                          meta->dt_mat, shape, device, device_id)
        ->permute({1, 0});
}

inline std::shared_ptr<Tensor> get_ffn_down(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    size_t layer, size_t idev, size_t ndev,
    DeviceType device, unsigned int device_id)
{
    auto di = meta->di;
    auto d = meta->d;
    size_t offset = idev * d * (di / ndev);
    auto shape = std::vector<index_t>({d, di / ndev});
    return Tensor::weight((char *)(w->ffn_down[layer]) + offset, meta->dt_mat,
                          shape, device, device_id)
        ->permute({1, 0});
}

inline std::shared_ptr<Tensor> get_sin_table(LlamaMeta const *meta,
                                             DeviceType device,
                                             unsigned int device_id) {
    float *table = (float *)std::malloc(meta->dctx * meta->dh * sizeof(float));
    auto half_dh = meta->dh / 2;
    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) /
                std::pow(meta->theta, static_cast<float>(j) / half_dh));
            table[i * meta->dh + 2 * j] = _sin;
            table[i * meta->dh + 2 * j + 1] = _sin;
        }
    }
    auto shape = std::vector<index_t>({meta->dctx, meta->dh});
    auto tensor = Tensor::weight(table, INFINI_F32, shape, device, device_id);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> get_cos_table(LlamaMeta const *meta,
                                             DeviceType device,
                                             unsigned int device_id) {
    float *table = (float *)std::malloc(meta->dctx * meta->dh * sizeof(float));
    auto half_dh = meta->dh / 2;
    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) /
                std::pow(meta->theta, static_cast<float>(j) / half_dh));
            table[i * meta->dh + 2 * j] = _cos;
            table[i * meta->dh + 2 * j + 1] = _cos;
        }
    }
    auto shape = std::vector<index_t>({meta->dctx, meta->dh});
    auto tensor = Tensor::weight(table, INFINI_F32, shape, device, device_id);
    std::free(table);
    return tensor;
}
