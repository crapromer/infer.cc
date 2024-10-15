#ifndef __INFINI_INFER_H__
#define __INFINI_INFER_H__

#if defined(_WIN32)
#define __export __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define __export __attribute__((visibility("default")))
#else
#define __export
#endif

#ifdef __cplusplus
#define __C extern "C"
#else
#define __C
#endif

#include <infini_operators.h>
#include "infinirt.h"

typedef enum
{
    DATA_TYPE_F32,
    DATA_TYPE_F16,
    DATA_TYPE_U64,
} DataType;

////////////////// Models //////////////////
typedef struct
{
    DataType dt_logits, dt_norm, dt_mat;
    unsigned int nlayer, d, nh, nkvh, dh, di, dctx, dvoc;
    float epsilon, theta;
} LlamaMeta;

typedef struct
{
    unsigned int nlayer;
    // [dvoc, d]
    void const *input_embd;
    // [d]
    void const *output_norm;
    // [dvoc, d]
    void const *output_embd;
    // nlayer * [d]
    void const *const *attn_norm;
    // nlayer * [ndev, (nh + 2 * nkvh) / ndev * dh, d]
    void const *const *attn_qkv;
    // nlayer * [ndev, d, nkvh / ndev * dh]
    void const *const *attn_o;
    // nlayer * [d]
    void const *const *ffn_norm;
    // nlayer * [ndev, 2 * di / ndev, d]
    void const *const *ffn_gate_up;
    // nlayer * [ndev, d, di / ndev]
    void const *const *ffn_down;
} LlamaWeights;

//////////////////// APIs ///////////////////////
/// @brief 创建模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct Model *
create_model(LlamaMeta const *,
             LlamaWeights const *,
             DeviceType device,
             unsigned int ndev,
             unsigned int const *dev_ids);

/// @brief 创建 KV Cache
__C __export struct KVCache *
create_kv_cache(struct Model const *);

/// @brief 复制 KV Cache
__C __export struct KVCache *
duplicate_kv_cache(struct Model const *,
                   struct KVCache const *, unsigned int seq_len);

/// @brief 销毁 KV Cache
__C __export void
drop_kv_cache(struct Model const *,
              struct KVCache *);

/// @brief 推理
/// @param ntok 输入 token 总数
/// @param tokens 输入 token
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param ans 每个请求的输出 token
/// @param temperature 采样温度（0. 表示贪心采样）
/// @param topk 采样 topk（1 表示贪心采样）
/// @param topp 采样 topp
__C __export void
infer(struct Model const *,
      unsigned int ntok, unsigned int const *tokens,
      unsigned int nreq, unsigned int const *req_lens, unsigned int const *req_pos,
      struct KVCache *kv_caches, unsigned int *ans,
      float temperature, unsigned int topk, float topp);

/// @brief 销毁模型
__C __export void
destroy_model(struct Model *);

#endif // __INFINI_INFER_H__
