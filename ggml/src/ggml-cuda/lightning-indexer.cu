#include "convert.cuh"
#include "lightning-indexer.cuh"

// DeepSeek V3.2/V4 lightning indexer, fused (see ggml_compute_forward_lightning_indexer):
//   dst[i_kv, i_batch, i_stream] = scale_heads *
//       sum_h relu(scale_embd * dot_e(q[e,h,i_batch,i_stream], k[e,i_kv,i_stream])) * w[h,i_batch,i_stream]
// Fusing the reduction over heads avoids materializing the [n_kv, n_head, n_batch]
// score tensor that the unfused graph produces.
//
// One warp per i_kv; the dot product over n_embd is split across the warp lanes and
// finished with a shuffle reduction. q and w for the block's (i_batch, i_stream) are
// staged in shared memory once and reused by every i_kv in the block; the k row is
// held in registers.

#define LID_MAX_EMB_PER_LANE 8 // supports n_embd up to 32*8 = 256

template <typename kv_t>
static __global__ void lightning_indexer_kernel(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ w,
        char       * __restrict__ dst,
        const int     n_embd,
        const int     n_head,
        const int     n_kv,
        const int64_t nb_q_head,    const int64_t nb_q_batch,   const int64_t nb_q_stream,
        const int64_t nb_k_kv,      const int64_t nb_k_stream,
        const int64_t nb_w_batch,   const int64_t nb_w_stream,
        const int64_t nb_dst_batch, const int64_t nb_dst_stream,
        const float scale_embd,
        const float scale_heads) {
    extern __shared__ float smem[];
    float * sq = smem;                   // [n_head*n_embd] q rows for this (i_batch, i_stream)
    float * sw = smem + n_head*n_embd;   // [n_head] indexer weights

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int lane     = threadIdx.x & 31;
    const int warp     = threadIdx.x >> 5;
    const int n_warp   = blockDim.x  >> 5;

    const char * q_base = q + (int64_t) i_batch*nb_q_batch + (int64_t) i_stream*nb_q_stream;
    for (int idx = threadIdx.x; idx < n_head*n_embd; idx += blockDim.x) {
        const int h = idx / n_embd;
        const int e = idx - h*n_embd;
        sq[idx] = *(const float *)(q_base + (int64_t) h*nb_q_head + (int64_t) e*sizeof(float));
    }
    const char * w_base = w + (int64_t) i_batch*nb_w_batch + (int64_t) i_stream*nb_w_stream;
    for (int h = threadIdx.x; h < n_head; h += blockDim.x) {
        sw[h] = *(const float *)(w_base + (int64_t) h*sizeof(float));
    }
    __syncthreads();

    const int i_kv = blockIdx.x*n_warp + warp;
    if (i_kv >= n_kv) {
        return;
    }

    const kv_t * k_row = (const kv_t *)(k + (int64_t) i_kv*nb_k_kv + (int64_t) i_stream*nb_k_stream);
    float kreg[LID_MAX_EMB_PER_LANE];
#pragma unroll
    for (int j = 0; j < LID_MAX_EMB_PER_LANE; ++j) {
        const int e = lane + j*32;
        kreg[j] = e < n_embd ? ggml_cuda_cast<float>(k_row[e]) : 0.0f;
    }

    float score = 0.0f; // meaningful on lane 0
    for (int h = 0; h < n_head; ++h) {
        const float * sqh = sq + h*n_embd;
        float part = 0.0f;
#pragma unroll
        for (int j = 0; j < LID_MAX_EMB_PER_LANE; ++j) {
            const int e = lane + j*32;
            if (e < n_embd) {
                part += sqh[e]*kreg[j];
            }
        }
        part += __shfl_down_sync(0xffffffff, part, 16);
        part += __shfl_down_sync(0xffffffff, part,  8);
        part += __shfl_down_sync(0xffffffff, part,  4);
        part += __shfl_down_sync(0xffffffff, part,  2);
        part += __shfl_down_sync(0xffffffff, part,  1);
        if (lane == 0) {
            score += fmaxf(part*scale_embd, 0.0f)*sw[h];
        }
    }

    if (lane == 0) {
        float * dst_row = (float *)(dst + (int64_t) i_batch*nb_dst_batch + (int64_t) i_stream*nb_dst_stream);
        dst_row[i_kv] = score*scale_heads;
    }
}

void ggml_cuda_op_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];

    GGML_ASSERT(q->type    == GGML_TYPE_F32);
    GGML_ASSERT(w->type    == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(q->nb[0]   == sizeof(float));
    GGML_ASSERT(w->nb[0]   == sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(k->nb[0]   == ggml_type_size(k->type));

    const int n_embd   = q->ne[0];
    const int n_head   = q->ne[1];
    const int n_batch  = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[2];

    GGML_ASSERT(n_embd <= 32*LID_MAX_EMB_PER_LANE);

    const float scale_embd  = ggml_get_op_params_f32(dst, 0);
    const float scale_heads = ggml_get_op_params_f32(dst, 1);

    cudaStream_t stream = ctx.stream();

    const int    n_warp = 8;
    const int    block  = n_warp*32;
    const size_t smem   = ((size_t) n_head*n_embd + n_head)*sizeof(float);
    GGML_ASSERT(smem <= 48*1024); // default per-block shared limit; raise via cudaFuncSetAttribute if exceeded

    const dim3 grid((n_kv + n_warp - 1)/n_warp, n_batch, n_stream);

    const char * q_d = (const char *) q->data;
    const char * k_d = (const char *) k->data;
    const char * w_d = (const char *) w->data;
    char       * d_d = (char       *) dst->data;

    switch (k->type) {
        case GGML_TYPE_F16:
            lightning_indexer_kernel<half><<<grid, block, smem, stream>>>(
                q_d, k_d, w_d, d_d, n_embd, n_head, n_kv,
                q->nb[1], q->nb[2], q->nb[3], k->nb[2], k->nb[3],
                w->nb[1], w->nb[3], dst->nb[1], dst->nb[3], scale_embd, scale_heads);
            break;
        case GGML_TYPE_F32:
            lightning_indexer_kernel<float><<<grid, block, smem, stream>>>(
                q_d, k_d, w_d, d_d, n_embd, n_head, n_kv,
                q->nb[1], q->nb[2], q->nb[3], k->nb[2], k->nb[3],
                w->nb[1], w->nb[3], dst->nb[1], dst->nb[3], scale_embd, scale_heads);
            break;
        default:
            GGML_ABORT("lightning_indexer: unsupported K type\n");
    }
}
