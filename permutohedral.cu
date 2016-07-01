/*!
 * Copyright (c) 2016 by Contributors
 * \file permutohedral.cu
 * \brief
 * \author Junyuan Xie
*/

#include "./permutohedral-inl.h"
#include "cu_hash_table.h"

namespace mxnet {
namespace op {

namespace permutohedral {

template<int key_size>
__global__ void init(CuHashTable<key_size> table,
                     const int n_elements,
                     const float *pos,
                     const float *scale,
                     Pair *matrix) {
  float elevated[key_size+1];
  int greedy[key_size+1];
  int rank[key_size+1];
  float barycentric[key_size+2];
  short key[key_size];

  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_elements) return;

  float sm = 0;
  for (int i = key_size; i > 0; i--) {
    float cf = pos[(i-1)*n_elements + idx]*scale[i-1];
    elevated[i] = sm - i*cf;
    sm += cf;
  }
  elevated[0] = sm;  
    
  // find the closest zero-colored lattice point

  // greedily search for the closest zero-colored lattice point
  short sum = 0;
  for (int i = 0; i <= key_size; i++) {
    float v = elevated[i]*(1.0f/(key_size+1));
    float up = ceilf(v) * (key_size+1);
    float down = floorf(v) * (key_size+1);
    if (up - elevated[i] < elevated[i] - down) {
      greedy[i] = static_cast<short>(up);
    } else {
      greedy[i] = static_cast<short>(down);
    }
    sum += greedy[i];
  }
  sum /= key_size+1;
  
  // sort differential to find the permutation between this simplex and the canonical one
  for (int i = 0; i <= key_size; i++) {
    rank[i] = 0;
    for (int j = 0; j <= key_size; j++) {
      if (elevated[i] - greedy[i] < elevated[j] - greedy[j] ||
          (elevated[i] - greedy[i] == elevated[j] - greedy[j]
           && i > j)) {
        rank[i]++;
      }
    }
  }
  
  if (sum > 0) { // sum too large, need to bring down the ones with the smallest differential
    for (int i = 0; i <= key_size; i++) {
      if (rank[i] >= key_size + 1 - sum) {
        greedy[i] -= key_size+1;
        rank[i] += sum - (key_size+1);
      } else {
        rank[i] += sum;
      }
    }
  } else if (sum < 0) { // sum too small, need to bring up the ones with largest differential
    for (int i = 0; i <= key_size; i++) {
      if (rank[i] < -sum) {
        greedy[i] += key_size+1;
        rank[i] += (key_size+1) + sum;
      } else {
        rank[i] += sum;
      }
    }
  }

  // turn delta into barycentric coords
  for (int i = 0; i <= key_size+1; i++) {
      barycentric[i] = 0;
  }
  
  for (int i = 0; i <= key_size; i++) {
    float delta = (elevated[i] - greedy[i]) * (1.0f/(key_size+1));
    barycentric[key_size-rank[i]] += delta;
    barycentric[key_size+1-rank[i]] -= delta;
  }
  barycentric[0] += 1.0f + barycentric[key_size+1];

  for (int color = 0; color <= key_size; color++) {
    // Compute the location of the lattice point explicitly (all but
    // the last coordinate - it's redundant because they sum to zero)
    for (int i = 0; i < key_size; i++) {
      key[i] = greedy[i] + color;
      if (rank[i] > key_size-color) key[i] -= (key_size+1);
    }

    Pair r;
    r.index = table.insert(key, idx*(key_size+1)+color);
    r.weight = barycentric[color];
    matrix[idx*(key_size+1) + color] = r;
  }
}

template<int key_size, bool norm>
__global__ void splat(CuHashTable<key_size> table,
                      const int32_t n_elements,
                      const int32_t val_size,
                      float *data,
                      float *val,
                      Pair *matrix) {
  const int idx = threadIdx.y + blockIdx.y * blockDim.y;
  if (idx >= n_elements) return;
  const int color = threadIdx.x;

  Pair r = matrix[idx*(key_size+1)+color];
  float *dst = val + r.index*val_size;
  if (!norm) {
    for (int j = 0; j < val_size; j++) {
      atomicAdd(dst+j, data[j*n_elements + idx]*r.weight);
    }
  } else {
    for (int j = 0; j < val_size-1; j++) {
      atomicAdd(dst+j, data[j*n_elements + idx]*r.weight);
    }
    atomicAdd(dst+val_size-1, 1.f*r.weight);
  }
}


template<int key_size>
__global__ static void blur(CuHashTable<key_size> table,
                            const int32_t val_size,
                            const int32_t color,
                            float *val,
                            float *new_val, 
                            Pair *matrix) {
  short key[key_size+1];
  short np[key_size+1];
  short nm[key_size+1];
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= table.n_keys_) return;

  // Check if I'm valid
  if (matrix[idx].index != idx) return;

  // find my key and the keys of my neighbours

  for (int i = 0; i < key_size; i++) {
    key[i] = table.keys_[idx*key_size+i];
    np[i] = key[i]+1;    
    nm[i] = key[i]-1;
  }

  np[color] -= key_size+1;
  nm[color] += key_size+1;

  int offNp = table.find(np);
  int offNm = table.find(nm);

  float *valMe = val + val_size*idx;
  float *valNp = val + val_size*offNp;
  float *valNm = val + val_size*offNm; 
  float *valOut = new_val + val_size*idx;

  for (int i = 0; i < val_size; i++) {
    float o = valMe[i];
    if (offNp >= 0) o += 0.5f*valNp[i];
    if (offNm >= 0) o += 0.5f*valNm[i];
    valOut[i] = o;
  }
}

template<int key_size, bool norm>
__global__ void slice(CuHashTable<key_size> table,
                      const int32_t n_elements,
                      const int32_t val_size,
                      float *val,
                      float *out,
                      Pair *matrix) {
  const float alpha = 1.0f / (1+powf(2, -key_size));
  int32_t index[key_size+1];
  float weight[key_size+1];

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;    
  if (idx >= n_elements) return;

  for (int i = 0; i <= key_size; ++i) {
    Pair r = matrix[idx*(key_size+1) + i];
    index[i] = r.index;
    weight[i] = r.weight;
  }

  if (!norm) {
    for (int j = 0; j < val_size; ++j) {
      float v = 0.0f;
      for (int i = 0; i <= key_size; ++i) {
        v += weight[i]*val[index[i]*val_size + j];
      }
      out[j*n_elements + idx] = v * alpha;
    }
  } else {
    float n = 0.0f;
    for (int i = 0; i <= key_size; ++i) {
      n += weight[i]*val[index[i]*val_size + val_size - 1];
    }
    n = 1.0f/n;
    for (int j = 0; j < val_size-1; ++j) {
      float v = 0.0f;
      for (int i = 0; i <= key_size; ++i) {
        v += weight[i]*val[index[i]*val_size + j];
      }
      out[j*n_elements + idx] = v * n;
    }
  }
}

}


template<int key_size>
void CuPermutohedralOp<key_size>::GetTempSpace(const OpContext &ctx) {
  using namespace mshadow;
  using namespace permutohedral;
  Stream<gpu> *s = ctx.get_stream<gpu>();

  Tensor<gpu, 1, uint8_t> tmp =
    ctx.requested[kTemp].get_space_typed<gpu, 1, uint8_t>(
      Shape1(n_keys_*2*sizeof(int32_t) +
             n_keys_*key_size*sizeof(int16_t) +
             n_keys_*val_size_*sizeof(float) +
             n_keys_*val_size_*sizeof(float) +
             n_keys_*sizeof(Pair)), s);
  uint8_t *ptr = tmp.dptr_;

  int32_t *entries = (int32_t*)ptr;
  entries_ = Tensor<gpu, 1, int32_t>(entries, Shape1(n_keys_*2), s);
  ptr += n_keys_*2*sizeof(int32_t);

  int16_t *keys = (int16_t*)ptr;
  keys_ = Tensor<gpu, 2, int16_t>(keys, Shape2(key_size, n_keys_), s);
  ptr += n_keys_*key_size*sizeof(int16_t);

  float *vals = (float*)ptr;
  vals_ = Tensor<gpu, 2, float>(vals, Shape2(val_size_, n_keys_), s);
  ptr += n_keys_*val_size_*sizeof(float);

  float *new_vals = (float*)ptr;
  new_vals_ = Tensor<gpu, 2, float>(new_vals, Shape2(val_size_, n_keys_), s);
  ptr += n_keys_*val_size_*sizeof(float);

  Pair *matrix = (Pair*)ptr;
  matrix_ = Tensor<gpu, 1, Pair>(matrix, Shape1(n_keys_), s);
  ptr += n_keys_*sizeof(Pair);

  CHECK_EQ(ptr, tmp.dptr_ + tmp.shape_.Size());
}

template<int key_size>
void CuPermutohedralOp<key_size>::Forward(const OpContext &ctx,
                                   const std::vector<TBlob> &in_data,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<TBlob> &out_data,
                                   const std::vector<TBlob> &aux_args)  {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace permutohedral;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  cudaStream_t stream = Stream<gpu>::GetStream(s);

  Tensor<gpu, 1, float> scale = aux_args[kScale].get<gpu, 1, float>(s);

  if (!init_) {
    TShape data_shape = in_data[kData].shape_;
    batch_size_ = data_shape[0];
    data_size_ = data_shape[1];
    if (param_.normalize) {
      val_size_ = data_size_ + 1;
    } else {
      val_size_ = data_size_;
    }
    n_elements_ = data_shape.Size()/batch_size_/data_size_;
    n_keys_ = n_elements_*(key_size+1);
    CHECK_EQ(in_data[kPos].size(1), key_size);
    
    lblock_ = cuda::kBaseThreadNum;
    nblock_ = (n_elements_-1)/lblock_+1;

    float cpu_scale[key_size];
    for (int i = 0; i < key_size; i++) {
      cpu_scale[i] = (key_size+1)*sqrtf((2.0/3.0)/((i+1)*(i+2)));
    }
    CHECK_EQ(cudaMemcpyAsync((void*)scale.dptr_, (void*)cpu_scale, key_size*sizeof(float), cudaMemcpyHostToDevice, stream), cudaSuccess);

    init_ = true;
  }
  

  Shape<3> shape = Shape3(batch_size_, data_size_, n_elements_); 
  Tensor<gpu, 3, float> in = in_data[kData].get_with_shape<gpu, 3, float>(shape, s);
  Tensor<gpu, 3, float> out = out_data[kOut].get_with_shape<gpu, 3, float>(shape, s);
  shape[1] = key_size;
  Tensor<gpu, 3, float> pos = in_data[kPos].get_with_shape<gpu, 3, float>(shape, s);

  GetTempSpace(ctx);

  CuHashTable<key_size> table(n_keys_, entries_.dptr_, keys_.dptr_);


  for (int i = 0; i < batch_size_; ++i) {
    entries_ = -1;
    vals_ = 0;

    init<key_size><<<dim3(nblock_, 1, 1), dim3(lblock_,1,1), 0, stream>>>(
      table, n_elements_, pos.dptr_ + i*key_size*n_elements_, scale.dptr_, matrix_.dptr_);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    if (param_.normalize) {
      splat<key_size, true><<<dim3(1, (n_elements_-1)/(lblock_/(key_size+1))+1, 1), dim3(key_size+1, lblock_/(key_size+1), 1), 0, stream>>>(
        table, n_elements_, val_size_, in.dptr_+i*data_size_*n_elements_, vals_.dptr_, matrix_.dptr_);
    } else {
      splat<key_size, false><<<dim3(1, (n_elements_-1)/(lblock_/(key_size+1))+1, 1), dim3(key_size+1, lblock_/(key_size+1), 1), 0, stream>>>(
        table, n_elements_, val_size_, in.dptr_+i*data_size_*n_elements_, vals_.dptr_, matrix_.dptr_);
    }
    CHECK_EQ(cudaGetLastError(), cudaSuccess);

    float *pval = vals_.dptr_;
    float *pnew_val = new_vals_.dptr_;
    for (int j = 0; j <= key_size; ++j) {
      blur<key_size><<<dim3((n_keys_-1)/lblock_+1, 1, 1), dim3(lblock_, 1, 1), 0, stream>>>(
        table, val_size_, j, pval, pnew_val, matrix_.dptr_);
      CHECK_EQ(cudaGetLastError(), cudaSuccess);
      std::swap(pval, pnew_val);
    }

    if (param_.normalize) {
      slice<key_size, true><<<dim3(nblock_, 1, 1), dim3(lblock_, 1, 1), 0, stream>>>(
        table, n_elements_, val_size_, pval, out.dptr_ + i*data_size_*n_elements_, matrix_.dptr_);
    } else {
      slice<key_size, false><<<dim3(nblock_, 1, 1), dim3(lblock_, 1, 1), 0, stream>>>(
        table, n_elements_, val_size_, pval, out.dptr_ + i*data_size_*n_elements_, matrix_.dptr_);
    }
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
  }
}

template<int key_size>
void CuPermutohedralOp<key_size>::Backward(const OpContext &ctx,
                                           const std::vector<TBlob> &out_grad,
                                           const std::vector<TBlob> &in_data,
                                           const std::vector<TBlob> &out_data,
                                           const std::vector<OpReqType> &req,
                                           const std::vector<TBlob> &in_grad,
                                           const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace permutohedral;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  cudaStream_t stream = Stream<gpu>::GetStream(s);

  Tensor<gpu, 1, float> scale = aux_args[kScale].get<gpu, 1, float>(s);

  Shape<3> shape = Shape3(batch_size_, data_size_, n_elements_); 
  Tensor<gpu, 3, float> ograd = out_grad[kOut].get_with_shape<gpu, 3, float>(shape, s);
  Tensor<gpu, 3, float> data_grad = in_grad[kData].get_with_shape<gpu, 3, float>(shape, s);
  shape[1] = key_size;
  Tensor<gpu, 3, float> pos = in_data[kPos].get_with_shape<gpu, 3, float>(shape, s);

  GetTempSpace(ctx);

  CuHashTable<key_size> table(n_keys_, entries_.dptr_, keys_.dptr_);

  for (int i = 0; i < batch_size_; ++i) {
    entries_ = -1;
    vals_ = 0;
    
    init<key_size><<<dim3(nblock_, 1, 1), dim3(lblock_,1,1), 0, stream>>>(
      table, n_elements_, pos.dptr_ + i*key_size*n_elements_, scale.dptr_, matrix_.dptr_);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    if (param_.normalize) {
      splat<key_size, true><<<dim3(1, (n_elements_-1)/(lblock_/(key_size+1))+1, 1), dim3(key_size+1, lblock_/(key_size+1), 1), 0, stream>>>(
        table, n_elements_, val_size_, ograd.dptr_ + i*data_size_*n_elements_, vals_.dptr_, matrix_.dptr_);
    } else {
      splat<key_size, false><<<dim3(1, (n_elements_-1)/(lblock_/(key_size+1))+1, 1), dim3(key_size+1, lblock_/(key_size+1), 1), 0, stream>>>(
        table, n_elements_, val_size_, ograd.dptr_ + i*data_size_*n_elements_, vals_.dptr_, matrix_.dptr_);
    }
    CHECK_EQ(cudaGetLastError(), cudaSuccess);

    float *pval = vals_.dptr_;
    float *pnew_val = new_vals_.dptr_;
    for (int j = 0; j <= key_size; ++j) {
      blur<key_size><<<dim3((n_keys_-1)/lblock_+1, 1, 1), dim3(lblock_, 1, 1), 0, stream>>>(
        table, val_size_, j, pval, pnew_val, matrix_.dptr_);
      CHECK_EQ(cudaGetLastError(), cudaSuccess);
      std::swap(pval, pnew_val);
    }

    if (param_.normalize) {
      slice<key_size, true><<<dim3(nblock_, 1, 1), dim3(lblock_, 1, 1), 0, stream>>>(
        table, n_elements_, val_size_, pval, data_grad.dptr_ + i*data_size_*n_elements_, matrix_.dptr_);
    } else {
      slice<key_size, false><<<dim3(nblock_, 1, 1), dim3(lblock_, 1, 1), 0, stream>>>(
        table, n_elements_, val_size_, pval, data_grad.dptr_ + i*data_size_*n_elements_, matrix_.dptr_);
    }
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
  }
}

  
template<>
Operator *CreateOp<gpu>(PermutohedralParam param, int key_size) {
  switch (key_size) {
   case 2: return new CuPermutohedralOp<2>(param);
   case 3: return new CuPermutohedralOp<3>(param);
   case 4: return new CuPermutohedralOp<4>(param);
   case 5: return new CuPermutohedralOp<5>(param);
   case 6: return new CuPermutohedralOp<6>(param);
   case 7: return new CuPermutohedralOp<7>(param);
   case 8: return new CuPermutohedralOp<8>(param);
   case 9: return new CuPermutohedralOp<9>(param);
   case 10: return new CuPermutohedralOp<10>(param);
   case 11: return new CuPermutohedralOp<11>(param);
   case 12: return new CuPermutohedralOp<12>(param);
   case 13: return new CuPermutohedralOp<13>(param);
   case 14: return new CuPermutohedralOp<14>(param);
   case 15: return new CuPermutohedralOp<15>(param);
   case 16: return new CuPermutohedralOp<16>(param);
   default:
    LOG(FATAL) << "GPU not supported";
    return NULL;
  }
}

}  // namespace op
}  // namespace mxnet

