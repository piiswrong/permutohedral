/*!
 * Copyright (c) 2016 by Contributors
 * \file permutohedral-inl.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MXNET_OPERATOR_PERMUTOHEDRAL_INL_H_
#define MXNET_OPERATOR_PERMUTOHEDRAL_INL_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../../src/operator/mshadow_op.h"
#include "../../src/operator/operator_common.h"
#include "modified_permutohedral.h"
#include "cu_hash_table.h"

namespace mxnet {
namespace op {

namespace permutohedral {
enum PermutohedralOpInputs {kData, kPos};
enum PermutohedralOpOutputs {kOut, kNorm};
enum PermutohedralOpTemps {kTemp};
enum PermutohedralOpAuxs {kScale};

struct Pair{
  int32_t index;
  float weight;
};
}  // namespace blockgrad

struct PermutohedralParam : public dmlc::Parameter<PermutohedralParam> {
  bool normalize;
  DMLC_DECLARE_PARAMETER(PermutohedralParam) {
    DMLC_DECLARE_FIELD(normalize).set_default(false)
    .describe("normalize output");
  }
};

class PermutohedralOp : public Operator {
 public:
  explicit PermutohedralOp(PermutohedralParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args);

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args);
 private:
  PermutohedralParam param_;
  permutohedral::ModifiedPermutohedral lattice_;
};  // class PermutohedralOp

#if defined(__CUDACC__)
template<int key_size>
class CuPermutohedralOp : public Operator {
 public:
  explicit CuPermutohedralOp(PermutohedralParam param) {
    this->param_ = param;
    init_ = false;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args);

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args);
 private:
  bool init_;
  PermutohedralParam param_;
  permutohedral::ModifiedPermutohedral lattice_;

  int batch_size_, data_size_, val_size_, n_elements_, n_keys_, lblock_, nblock_;
  mshadow::Tensor<gpu, 1, int32_t> entries_;
  mshadow::Tensor<gpu, 2, int16_t> keys_;
  mshadow::Tensor<gpu, 2, float> vals_, new_vals_;
  mshadow::Tensor<gpu, 1, permutohedral::Pair> matrix_;
  void GetTempSpace(const OpContext &ctx, int val_size);
  void Filter(cudaStream_t stream, permutohedral::CuHashTable<key_size> table, bool normalize, int val_size,
              float *scale, float *data, float *pos, float *out, float *norm);

};  // class PermutohedralOp
#endif  // __CUDACC__

template<typename xpu>
Operator *CreateOp(PermutohedralParam param, int key_size);

#if DMLC_USE_CXX11
class PermutohedralProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "pos"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"bias"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "norm"};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2);
    TShape val_shape = in_shape->at(permutohedral::kData);
    TShape pos_shape = in_shape->at(permutohedral::kPos);
    CHECK_EQ(val_shape.ndim(), 4);
    CHECK_EQ(pos_shape.ndim(), 4);
    CHECK_EQ(val_shape[0], pos_shape[0]);
    CHECK_EQ(val_shape[2], pos_shape[2]);
    CHECK_EQ(val_shape[3], pos_shape[3]);
    out_shape->clear();
    out_shape->push_back(val_shape);
    val_shape[1] = 1;
    out_shape->push_back(val_shape);
    aux_shape->clear();
    aux_shape->push_back(Shape1(pos_shape[1]));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new PermutohedralProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Permutohedral";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[permutohedral::kOut],
            in_data[permutohedral::kData],
            in_data[permutohedral::kPos],
            out_data[permutohedral::kOut],
            out_data[permutohedral::kNorm]
           };
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;


 private:
  PermutohedralParam param_;
};  // class PermutohedralProperty

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_PERMUTOHEDRAL_INL_H_
