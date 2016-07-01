/*!
 * Copyright (c) 2016 by Contributors
 * \file permutohedral.cc
 * \brief
 * \author Junyuan Xie
*/

#include "./permutohedral-inl.h"

namespace mxnet {
namespace op {

void PermutohedralOp::Forward(const OpContext &ctx,
                                   const std::vector<TBlob> &in_data,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<TBlob> &out_data,
                                   const std::vector<TBlob> &aux_args)  {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<cpu> *s = ctx.get_stream<cpu>();
  Tensor<cpu, 4> val = in_data[permutohedral::kData].get<cpu, 4, real_t>(s);
  Tensor<cpu, 4> pos = in_data[permutohedral::kPos].get<cpu, 4, real_t>(s);
  Tensor<cpu, 4> out = out_data[permutohedral::kOut].get<cpu, 4, real_t>(s);
  int vstride = val.shape_.Size() / val.shape_[0];
  int pstride = pos.shape_.Size() / pos.shape_[0];
  for (index_t i = 0; i < val.shape_[0]; ++i) {
    lattice_.init(pos.dptr_ + i*pstride, pos.shape_[1], pstride/pos.shape_[1]);
    lattice_.compute(out.dptr_ + i*vstride, val.dptr_ + i*vstride, val.shape_[1]);
  }
}

void PermutohedralOp::Backward(const OpContext &ctx,
                                    const std::vector<TBlob> &out_grad,
                                    const std::vector<TBlob> &in_data,
                                    const std::vector<TBlob> &out_data,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<TBlob> &in_grad,
                                    const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<cpu> *s = ctx.get_stream<cpu>();
  Tensor<cpu, 4> ograd = out_grad[permutohedral::kOut].get<cpu, 4, real_t>(s);
  Tensor<cpu, 4> pos = in_data[permutohedral::kPos].get<cpu, 4, real_t>(s);
  Tensor<cpu, 4> igrad = in_grad[permutohedral::kData].get<cpu, 4, real_t>(s);
  int vstride = ograd.shape_.Size() / ograd.shape_[0];
  int pstride = pos.shape_.Size() / pos.shape_[0];
  for (index_t i = 0; i < ograd.shape_[0]; ++i) {
    lattice_.init(pos.dptr_ + i*pstride, pos.shape_[1], pstride/pos.shape_[1]);
    lattice_.compute(igrad.dptr_ + i*vstride, ograd.dptr_ + i*vstride, ograd.shape_[1]);
  }
}

template<>
Operator *CreateOp<cpu>(PermutohedralParam param, int key_size) {
  return new PermutohedralOp(param);
}

Operator *PermutohedralProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                              std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_shape->at(1)[1]);
}

DMLC_REGISTER_PARAMETER(PermutohedralParam);

MXNET_REGISTER_OP_PROPERTY(Permutohedral, PermutohedralProp)
.describe("Apply batch normalization to input.")
.add_argument("data", "Symbol", "Input data to batch normalization")
.add_argument("pos", "Symbol", "Input data to batch normalization")
.add_arguments(PermutohedralParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

